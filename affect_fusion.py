"""Utterance-local fusion for CRG modality features."""

from math import log

import torch
from torch import nn

from .graph import MODALITIES


def _make_gmu_gates(hidden, *, initial_bias=log(9.0)):
    gates = nn.ModuleDict({m: nn.Linear(2 * hidden, hidden) for m in MODALITIES})
    for gate in gates.values():
        nn.init.zeros_(gate.weight)
        nn.init.constant_(gate.bias, initial_bias)
    return gates


def _gmu_blend(gate, base, candidate):
    keep = gate(torch.cat((base, candidate), dim=-1)).sigmoid().to(base.dtype)
    return keep * base + (1.0 - keep) * candidate


class PairwiseGMUFusion(nn.Module):
    """Blend each CRG output with a candidate built from the other two modalities."""

    def __init__(self, hidden, *, use_product=True):
        super().__init__()
        rank = max(1, hidden // 4) if use_product else 0
        self.interaction_projections = None
        if use_product:
            self.interaction_projections = nn.ModuleDict({
                m: nn.Linear(hidden, rank, bias=False) for m in MODALITIES
            })
        self.candidate_projections = nn.ModuleDict({
            m: nn.Linear(2 * hidden + rank, hidden) for m in MODALITIES
        })
        self.gates = _make_gmu_gates(hidden)

    def forward(self, features):
        projected = None
        if self.interaction_projections is not None:
            projected = {m: self.interaction_projections[m](features[m]) for m in MODALITIES}
        updated = []
        for target in MODALITIES:
            left, right = (source for source in MODALITIES if source != target)
            inputs = (features[left], features[right])
            if projected is not None:
                inputs += (projected[left] * projected[right],)
            candidate = self.candidate_projections[target](torch.cat(inputs, dim=-1))
            updated.append(_gmu_blend(self.gates[target], features[target], candidate))
        return torch.cat(updated, dim=-1)


class TextAnchoredGMUFusion(nn.Module):
    """Build gated messages and blend modality-specific interaction candidates."""

    def __init__(self, hidden, *, pair_interaction=False):
        super().__init__()
        width = len(MODALITIES) * hidden
        self.source_gate = nn.Linear(width, width)
        self.message_projections = nn.ModuleDict({m: nn.Linear(hidden, hidden, bias=False) for m in MODALITIES})
        self.gates = _make_gmu_gates(hidden, initial_bias=0.0) if pair_interaction else _make_gmu_gates(hidden)
        self.interaction_projections = (
            nn.ModuleDict({m: nn.Linear(2 * hidden, hidden) for m in MODALITIES})
            if pair_interaction else None
        )

    def forward(self, features):
        joined = torch.cat([features[m] for m in MODALITIES], dim=-1)
        source_gates = self.source_gate(joined).sigmoid().to(joined.dtype).chunk(len(MODALITIES), dim=-1)
        projected = {
            m: self.message_projections[m](gate * features[m])
            for m, gate in zip(MODALITIES, source_gates)
        }
        if self.interaction_projections is not None:
            return self._interaction_blend(features, projected)

        messages = {m: value.tanh() for m, value in projected.items()}
        candidates = {"t": messages["a"] + messages["v"], "a": messages["t"], "v": messages["t"]}
        updated = [_gmu_blend(self.gates[m], features[m], candidates[m]) for m in MODALITIES]
        return torch.cat(updated, dim=-1)

    def _interaction_blend(self, features, projected):
        pairs = {"t": ("a", "v"), "a": ("a", "t"), "v": ("v", "t")}
        updated = []
        for target in MODALITIES:
            left, right = (projected[m] for m in pairs[target])
            interaction = torch.cat((left + right, left * right), dim=-1)
            candidate = self.interaction_projections[target](interaction)
            logit = self.gates[target](torch.cat((features[target], candidate), dim=-1))
            # At zero, select the sigmoid branch so the gate starts at 1 with a gradient.
            base = projected[target]
            scaled_keep = 2.0 * logit.sigmoid()
            keep = torch.where(logit > 0, 1.0, scaled_keep).to(base.dtype)
            updated.append(keep * base + (1.0 - keep) * candidate)
        return torch.cat(updated, dim=-1)
