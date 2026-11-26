"""Post-CRG modality-anchored attention."""

import torch
from torch import nn

from .graph import MODALITIES


class ModalityAnchoredUpdate(nn.Module):
    """Score same-turn peers against a learned self score."""

    def __init__(self, hidden, *, target, sources=None):
        super().__init__()
        self.target = target
        self.sources = tuple(m for m in MODALITIES if m != target) if sources is None else sources
        self.scored_modalities = tuple(m for m in MODALITIES if m == target or m in self.sources)
        rank = max(1, hidden // 4)
        self.score_scale = rank ** -0.5
        self.relation = nn.ParameterDict({
            m: nn.Parameter(torch.zeros(hidden)) for m in self.sources
        })
        self.query = nn.Linear(hidden, rank, bias=False)
        self.keys = nn.ModuleDict({m: nn.Linear(hidden, rank, bias=False)
                                   for m in self.scored_modalities})
        self.output = nn.Linear(len(self.sources) * hidden, hidden, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, features):
        target = features[self.target]
        candidates = {self.target: target}
        for source, relation in self.relation.items():
            feature = features[source]
            scale = 1 + relation.tanh()
            candidates[source] = feature * scale.to(feature.dtype)
        query = self.query(target).float().unsqueeze(-2)
        keys = torch.stack([self.keys[m](candidates[m]) for m in self.scored_modalities], dim=-2)
        scores = (query * keys.float()).sum(-1) * self.score_scale
        weights = dict(zip(self.scored_modalities, scores.softmax(dim=-1).unsqueeze(-1).unbind(dim=-2)))
        messages = torch.cat([weights[m].to(candidates[m].dtype) * candidates[m] for m in self.sources], dim=-1)
        return target + self.output(messages)
