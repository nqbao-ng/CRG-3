"""Target-conditioned cross-modal messages -> ordered selective scan -> GateUpdate.

joint: softmax across (history position, source modality).
per_modality: softmax over history independently per source modality, then /3.
Both retain one message per history utterance; neither pools time before scan.
Diff -> Same is synchronous: all Same sources read the completed Diff outputs.
history uses one chronological stage and no speaker-relation parameters.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .graph import MODALITIES, GateUpdate
from .mamba import RMSNorm, SelectiveStateSpace


def attention_messages(query, keys, values, mask, normalization):
    """[rows,d], [rows,K,3,d], [rows,K] -> messages [rows,K,d], effective weights.

    Empty rows have exactly zero weights and finite gradients. Effective weights
    sum to one across K*3 for either normalization (zero for empty rows).
    """
    scores = (query[:, None, None].float() * keys.float()).sum(-1) / math.sqrt(query.shape[-1])
    expanded_mask = mask.unsqueeze(-1).expand_as(scores)
    scores = scores.masked_fill(~expanded_mask, -torch.inf)
    scores = torch.where(mask.any(-1)[:, None, None], scores, torch.zeros_like(scores))
    if normalization == "joint":
        weights = scores.flatten(1).softmax(-1).reshape_as(scores)
    elif normalization == "per_modality":
        weights = scores.softmax(dim=1) / len(MODALITIES)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    weights = weights.masked_fill(~expanded_mask, 0)
    messages = (weights.to(values.dtype).unsqueeze(-1) * values).sum(dim=2)
    return messages, weights


def merge_history(diff, same, window):
    """Merge disjoint Same/Diff lists into the original chronological history."""
    idx = torch.cat([diff["idx"], same["idx"]], -1)
    mask = torch.cat([diff["mask"], same["mask"]], -1)
    sentinel = idx.shape[1]
    idx = idx.masked_fill(~mask, sentinel).sort(-1).values
    # Canonical width independent of how speakers partitioned the same history.
    width = max(1, min(window, idx.shape[1] - 1))
    idx = F.pad(idx, (0, max(0, width - idx.shape[-1])), value=sentinel)[..., :width]
    mask = idx.ne(sentinel)
    return {"idx": idx.masked_fill(~mask, -1), "mask": mask}


class AttentionScanBlock(nn.Module):
    """RMSNorm -> Linear -> SiLU -> selective SSM -> gate -> Linear, plus token residual.

    There is no Conv1d: the graph attention interaction replaces local convolution.
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm = RMSNorm(cfg.hidden)
        self.in_proj = nn.Linear(cfg.hidden, 2 * cfg.expand * cfg.hidden, bias=False)
        self.ssm = SelectiveStateSpace(cfg.hidden, cfg.d_state, cfg.expand)
        self.out_proj = nn.Linear(cfg.expand * cfg.hidden, cfg.hidden, bias=False)

    def forward(self, sequence):
        u, gate = self.in_proj(self.norm(sequence)).chunk(2, -1)
        y = self.ssm(F.silu(u)) * F.silu(gate)
        return sequence + self.out_proj(y)


class AttentionRelationStage(nn.Module):
    def __init__(self, cfg, normalization, relation):
        super().__init__()
        d = cfg.hidden
        self.normalization = normalization
        self.chunk_size = cfg.attention_chunk_size
        self.use_checkpoint = cfg.attention_checkpoint
        self.pre_norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.query = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        self.key = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        self.value = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        # Separate stage modules encode Diff/Same; history deliberately has no relation embedding.
        self.relation_embedding = nn.Parameter(torch.zeros(d)) if relation != "history" else None
        self.read_proj = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})
        self.read_embedding = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in MODALITIES})
        self.scan = nn.ModuleDict({m: AttentionScanBlock(cfg) for m in MODALITIES})
        self.update = nn.ModuleDict({m: GateUpdate(d, cfg.dropout) for m in MODALITIES})

    def _scan_rows(self, query, keys, values, indices, mask, read, modality):
        """Gather within checkpoint so large source tensors are not retained per chunk."""
        gathered_keys = keys[indices]
        gathered_values = values[indices]
        messages, _ = attention_messages(query, gathered_keys, gathered_values, mask, self.normalization)
        lengths = mask.sum(-1)
        K = mask.shape[-1]
        sequence = F.pad(messages, (0, 0, 0, 1))
        sequence = sequence + F.one_hot(lengths, K + 1).to(sequence.dtype).unsqueeze(-1) * read[:, None]
        out = self.scan[modality](sequence)
        return out[torch.arange(len(out), device=out.device), lengths]

    def forward(self, H, table, valid):
        B, N, d = H["t"].shape
        idx, mask = table["idx"], table["mask"]
        lengths = mask.sum(-1)
        rows = (lengths.gt(0) & valid).reshape(-1).nonzero(as_tuple=True)[0]
        normed = {m: self.pre_norm[m](H[m]) for m in MODALITIES}
        keys = torch.stack([self.key[m](normed[m]) for m in MODALITIES], dim=2).reshape(B * N, 3, d)
        values = torch.stack([self.value[m](normed[m]) for m in MODALITIES], dim=2).reshape(B * N, 3, d)
        if self.relation_embedding is not None:
            values = values + self.relation_embedding
        # Tables are padded chronologically. Convert dialogue-local indices to flat indices.
        offsets = torch.arange(B, device=idx.device)[:, None, None] * N
        source_indices = (idx.clamp_min(0) + offsets).reshape(B * N, -1)
        source_masks = mask.reshape(B * N, -1)
        result = {}
        for m in MODALITIES:
            queries = self.query[m](normed[m]).reshape(B * N, d)
            read = (self.read_proj[m](normed[m]) + self.read_embedding[m]).reshape(B * N, d)
            chunks = []
            for chunk in rows.split(self.chunk_size):
                inputs = (queries[chunk], keys, values, source_indices[chunk], source_masks[chunk], read[chunk], m)
                if self.use_checkpoint and self.training and torch.is_grad_enabled():
                    context = checkpoint(self._scan_rows, *inputs, use_reentrant=False)
                else:
                    context = self._scan_rows(*inputs)
                chunks.append(context)
            context = H[m].new_zeros(B * N, d)
            if chunks:
                context = context.index_copy(0, rows, torch.cat(chunks).to(context.dtype))
            result[m] = self.update[m](H[m], context.reshape(B, N, d), valid)
        return result, {"neighbor_count": lengths.detach()}


class AttentionGraphLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.history_only = cfg.graph_update_mode == "attention_history"
        self.window = cfg.local_window
        normalization = "per_modality" if cfg.graph_update_mode == "attention_per_modality" else "joint"
        if self.history_only:
            self.history = AttentionRelationStage(cfg, normalization, "history")
        else:
            self.diff = AttentionRelationStage(cfg, normalization, "diff")
            self.same = AttentionRelationStage(cfg, normalization, "same")

    def forward(self, H, diff, same, valid):
        if self.history_only:
            out, diag = self.history(H, merge_history(diff, same, self.window), valid)
            return out, {"history": diag}
        U, diag_d = self.diff(H, diff, valid)
        out, diag_s = self.same(U, same, valid)
        return out, {"diff": diag_d, "same": diag_s}


class AttentionGraphEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([AttentionGraphLayer(cfg) for _ in range(cfg.graph_layers)])
        self.final_norm = nn.ModuleDict({m: nn.LayerNorm(cfg.hidden) for m in MODALITIES})

    def forward(self, H, diff, same, valid):
        diagnostics = {}
        for index, layer in enumerate(self.layers):
            H, diagnostics[index] = layer(H, diff, same, valid)
        mask = valid.unsqueeze(-1)
        return {m: self.final_norm[m](H[m]) * mask.to(H[m].dtype) for m in MODALITIES}, diagnostics
