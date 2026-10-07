"""Causal Diff-then-Same Graph Mamba encoder with source-only messages."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaBlock


MODALITIES = ("t", "a", "v")
EDGE_WEIGHT_MODES = ("none", "dot_tanh", "dot_softmax", "vuemo_tanh")


def batched_gather(H, idx):
    """H: [B, N, d], idx: [B, N, K] -> [B, N, K, d]."""
    B, N, K = idx.shape
    flat = idx.reshape(B, N * K, 1).expand(-1, -1, H.shape[-1])
    return H.gather(1, flat).reshape(B, N, K, H.shape[-1])


class EdgeWeight(nn.Module):
    """Score existing causal edges from the frozen, stage-normalized features.

    VUEMO's affine concatenation score is evaluated as two node-wise scalars
    rather than materializing [target; source] for every edge. Only its scorer
    is adapted here, not its degree normalization or graph convolution.
    """

    def __init__(self, d, mode, rank=32):
        super().__init__()
        if mode not in EDGE_WEIGHT_MODES or mode == "none":
            raise ValueError(f"Unknown enabled edge weight mode: {mode}")
        if rank < 1:
            raise ValueError("edge_weight_dim must be positive")
        self.mode = mode
        if mode.startswith("dot_"):
            self.query = nn.Linear(d, rank, bias=False)
            self.key = nn.Linear(d, rank, bias=False)
            if mode == "dot_tanh":
                self.bias = nn.Parameter(torch.zeros(()))
            self.scale = rank ** -0.5
        else:
            self.gate = nn.Linear(2 * d, 1)

    def forward(self, H, idx, mask):
        if self.mode.startswith("dot_"):
            q = self.query(H).float().unsqueeze(2)
            k = batched_gather(self.key(H), idx.clamp_min(0)).float()
            score = (q * k).sum(-1) * self.scale
            if self.mode == "dot_tanh":
                score = score + self.bias.float()
        else:
            # Exactly equivalent to gate(cat([H_i, H_j])), with O(Nd+E)
            # scoring instead of an [B,N,K,2d] concatenation.
            d = H.shape[-1]
            target = F.linear(H, self.gate.weight[:, :d], self.gate.bias)
            source = F.linear(H, self.gate.weight[:, d:])
            score = (target.unsqueeze(2) + batched_gather(source, idx.clamp_min(0))).squeeze(-1).float()
        if self.mode == "dot_softmax":
            # Empty relations must yield zero, never softmax([-inf,...]) NaNs.
            safe_score = score.masked_fill(~mask, -torch.inf)
            safe_score = torch.where(mask.any(-1, keepdim=True), safe_score, torch.zeros_like(safe_score))
            weights = torch.softmax(safe_score, dim=-1).masked_fill(~mask, 0)
        else:
            weights = torch.tanh(score).masked_fill(~mask, 0)
        return weights


class RelationScan(nn.Module):
    """Read ordered source messages and one target read token for a relation."""

    def __init__(self, d, mamba_kw, message_mode="projection", edge_weight_mode="none", edge_weight_dim=32):
        super().__init__()
        if message_mode not in ("projection", "embedding"):
            raise ValueError(f"Unknown message mode: {message_mode}")
        self.message_proj = nn.Linear(d, d, bias=False) if message_mode == "projection" else None
        self.relation_embedding = nn.Parameter(torch.zeros(d)) if message_mode == "embedding" else None
        self.read_proj = nn.Linear(d, d)
        self.read_embedding = nn.Parameter(torch.zeros(d))
        self.mamba = MambaBlock(d, **mamba_kw)
        if edge_weight_mode not in EDGE_WEIGHT_MODES:
            raise ValueError(f"Unknown edge weight mode: {edge_weight_mode}")
        self.edge_weight = (EdgeWeight(d, edge_weight_mode, edge_weight_dim)
                            if edge_weight_mode != "none" else None)

    def forward(self, H, table, shared_message_proj=None):
        """All messages use the frozen stage input H; no node changes during a scan."""
        idx, mask = table["idx"], table["mask"]
        B, N, K = idx.shape
        d = H.shape[-1]

        # Project each source once; the same message can be used by many targets.
        projection = self.message_proj if self.message_proj is not None else shared_message_proj
        if projection is None:
            raise ValueError("Embedding message mode requires a shared message projection")
        messages = batched_gather(projection(H), idx.clamp_min(0))
        if self.relation_embedding is not None:
            messages = messages + self.relation_embedding
        messages = messages * mask.unsqueeze(-1).to(messages.dtype)

        # Valid messages are packed chronologically; read(H_i) follows the last one.
        lengths = mask.sum(-1)
        weights = self.edge_weight(H, idx, mask) if self.edge_weight is not None else None
        read = self.read_proj(H) + self.read_embedding
        seq = torch.cat([messages, messages.new_zeros(B, N, 1, d)], dim=2)
        seq = seq + F.one_hot(lengths, K + 1).to(seq.dtype).unsqueeze(-1) * read.unsqueeze(2)
        token_weights = None
        if weights is not None:
            token_weights = torch.cat([weights, weights.new_zeros(B, N, 1)], dim=2)
            token_weights = token_weights + F.one_hot(lengths, K + 1).to(weights.dtype)

        flat_len = lengths.reshape(-1)
        rows = flat_len.gt(0).nonzero(as_tuple=True)[0]
        context = H.new_zeros(B * N, d)
        if rows.numel() > 0:
            inputs = seq.reshape(B * N, K + 1, d).index_select(0, rows)
            if token_weights is None:
                out = self.mamba(inputs)
            else:
                out = self.mamba(inputs, token_weights=token_weights.reshape(B * N, K + 1).index_select(0, rows))
            read_out = out[torch.arange(rows.numel(), device=H.device), flat_len.index_select(0, rows)]
            context = context.index_copy(0, rows, read_out.to(context.dtype))
        diag = {"neighbor_count": lengths.detach()}
        if weights is not None:
            diag["edge_weights"] = weights.detach()
        return context.reshape(B, N, d), diag


class GateUpdate(nn.Module):
    """Update a target from its original vector and one relation context."""

    def __init__(self, d, dropout):
        super().__init__()
        self.gate = nn.Linear(2 * d, d)
        self.context_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node, context, valid_mask):
        g = torch.sigmoid(self.gate(torch.cat([node, context], dim=-1)))
        out = node + self.dropout(g * self.context_proj(context))
        return out * valid_mask.unsqueeze(-1).to(out.dtype)


class RelationStage(nn.Module):
    def __init__(self, d, dropout, mamba_kw, message_mode="projection", edge_weight_mode="none", edge_weight_dim=32):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d)
        self.scan = RelationScan(d, mamba_kw, message_mode, edge_weight_mode, edge_weight_dim)
        self.update = GateUpdate(d, dropout)

    def forward(self, H, table, valid_mask, shared_message_proj=None):
        context, diag = self.scan(self.pre_norm(H), table, shared_message_proj)
        return self.update(H, context, valid_mask), diag


class GraphLayer(nn.Module):
    """Every Diff context reads H; every Same context reads Diff-updated U."""

    def __init__(self, cfg, mamba_kw):
        super().__init__()
        self.shared_message_proj = (nn.Linear(cfg.hidden, cfg.hidden, bias=False)
                                    if cfg.message_mode == "embedding" else None)
        self.diff = RelationStage(cfg.hidden, cfg.dropout, mamba_kw, cfg.message_mode,
                                  cfg.edge_weight_mode, cfg.edge_weight_dim)
        self.same = RelationStage(cfg.hidden, cfg.dropout, mamba_kw, cfg.message_mode,
                                  cfg.edge_weight_mode, cfg.edge_weight_dim)

    def forward(self, H, diff_table, same_table, valid_mask):
        U, diag_d = self.diff(H, diff_table, valid_mask, self.shared_message_proj)
        H_next, diag_s = self.same(U, same_table, valid_mask, self.shared_message_proj)
        return H_next, {"diff": diag_d, "same": diag_s}


class GraphEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mamba_kw = dict(d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand)
        self.layers = nn.ModuleList([
            nn.ModuleDict({m: GraphLayer(cfg, mamba_kw) for m in MODALITIES})
            for _ in range(cfg.graph_layers)
        ])
        self.final_norm = nn.ModuleDict({m: nn.LayerNorm(cfg.hidden) for m in MODALITIES})

    def forward(self, H, diff_table, same_table, valid_mask):
        """H: {modality: [B, N, d]} -> Q_G: {modality: [B, N, d]}."""
        diagnostics = {}
        for l, layer in enumerate(self.layers):
            H_next = {}
            for m in MODALITIES:
                H_next[m], diagnostics[(l, m)] = layer[m](
                    H[m], diff_table, same_table, valid_mask)
            H = H_next
        mask = valid_mask.unsqueeze(-1)
        Q = {m: self.final_norm[m](H[m]) * mask.to(H[m].dtype) for m in MODALITIES}
        return Q, diagnostics
