"""Original baseline. Weight names/order are preserved for old checkpoints."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import MambaBlock

MODALITIES = ("t", "a", "v")


def batched_gather(H, idx):
    B, N, K = idx.shape
    flat = idx.reshape(B, N * K, 1).expand(-1, -1, H.shape[-1])
    return H.gather(1, flat).reshape(B, N, K, H.shape[-1])


class RelationScan(nn.Module):
    def __init__(self, d, mamba_kw, message_mode="embedding", update_mode="baseline",
                 dropout=0.2, use_gate=True):
        super().__init__()
        if update_mode != "baseline":
            raise ValueError("RelationScan implements the original baseline only")
        if message_mode not in ("projection", "embedding"):
            raise ValueError(f"Unknown message mode: {message_mode}")
        self.message_proj = nn.Linear(d, d, bias=False) if message_mode == "projection" else None
        self.relation_embedding = nn.Parameter(torch.zeros(d)) if message_mode == "embedding" else None
        self.read_proj = nn.Linear(d, d)
        self.read_embedding = nn.Parameter(torch.zeros(d))
        self.mamba = MambaBlock(d, **mamba_kw)

    def forward(self, H, table, shared_message_proj=None):
        idx, mask = table["idx"], table["mask"]
        B, N, K = idx.shape
        d = H.shape[-1]
        projection = self.message_proj if self.message_proj is not None else shared_message_proj
        if projection is None:
            raise ValueError("Embedding messages require a shared message projection")
        messages = batched_gather(projection(H), idx.clamp_min(0))
        if self.relation_embedding is not None:
            messages = messages + self.relation_embedding
        messages = messages * mask.unsqueeze(-1).to(messages.dtype)
        lengths = mask.sum(-1)
        read = self.read_proj(H) + self.read_embedding
        seq = torch.cat([messages, messages.new_zeros(B, N, 1, d)], dim=2)
        seq = seq + F.one_hot(lengths, K + 1).to(seq.dtype).unsqueeze(-1) * read.unsqueeze(2)
        flat_len = lengths.reshape(-1)
        rows = flat_len.gt(0).nonzero(as_tuple=True)[0]
        context = H.new_zeros(B * N, d)
        if rows.numel():
            inputs = seq.reshape(B * N, K + 1, d).index_select(0, rows)
            out = self.mamba(inputs)
            read_out = out[torch.arange(rows.numel(), device=H.device), flat_len[rows]]
            context = context.index_copy(0, rows, read_out.to(context.dtype))
        return context.reshape(B, N, d), {"neighbor_count": lengths.detach()}


class GateUpdate(nn.Module):
    def __init__(self, d, dropout, use_gate=True):
        super().__init__()
        self.gate = nn.Linear(2 * d, d) if use_gate else None
        self.context_proj = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def delta(self, node, context):
        update = self.context_proj(context)
        if self.gate is not None:
            update = torch.sigmoid(self.gate(torch.cat([node, context], -1))) * update
        return self.dropout(update)

    def forward(self, node, context, valid_mask):
        out = node + self.delta(node, context)
        return out * valid_mask.unsqueeze(-1).to(out.dtype)


class RelationStage(nn.Module):
    def __init__(self, d, dropout, mamba_kw, message_mode="embedding",
                 update_mode="baseline", use_gate=True):
        super().__init__()
        self.pre_norm = nn.LayerNorm(d)
        self.scan = RelationScan(d, mamba_kw, message_mode, update_mode)
        self.update = GateUpdate(d, dropout, use_gate)

    def forward(self, H, table, valid_mask, shared_message_proj=None):
        context, diag = self.scan(self.pre_norm(H), table, shared_message_proj)
        return self.update(H, context, valid_mask), diag


class GraphLayer(nn.Module):
    def __init__(self, cfg, mamba_kw):
        super().__init__()
        if cfg.graph_update_mode != "baseline":
            raise ValueError("Use AttentionGraphEncoder for attention modes")
        self.shared_message_proj = nn.Linear(cfg.hidden, cfg.hidden, bias=False)
        self.diff = RelationStage(cfg.hidden, cfg.dropout, mamba_kw, cfg.message_mode)
        self.same = RelationStage(cfg.hidden, cfg.dropout, mamba_kw, cfg.message_mode)

    def forward(self, H, diff_table, same_table, valid_mask):
        U, diag_d = self.diff(H, diff_table, valid_mask, self.shared_message_proj)
        out, diag_s = self.same(U, same_table, valid_mask, self.shared_message_proj)
        return out, {"diff": diag_d, "same": diag_s}


class GraphEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        mamba_kw = dict(d_state=cfg.d_state, d_conv=cfg.d_conv, expand=cfg.expand,
                        use_rmsnorm=True)
        self.layers = nn.ModuleList([
            nn.ModuleDict({m: GraphLayer(cfg, mamba_kw) for m in MODALITIES})
            for _ in range(cfg.graph_layers)
        ])
        self.final_norm = nn.ModuleDict({m: nn.LayerNorm(cfg.hidden) for m in MODALITIES})

    def forward(self, H, diff_table, same_table, valid_mask):
        diagnostics = {}
        for l, layer in enumerate(self.layers):
            H_next = {}
            for m in MODALITIES:
                H_next[m], diagnostics[(l, m)] = layer[m](H[m], diff_table, same_table, valid_mask)
            H = H_next
        mask = valid_mask.unsqueeze(-1)
        return {m: self.final_norm[m](H[m]) * mask.to(H[m].dtype) for m in MODALITIES}, diagnostics
