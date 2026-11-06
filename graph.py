"""Shared modality names and residual context update for the CRG encoder."""
import torch
import torch.nn as nn

MODALITIES = ("t", "a", "v")


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

