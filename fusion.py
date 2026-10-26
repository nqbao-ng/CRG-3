"""Per-utterance hyperedge fusion with private and shared linear projections.

Each utterance is one hyperedge containing exactly its three modality nodes.
This grouped incidence is implicit in the [B,N,3,d] tensor; no dense matrix
or extra temporal edge is needed. rho measures learned modality contribution.
F_m = W_private_m H_m + W_shared c; no extra norm, confidence or auxiliary head.
"""
import torch
import torch.nn as nn

from .graph import MODALITIES


class HyperedgeFusion(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.projection = nn.ModuleDict({m: nn.Linear(hidden, hidden) for m in MODALITIES})
        self.score = nn.ModuleDict({m: nn.Linear(hidden, 1) for m in MODALITIES})
        # Bias-free: these two paths implement the matrix products in the formula.
        self.private_projection = nn.ModuleDict({m: nn.Linear(hidden, hidden, bias=False)
                                                 for m in MODALITIES})
        self.shared_projection = nn.Linear(hidden, hidden, bias=False)

    def forward(self, features, valid):
        nodes = torch.stack([self.projection[m](features[m]) for m in MODALITIES], dim=-2)
        scores = torch.cat([self.score[m](nodes[..., k, :])
                            for k, m in enumerate(MODALITIES)], dim=-1)
        # Compute the modality softmax and weighted sum in float32 under AMP.
        weights = scores.float().softmax(-1)
        shared = (weights.unsqueeze(-1) * nodes.float()).sum(-2)
        shared = shared * valid.unsqueeze(-1)
        feedback = self.shared_projection(shared.to(nodes.dtype))
        result = {
            m: (self.private_projection[m](features[m]) + feedback) * valid.unsqueeze(-1)
            for m in MODALITIES
        }
        return result, {
            "modality_weights": weights * valid.unsqueeze(-1),
            "shared_feature": shared,
            "shared_update": feedback * valid.unsqueeze(-1),
        }
