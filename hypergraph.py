"""Within-utterance mean hypergraphs with node-update or fusion readout.

The same post-CRG Z supplies routing and values. Shared learned anchors determine
soft incidence P over K hyperedges for each node. Each utterance independently
writes its three current nodes, including self, into degree-normalized hyperedge
content, then reads it back to nodes or pools it with learned edge attention.
No history accumulation, confidence,
auxiliary loss, output normalization or direct Z residual/concatenation is added.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import MODALITIES


class UtteranceHypergraph(nn.Module):
    def __init__(self, hidden, num_hyperedges=4):
        super().__init__()
        if hidden < 1 or num_hyperedges < 1:
            raise ValueError("hidden and num_hyperedges must be positive")
        self.hidden = hidden
        self.num_hyperedges = num_hyperedges
        self.eps = 1e-8
        self.route = nn.ModuleDict({m: nn.Linear(hidden, hidden, bias=False) for m in MODALITIES})
        self.hyperedge_vectors = nn.Parameter(torch.empty(num_hyperedges, hidden))
        nn.init.xavier_uniform_(self.hyperedge_vectors)

    def encode_hyperedges(self, contextual, valid):
        """Shared node-to-hyperedge weighted mean; return nodes, P and E.

        P[b,i,m,k] = softmax_k((W_m Z[b,i,m]) dot e[k] / sqrt(d))
        E[b,i,k] = sum_m P[b,i,m,k] Z[b,i,m] / (eps + sum_m P[b,i,m,k])

        Only the three nodes at i participate. Anchors/route weights are shared
        over utterances and dialogues, while P/E/G are recomputed per utterance.
        FP32 aggregation is retained under mixed precision; padding is zeroed
        before routing, so even invalid NaN/Inf payloads cannot enter the graph.
        """
        nodes = torch.stack([contextual[m] for m in MODALITIES], dim=2)
        nodes = nodes.masked_fill(~valid[:, :, None, None], 0)
        routes = torch.stack([self.route[m](nodes[:, :, k])
                              for k, m in enumerate(MODALITIES)], dim=2)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            scores = F.linear(routes.float(), self.hyperedge_vectors.float()) / math.sqrt(self.hidden)
            incidence = scores.softmax(dim=-1).masked_fill(~valid[:, :, None, None], 0)
            edge_sum = torch.einsum('bnmk,bnmd->bnkd', incidence, nodes.float())
            edge_degree = incidence.sum(dim=2)
            hyperedges = edge_sum / (edge_degree.unsqueeze(-1) + self.eps)
        return nodes, incidence, hyperedges

    def forward(self, contextual, valid):
        """Direction 1: G[b,i,m] = sum_k P[b,i,m,k] E[b,i,k]."""
        nodes, incidence, hyperedges = self.encode_hyperedges(contextual, valid)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            raw = torch.einsum('bnmk,bnkd->bnmd', incidence, hyperedges)
            raw = raw.masked_fill(~valid[:, :, None, None], 0)
        result = {m: raw[:, :, k].to(nodes.dtype) for k, m in enumerate(MODALITIES)}
        return result, {'hypergraph_membership': incidence,
                        'hypergraph_content': hyperedges,
                        'hypergraph_raw': raw}


class UtteranceHypergraphFusion(UtteranceHypergraph):
    """Direction 2: pool mean hyperedges into one d-wide utterance feature.

    b[i,k] = w_f dot tanh(W_f E[i,k] + b_f)
    beta[i,:] = softmax_k(b[i,:])
    fused[i] = sum_k beta[i,k] E[i,k]

    Edge attention is a learned aggregation weight, not emotion confidence.
    No edge-to-node readback or direct contextual Z skip is computed here.
    """
    def __init__(self, hidden, num_hyperedges=4):
        super().__init__(hidden, num_hyperedges)
        self.edge_projection = nn.Linear(hidden, hidden)
        self.edge_score = nn.Linear(hidden, 1, bias=False)

    def forward(self, contextual, valid):
        nodes, incidence, hyperedges = self.encode_hyperedges(contextual, valid)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            scores = self.edge_score(torch.tanh(self.edge_projection(hyperedges))).squeeze(-1)
            beta = scores.softmax(dim=-1).masked_fill(~valid.unsqueeze(-1), 0)
            fused = (beta.unsqueeze(-1) * hyperedges).sum(dim=2)
            fused = fused.masked_fill(~valid.unsqueeze(-1), 0)
        return fused.to(nodes.dtype), {'hypergraph_membership': incidence,
                                       'hypergraph_content': hyperedges,
                                       'hypergraph_attention': beta,
                                       'hypergraph_fused': fused}
