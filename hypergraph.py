"""Within-utterance mean hypergraphs with node-update or fusion readout.

The same post-CRG Z supplies routing and values. Shared learned anchors determine
soft incidence P over K hyperedges for each node. Each utterance independently
writes its three current nodes, including self, into degree-normalized hyperedge
content, then reads it back to nodes or pools it with learned edge attention.
Optional modality classifiers supply detached max-probability confidence for
node-to-edge writes. The membership used for node readback stays unchanged.
No history accumulation, output normalization or direct Z residual is added.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import MODALITIES


class UtteranceHypergraph(nn.Module):
    def __init__(self, hidden, num_hyperedges=4, *, confidence=False, num_classes=None):
        super().__init__()
        if hidden < 1 or num_hyperedges < 1:
            raise ValueError("hidden and num_hyperedges must be positive")
        self.hidden = hidden
        self.num_hyperedges = num_hyperedges
        self.eps = 1e-8
        self.route = nn.ModuleDict({m: nn.Linear(hidden, hidden, bias=False) for m in MODALITIES})
        self.hyperedge_vectors = nn.Parameter(torch.empty(num_hyperedges, hidden))
        nn.init.xavier_uniform_(self.hyperedge_vectors)
        self.confidence_heads = None
        if confidence:
            if num_classes is None or num_classes < 2:
                raise ValueError("Confidence requires num_classes >= 2")
            # Keep common route/anchor/scorer initialization and RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                self.confidence_heads = nn.ModuleDict({
                    m: nn.Linear(hidden, num_classes) for m in MODALITIES})

    def encode_hyperedges(self, contextual, valid, *, return_aux=False):
        """Shared node-to-hyperedge weighted mean; return nodes, P and E.

        P[b,i,m,k] = softmax_k((W_m Z[b,i,m]) dot e[k] / sqrt(d))
        With confidence off, A=P. Otherwise A[b,i,m,k]=P[b,i,m,k]*c[b,i,m],
        where c=max_class softmax(head_m(Z)) is detached from main CE.
        E[b,i,k] = sum_m A[b,i,m,k] Z[b,i,m] / (eps + sum_m A[b,i,m,k])

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
            info = {}
            write_incidence = incidence
            if self.confidence_heads is not None:
                logits = torch.stack([self.confidence_heads[m](nodes[:, :, k].float())
                                      for k, m in enumerate(MODALITIES)], dim=2)
                logits = logits.masked_fill(~valid[:, :, None, None], 0)
                confidence = logits.detach().softmax(-1).amax(-1)
                confidence = confidence.masked_fill(~valid.unsqueeze(-1), 0)
                write_incidence = incidence * confidence.unsqueeze(-1)
                info = {'hypergraph_confidence_logits': logits,
                        'hypergraph_confidence': confidence,
                        'hypergraph_write_incidence': write_incidence}
            edge_sum = torch.einsum('bnmk,bnmd->bnkd', write_incidence, nodes.float())
            edge_degree = write_incidence.sum(dim=2)
            hyperedges = edge_sum / (edge_degree.unsqueeze(-1) + self.eps)
            if self.confidence_heads is not None:
                info['hypergraph_write_weights'] = write_incidence / (edge_degree.unsqueeze(2) + self.eps)
        if return_aux:
            return nodes, incidence, hyperedges, info
        return nodes, incidence, hyperedges

    def forward(self, contextual, valid):
        """Direction 1: G[b,i,m] = sum_k P[b,i,m,k] E[b,i,k]."""
        nodes, incidence, hyperedges, info = self.encode_hyperedges(contextual, valid, return_aux=True)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            raw = torch.einsum('bnmk,bnkd->bnmd', incidence, hyperedges)
            raw = raw.masked_fill(~valid[:, :, None, None], 0)
        result = {m: raw[:, :, k].to(nodes.dtype) for k, m in enumerate(MODALITIES)}
        return result, {'hypergraph_membership': incidence,
                        'hypergraph_content': hyperedges,
                        'hypergraph_raw': raw, **info}


class UtteranceHypergraphFusion(UtteranceHypergraph):
    """Direction 2: pool mean hyperedges into one d-wide utterance feature.

    b[i,k] = w_f dot tanh(W_f E[i,k] + b_f)
    beta[i,:] = softmax_k(b[i,:] / temperature)
    fused[i] = sum_k beta[i,k] E[i,k]

    Edge attention is a learned aggregation weight, not emotion confidence.
    No edge-to-node readback or direct contextual Z skip is computed here.
    """
    def __init__(self, hidden, num_hyperedges=4, temperature=1.0, *, confidence=False, num_classes=None):
        super().__init__(hidden, num_hyperedges, confidence=confidence, num_classes=num_classes)
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)
        self.edge_projection = nn.Linear(hidden, hidden)
        self.edge_score = nn.Linear(hidden, 1, bias=False)

    def forward(self, contextual, valid):
        nodes, incidence, hyperedges, info = self.encode_hyperedges(contextual, valid, return_aux=True)
        with torch.autocast(device_type=nodes.device.type, enabled=False):
            scores = self.edge_score(torch.tanh(self.edge_projection(hyperedges))).squeeze(-1)
            beta = (scores / self.temperature).softmax(dim=-1).masked_fill(~valid.unsqueeze(-1), 0)
            fused = (beta.unsqueeze(-1) * hyperedges).sum(dim=2)
            fused = fused.masked_fill(~valid.unsqueeze(-1), 0)
        return fused.to(nodes.dtype), {'hypergraph_membership': incidence,
                                       'hypergraph_content': hyperedges,
                                       'hypergraph_attention': beta,
                                       'hypergraph_fused': fused, **info}
