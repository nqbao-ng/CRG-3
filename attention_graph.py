"""Window-gathered Diff/Same pooling -> [Diff, Same, Adjacent, Target] scan.

Each nonempty relation applies softmax attention, then takes the arithmetic mean
of weighted historical T/A/V sources (sum is retained for old checkpoints).
Both groups read the same layer-input snapshot. One Mamba (original causal
Conv1d or a scalar-gated local mixer) and one GateUpdate per receiving modality.
Missing roles are compacted out; scan state resets per target. No future input.
Adjacent can use the input snapshot (A), or the previous updated node in this
same layer (B). B preserves gradient through the node chain; groups still use H.
Same/Diff connectivity remains dense [B,N,N], row=target and column=source.
Gather at most W historical K/V positions before attention; slice connectivity
to the same positions. Only allowed edges participate in the joint softmax.
"""
import math

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .graph import MODALITIES, GateUpdate
from .mamba import MambaBlock
from .gated_local import GatedRoleMambaBlock
from .adjacency import (adjacency_from_neighbors, build_speaker_adjacency,
                        mask_dense_adjacency)


def attention_messages(query, keys, values, mask, normalization="joint"):
    """[rows,d], [rows,K,3,d], [rows,K] -> messages [rows,K,d], effective weights.

    Empty rows have exactly zero weights and finite gradients. Effective weights
    sum to one across K*3 (zero for empty rows).
    """
    scores = (query[:, None, None].float() * keys.float()).sum(-1) / math.sqrt(query.shape[-1])
    expanded_mask = mask.unsqueeze(-1).expand_as(scores)
    scores = scores.masked_fill(~expanded_mask, -torch.inf)
    scores = torch.where(mask.any(-1)[:, None, None], scores, torch.zeros_like(scores))
    if normalization == "joint":
        weights = scores.flatten(1).softmax(-1).reshape_as(scores)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    weights = weights.masked_fill(~expanded_mask, 0)
    messages = (weights.to(values.dtype).unsqueeze(-1) * values).sum(dim=2)
    return messages, weights



def attention_pool(query, keys, values, mask, reduction="mean"):
    """Pool alpha*v; mean divides by valid source count, excluding padding.

    Softmax still sums to one before this reduction. Thus mean is the old
    attention sum scaled by 1/(valid utterances * source modalities).
    Empty groups remain zero. sum preserves historical checkpoint behavior.
    """
    messages, weights = attention_messages(query, keys, values, mask)
    pooled = messages.sum(dim=1)
    if reduction == "mean":
        count = (mask.sum(dim=1) * values.shape[2]).clamp_min(1)
        pooled = pooled / count.to(pooled.dtype).unsqueeze(-1)
    elif reduction != "sum":
        raise ValueError(f"Unknown attention pool reduction: {reduction}")
    return pooled, weights


def pack_role_tokens(tokens, present):
    """Compact present roles in their original order; padding stays after the read.

    tokens [rows,4,d], present [rows,4]. The target read is always present for
    scanned rows. Return a max-4-token padded sequence and the read position.
    """
    positions = (present.long().cumsum(1) - 1).clamp_min(0)
    packed = torch.zeros_like(tokens).scatter_add(
        1, positions.unsqueeze(-1).expand_as(tokens), tokens * present.unsqueeze(-1))
    return packed, present.sum(1) - 1


def dense_adjacent_features(adjacency, features):
    """A_adj @ H within one modality; A_adj is constrained to its subdiagonal."""
    # Preserve the source dtype: AMP must not introduce an extra cast before
    # the existing adjacent projection compared with the former shifted H.
    with torch.autocast(device_type=features.device.type, enabled=False):
        return torch.bmm(adjacency.to(features.dtype), features)


class AttentionGraphLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.graph_update_mode != "attention_joint":
            raise ValueError("AttentionGraphLayer requires attention_joint")
        d = cfg.hidden
        self.window = cfg.local_window
        self.chunk_size = cfg.attention_chunk_size
        self.use_checkpoint = cfg.attention_checkpoint
        self.adjacent_source = cfg.attention_adjacent_source
        self.pool_reduction = cfg.attention_pool_reduction
        # Q/K/V projections are shared BETWEEN groups within this layer, while
        # each group has its own softmax denominator and role embedding.
        self.pre_norm = nn.ModuleDict({m: nn.LayerNorm(d) for m in MODALITIES})
        self.query = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        self.key = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        self.value = nn.ModuleDict({m: nn.Linear(d, d, bias=False) for m in MODALITIES})
        self.relation_embedding = nn.ParameterDict({
            r: nn.Parameter(torch.zeros(d)) for r in ("diff", "same")})
        self.adjacent_proj = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})
        self.adjacent_embedding = nn.Parameter(torch.zeros(d))
        self.read_proj = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})
        self.read_embedding = nn.ParameterDict({m: nn.Parameter(torch.zeros(d)) for m in MODALITIES})
        if cfg.joint_local_mixer == "conv":
            self.scan = nn.ModuleDict({m: MambaBlock(d, cfg.d_state, cfg.d_conv,
                                                   use_rmsnorm=True) for m in MODALITIES})
        else:
            self.scan = nn.ModuleDict({m: GatedRoleMambaBlock(
                d, cfg.d_state, cfg.d_conv, mode=cfg.joint_local_mixer,
                gate_normalization=cfg.joint_gate_normalization,
                use_kernel=cfg.joint_local_kernel == "on") for m in MODALITIES})
        self.update = nn.ModuleDict({m: GateUpdate(d, cfg.dropout) for m in MODALITIES})
        if cfg.graph_gate == "off":
            # Construct identically first so removing gates does not shift the
            # random initialization of context projections or subsequent layers.
            # No unused gate parameters remain; W_c/dropout/residual are retained.
            for update in self.update.values():
                update.gate = None

    def _scan_rows(self, query, keys, values, target_rows, diff_mask, same_mask,
                   adjacent, adjacent_mask, read, modality):
        # target_rows are flattened B*N indices. Gather only the chronological
        # interval [i-W, i), inside checkpoint so its K/V activations can be
        # discarded and recomputed. No [rows,N,3,d] source copy is created.
        length = keys.shape[1]
        width = min(self.window, max(0, length - 1))
        dialogue_rows = torch.div(target_rows, length, rounding_mode='floor')
        positions = (target_rows.remainder(length)[:, None] - width
                     + torch.arange(width, device=keys.device)[None])
        present_sources = positions >= 0
        source_indices = positions.clamp_min(0)
        source_keys = keys[dialogue_rows[:, None], source_indices]
        source_values = values[dialogue_rows[:, None], source_indices]
        # Near the start of a dialogue, clamped negative positions are padding,
        # not repeated edges to source0. Preserve any explicitly blocked edges.
        diff_mask = diff_mask.gather(1, source_indices) & present_sources
        same_mask = same_mask.gather(1, source_indices) & present_sources
        diff, _ = attention_pool(query, source_keys, source_values, diff_mask,
                                 self.pool_reduction)
        same, _ = attention_pool(query, source_keys, source_values, same_mask,
                                 self.pool_reduction)
        tokens = torch.stack([
            diff + self.relation_embedding['diff'],
            same + self.relation_embedding['same'],
            adjacent + self.adjacent_embedding, read], dim=1)
        present = torch.stack([diff_mask.any(-1), same_mask.any(-1), adjacent_mask,
                               torch.ones_like(adjacent_mask)], dim=1)
        sequence, read_position = pack_role_tokens(tokens, present)
        output = self.scan[modality](sequence)
        return output[torch.arange(len(output), device=output.device), read_position]

    def _updated_adjacent_nodes(self, H, valid, queries, keys, values, diff_mask,
                                same_mask, adjacent_mask, read, active, modality):
        """B: target i waits for Y_(i-1), while group queries/keys/values stay from H.

        Batch dialogues at the same time step. Never detach Y_(i-1): this is
        feature recurrence across targets, not reuse of the inner SSM state.
        """
        B, N, d = H.shape
        nodes = []
        previous = H.new_zeros(B, d)
        batch_offsets = torch.arange(B, device=H.device) * N
        for time in range(N):
            all_rows = batch_offsets + time
            rows = all_rows[active[all_rows]]
            # Same normalization/projection as A, but applied to updated Y.
            adjacent = self.adjacent_proj[modality](self.pre_norm[modality](previous))
            chunks = []
            for chunk in rows.split(self.chunk_size):
                # split() on an empty tensor returns one empty chunk. CUDA scan
                # cannot consume that batch (notably at time0 and window0).
                if chunk.numel() == 0:
                    continue
                dialogue_rows = torch.div(chunk, N, rounding_mode='floor')
                inputs = (queries[chunk], keys, values, chunk, diff_mask[chunk],
                          same_mask[chunk], adjacent[dialogue_rows],
                          adjacent_mask[chunk], read[chunk], modality)
                if self.use_checkpoint and self.training and torch.is_grad_enabled():
                    context = checkpoint(self._scan_rows, *inputs, use_reentrant=False)
                else:
                    context = self._scan_rows(*inputs)
                chunks.append(context)
            context = H.new_zeros(B, d)
            if chunks:
                dialogue_rows = torch.div(rows, N, rounding_mode='floor')
                context = context.index_copy(0, dialogue_rows, torch.cat(chunks).to(context.dtype))
            current = self.update[modality](H[:, time], context, valid[:, time])
            nodes.append(current)
            previous = current
        return torch.stack(nodes, dim=1)

    def forward(self, H, diff, same, valid, *, adjacency=None):
        B, N, d = H['t'].shape
        if adjacency is None:
            adjacency = adjacency_from_neighbors(diff, same, valid, self.window)
        else:
            adjacency = mask_dense_adjacency(adjacency, valid, self.window)
        normed = {m: self.pre_norm[m](H[m]) for m in MODALITIES}
        keys = torch.stack([self.key[m](normed[m]) for m in MODALITIES], 2)
        values = torch.stack([self.value[m](normed[m]) for m in MODALITIES], 2)
        diff_mask = adjacency['diff'].reshape(B * N, N)
        same_mask = adjacency['same'].reshape(B * N, N)
        adjacent_mask = adjacency['adjacent'].any(-1).reshape(-1)
        active = (diff_mask.any(-1) | same_mask.any(-1) | adjacent_mask) & valid.reshape(-1)
        rows = active.nonzero(as_tuple=True)[0]
        result = {}
        for m in MODALITIES:
            queries = self.query[m](normed[m]).reshape(B * N, d)
            read = (self.read_proj[m](normed[m]) + self.read_embedding[m]).reshape(B * N, d)
            if self.adjacent_source == "updated":
                result[m] = self._updated_adjacent_nodes(
                    H[m], valid, queries, keys, values, diff_mask,
                    same_mask, adjacent_mask, read, active, m)
                continue
            # Adjacent adjacency only allows source i-1 in the same modality.
            prev_features = dense_adjacent_features(adjacency['adjacent'], normed[m])
            adjacent = self.adjacent_proj[m](prev_features).reshape(B * N, d)
            chunks = []
            for chunk in rows.split(self.chunk_size):
                if chunk.numel() == 0:
                    continue
                inputs = (queries[chunk], keys, values, chunk, diff_mask[chunk],
                          same_mask[chunk], adjacent[chunk],
                          adjacent_mask[chunk], read[chunk], m)
                if self.use_checkpoint and self.training and torch.is_grad_enabled():
                    context = checkpoint(self._scan_rows, *inputs, use_reentrant=False)
                else:
                    context = self._scan_rows(*inputs)
                chunks.append(context)
            context = H[m].new_zeros(B * N, d)
            if chunks:
                context = context.index_copy(0, rows, torch.cat(chunks).to(context.dtype))
            result[m] = self.update[m](H[m], context.reshape(B, N, d), valid)
        counts_d = diff_mask.sum(-1).reshape(B, N)
        counts_s = same_mask.sum(-1).reshape(B, N)
        token_count = (diff_mask.any(-1).long() + same_mask.any(-1).long()
                       + adjacent_mask.long() + 1) * active.long()
        return result, {'diff': {'neighbor_count': counts_d.detach()},
                        'same': {'neighbor_count': counts_s.detach()},
                        'adjacent_present': adjacent_mask.reshape(B, N).detach(),
                        'scan_token_count': token_count.reshape(B, N).detach()}


class AttentionGraphEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.window = cfg.local_window
        self.layers = nn.ModuleList([AttentionGraphLayer(cfg) for _ in range(cfg.graph_layers)])
        self.final_norm = nn.ModuleDict({m: nn.LayerNorm(cfg.hidden) for m in MODALITIES})

    def forward(self, H, diff, same, valid, *, speakers=None, adjacency=None):
        # Build structural matrices once per encoder call, reuse across layers.
        # Attention coefficients still depend on each layer's current features.
        if adjacency is None:
            adjacency = (build_speaker_adjacency(speakers, valid, self.window)
                         if speakers is not None else
                         adjacency_from_neighbors(diff, same, valid, self.window))
        diagnostics = {}
        for index, layer in enumerate(self.layers):
            H, diagnostics[index] = layer(H, diff, same, valid, adjacency=adjacency)
        mask = valid.unsqueeze(-1)
        return {m: self.final_norm[m](H[m]) * mask.to(H[m].dtype) for m in MODALITIES}, diagnostics
