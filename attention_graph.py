"""Window-gathered Diff/Same pooling -> [Diff, Same, Adjacent, Target] scan.

Each nonempty relation uses joint softmax + sum, separate softmax per modality
+ mean of modality contexts, or legacy joint softmax + source-count mean.
Both groups read the same layer-input snapshot. Each modality uses a kernel-free
gated tanh SSM, with optional sequential or aligned bidirectional readout.
Missing roles are compacted out; scan state resets per target. No future input.
Adjacent can use the input snapshot (A), or the previous updated node in this
same layer (B). B preserves gradient through the node chain; groups still use H.
Same/Diff connectivity remains dense [B,N,N], row=target and column=source.
Gather at most W historical K/V positions before attention. The history scope
selects either all source modalities or only the receiving modality.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import MODALITIES, GateUpdate
from .gated_local import DirectGatedRoleSSMBlock, DirectGatedRoleSSMMixer
from .adjacency import (adjacency_from_neighbors, build_speaker_adjacency,
                        mask_dense_adjacency)


def attention_messages(query, keys, values, mask, normalization="joint", *, scores=None):
    """[rows,d], [rows,K,M,d], [rows,K] -> messages [rows,K,d], effective weights.

    Empty rows have exactly zero weights and finite gradients. Joint weights sum
    to one across K*M; per_modality weights sum to one over K in each modality.
    """
    if scores is None:
        scores = (query[:, None, None].float() * keys.float()).sum(-1) / math.sqrt(query.shape[-1])
    elif scores.shape != keys.shape[:-1]:
        raise ValueError('Precomputed attention scores must match [rows, sources, modalities]')
    else:
        scores = scores.float()
    expanded_mask = mask.unsqueeze(-1).expand_as(scores)
    scores = scores.masked_fill(~expanded_mask, -torch.inf)
    scores = torch.where(mask.any(-1)[:, None, None], scores, torch.zeros_like(scores))
    if normalization == "joint":
        weights = scores.flatten(1).softmax(-1).reshape_as(scores)
    elif normalization == 'per_modality':
        weights = scores.softmax(dim=1)
    else:
        raise ValueError(f"Unknown normalization: {normalization}")
    weights = weights.masked_fill(~expanded_mask, 0)
    messages = (weights.to(values.dtype).unsqueeze(-1) * values).sum(dim=2)
    return messages, weights



def attention_pool(query, keys, values, mask, reduction="mean", *, scores=None):
    """Sum, legacy source-count mean, or mean of separate modality contexts.

    sum: joint softmax over source time x modality, no additional division.
    mean: same joint softmax, divide by valid utterances * source modalities.
    modality_mean: softmax over time separately for each source modality, sum
    over time, then average modality contexts. No utterance-count division.
    Empty groups remain zero. Preserve legacy operations exactly for old modes.
    """
    if reduction == 'modality_mean':
        messages, weights = attention_messages(query, keys, values, mask,
                                               'per_modality', scores=scores)
    elif scores is None:
        messages, weights = attention_messages(query, keys, values, mask)
    else:
        messages, weights = attention_messages(query, keys, values, mask, scores=scores)
    pooled = messages.sum(dim=1)
    if reduction == "mean":
        count = (mask.sum(dim=1) * values.shape[2]).clamp_min(1)
        pooled = pooled / count.to(pooled.dtype).unsqueeze(-1)
    elif reduction == 'modality_mean':
        pooled = pooled / values.shape[2]
    elif reduction != "sum":
        raise ValueError(f"Unknown attention pool reduction: {reduction}")
    return pooled, weights


class AdditiveRelationScorer(nn.Module):
    """GAT-style scoring on retained Q/K, without per-edge concatenation.

    e = LeakyReLU(a_target^T Q_i + a_source^T K_j + a_relation^T e_R).
    Value projection, graph structure, joint softmax and role pooling stay separate.
    """
    negative_slope = 0.2

    def __init__(self, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(3, hidden))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, query, keys, relation_embedding):
        weight = self.weight.float()
        target = (query.float() * weight[0]).sum(-1)[:, None, None]
        source = (keys.float() * weight[1]).sum(-1)
        relation = (relation_embedding.float() * weight[2]).sum()
        return F.leaky_relu(target + source + relation, self.negative_slope)


def pack_role_tokens(tokens, present):
    """Compact present roles in their original order; padding stays after the read.

    tokens [rows,4,d], present [rows,4]. The target read is always present for
    scanned rows. Return a max-4-token padded sequence and the read position.
    """
    positions = (present.long().cumsum(1) - 1).clamp_min(0)
    packed = torch.zeros_like(tokens).scatter_add(
        1, positions.unsqueeze(-1).expand_as(tokens), tokens * present.unsqueeze(-1))
    return packed, present.sum(1) - 1


def reverse_packed_roles(sequence, read_position):
    """Reverse only the valid prefix, keeping Target first and padding last."""
    positions = read_position[:, None] - torch.arange(sequence.shape[1], device=sequence.device)
    reversed_roles = sequence.gather(1, positions.clamp_min(0).unsqueeze(-1).expand_as(sequence))
    return reversed_roles.masked_fill(positions.unsqueeze(-1) < 0, 0)


def dense_adjacent_features(adjacency, features):
    """A_adj @ H within one modality; A_adj is constrained to its subdiagonal."""
    # Preserve the source dtype: AMP must not introduce an extra cast before
    # the existing adjacent projection compared with the former shifted H.
    with torch.autocast(device_type=features.device.type, enabled=False):
        return torch.bmm(adjacency.to(features.dtype), features)


class GatedSharedRoleSSM(nn.Module):
    """Aligned T/A/V roles -> one shared SSM -> gated residual per modality.

    X_p concatenates the three modalities at role p and projects to hidden.
    Historical full-width checkpoints omit that projection. J is the shared
    SSM's Target output.
    C_m = P_m J, F_m = U_m + sigmoid(W_gm [U_m; C_m] + b_gm) * C_m.
    This is a CRG adaptation inspired by GCMN, not its bidirectional BSSM.
    """
    def __init__(self, hidden, d_state, d_conv, *, project_input=True):
        super().__init__()
        concat_width = len(MODALITIES) * hidden
        scan_width = hidden if project_input else concat_width
        self.joint_proj = nn.Linear(concat_width, hidden, bias=False) if project_input else nn.Identity()
        self.shared_scan = DirectGatedRoleSSMBlock(scan_width, d_state, d_conv)
        self.cross_proj = nn.ModuleDict({m: nn.Linear(scan_width, hidden, bias=False)
                                        for m in MODALITIES})
        self.fusion_gate = nn.ModuleDict({m: nn.Linear(2 * hidden, hidden)
                                         for m in MODALITIES})

    def forward(self, private, sequences, read_position):
        combined = self.joint_proj(torch.cat([sequences[m] for m in MODALITIES], dim=-1))
        joint = self.shared_scan(combined, read_position)
        result = {}
        for m in MODALITIES:
            cross = self.cross_proj[m](joint)
            gate = torch.sigmoid(self.fusion_gate[m](torch.cat([private[m], cross], dim=-1)))
            result[m] = private[m] + gate * cross
        return result


class AttentionGraphLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.graph_update_mode != "attention_joint":
            raise ValueError("AttentionGraphLayer requires attention_joint")
        d = cfg.hidden
        self.window = cfg.local_window
        self.chunk_size = cfg.attention_chunk_size
        self.adjacent_source = cfg.attention_adjacent_source
        self.pool_reduction = cfg.attention_pool_reduction
        self.attention_score = cfg.attention_score
        self.history_scope = cfg.attention_history_scope
        self.role_scan_mode = cfg.role_scan_mode
        self.joint_ssm_fusion = cfg.joint_ssm_fusion
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
        self.scan = nn.ModuleDict({m: DirectGatedRoleSSMBlock(
            d, cfg.d_state, cfg.d_conv) for m in MODALITIES})
        self.update = nn.ModuleDict({m: GateUpdate(d, cfg.dropout) for m in MODALITIES})
        if cfg.graph_gate == "off":
            # Construct identically first so removing gates does not shift the
            # random initialization of context projections or subsequent layers.
            # No unused gate parameters remain; W_c/dropout/residual are retained.
            for update in self.update.values():
                update.gate = None
        if self.attention_score == 'gat':
            # Isolate new initialization so subsequent layers/classifier and
            # common weights match a same-seed dot-product control.
            with torch.random.fork_rng(devices=[]):
                self.gat_scorer = nn.ModuleDict({m: AdditiveRelationScorer(d) for m in MODALITIES})
            # Separate from the retained post-pooling relation embeddings.
            self.score_relation_embedding = nn.ParameterDict({
                r: nn.Parameter(torch.zeros(d)) for r in ('diff', 'same')})
        if self.joint_ssm_fusion != 'off':
            # Preserve common weights and subsequent-layer RNG for matched controls.
            with torch.random.fork_rng(devices=[]):
                self.shared_fusion = GatedSharedRoleSSM(
                    d, cfg.d_state, cfg.d_conv,
                    project_input=self.joint_ssm_fusion == 'gated',
                )
        if self.role_scan_mode != 'forward':
            # Keep the forward baseline's initialization under the same seed.
            with torch.random.fork_rng(devices=[]):
                if self.role_scan_mode == 'sequential':
                    self.reverse_scan = nn.ModuleDict({m: DirectGatedRoleSSMBlock(
                        d, cfg.d_state, cfg.d_conv) for m in MODALITIES})
                    self.reverse_update = nn.ModuleDict({m: GateUpdate(
                        d, cfg.dropout, use_gate=cfg.graph_gate == 'on') for m in MODALITIES})
                else:
                    self.reverse_scan = nn.ModuleDict({m: DirectGatedRoleSSMMixer(
                        d, cfg.d_state, cfg.d_conv, output_projection=False) for m in MODALITIES})
                    self.bssm_gate = nn.ModuleDict({m: nn.Linear(d, d) for m in MODALITIES})

    def _role_rows(self, query, keys, values, target_rows, diff_mask, same_mask,
                   adjacent, adjacent_mask, read, modality):
        # target_rows are flattened B*N indices. Gather only the chronological
        # interval [i-W, i). Restrict modalities before gathering the window.
        if self.history_scope == 'intra':
            source_modality = MODALITIES.index(modality)
            keys = keys[:, :, source_modality:source_modality + 1]
            values = values[:, :, source_modality:source_modality + 1]
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
        if self.attention_score == 'gat':
            diff_scores = self.gat_scorer[modality](
                query, source_keys, self.score_relation_embedding['diff'])
            same_scores = self.gat_scorer[modality](
                query, source_keys, self.score_relation_embedding['same'])
            diff_h, _ = attention_pool(query, source_keys, source_values, diff_mask,
                                      self.pool_reduction, scores=diff_scores)
            same_h, _ = attention_pool(query, source_keys, source_values, same_mask,
                                      self.pool_reduction, scores=same_scores)
        else:
            diff_h, _ = attention_pool(query, source_keys, source_values, diff_mask,
                                      self.pool_reduction)
            same_h, _ = attention_pool(query, source_keys, source_values, same_mask,
                                      self.pool_reduction)
        tokens = torch.stack([
            diff_h + self.relation_embedding['diff'],
            same_h + self.relation_embedding['same'],
            adjacent + self.adjacent_embedding, read], dim=1)
        present = torch.stack([diff_mask.any(-1), same_mask.any(-1), adjacent_mask,
                               torch.ones_like(adjacent_mask)], dim=1)
        return pack_role_tokens(tokens, present)

    def _scan_rows(self, *args):
        sequence, read_position = self._role_rows(*args)
        modality = args[-1]
        forward = self.scan[modality]
        if self.role_scan_mode == 'cagmamba':
            normalized = forward.norm(sequence)
            target = normalized[torch.arange(normalized.shape[0], device=normalized.device),
                                read_position]
            forward_target = forward.mixer(normalized, read_position, project=False)
            # Target aligns with step zero of the reversed valid prefix.
            backward_roles = reverse_packed_roles(normalized, read_position)
            backward_target = self.reverse_scan[modality](backward_roles)[:, 0]
            gate = F.silu(self.bssm_gate[modality](target))
            return forward.mixer.out_proj(gate * (forward_target * backward_target))
        return forward(sequence, read_position)

    def _scan_chunk(self, H, inputs):
        if self.role_scan_mode != 'sequential':
            return self._scan_rows(*inputs)
        rows, modality = inputs[3], inputs[-1]
        node = H.reshape(-1, H.shape[-1])[rows]
        sequence, read_position = self._role_rows(*inputs)
        forward_context = self.scan[modality](sequence, read_position)
        valid = torch.ones_like(read_position, dtype=torch.bool)
        updated = self.update[modality](node, forward_context, valid)
        backward_roles = reverse_packed_roles(sequence, read_position)
        backward_roles = torch.cat([updated.unsqueeze(1), backward_roles[:, 1:]], dim=1)
        reverse = self.reverse_scan[modality]
        backward_context = reverse.mixer(reverse.norm(backward_roles), read_position)
        return self.reverse_update[modality](updated, backward_context, valid)

    def _finish_scan_rows(self, node, valid, rows, chunks, modality):
        updated_in_chunk = self.role_scan_mode == 'sequential'
        flat_node = node.reshape(-1, node.shape[-1])
        output = flat_node if updated_in_chunk else torch.zeros_like(flat_node)
        if chunks:
            output = output.index_copy(0, rows, torch.cat(chunks).to(output.dtype))
        output = output.reshape_as(node)
        if updated_in_chunk:
            return output * valid.unsqueeze(-1).to(output.dtype)
        return self.update[modality](node, output, valid)

    def _fused_context_rows(self, rows, queries, keys, values, diff_mask,
                            same_mask, adjacent, adjacent_mask, read):
        sequences, private = {}, {}
        read_position = None
        for m in MODALITIES:
            sequence, read_position = self._role_rows(
                queries[m][rows], keys, values, rows, diff_mask[rows],
                same_mask[rows], adjacent[m], adjacent_mask[rows], read[m][rows], m)
            sequences[m] = sequence
            # Old private streams bypass targets with no historical roles.
            # Keep that behavior; the new shared stream can still fuse T/A/V there.
            selected = (read_position > 0).nonzero(as_tuple=True)[0]
            context = sequence.new_zeros(len(rows), sequence.shape[-1])
            if selected.numel():
                context = context.index_copy(0, selected,
                    self.scan[m](sequence[selected], read_position[selected]))
            private[m] = context
        # Structural masks are common across modalities, so packing aligns roles.
        return self.shared_fusion(private, sequences, read_position)

    def _forward_fused(self, H, valid, normed, queries, keys, values, diff_mask,
                       same_mask, adjacent_mask, read, adjacency):
        """Four streams; updated Adjacent advances all modalities in lockstep."""
        B, N, d = H['t'].shape
        if self.adjacent_source == 'input':
            adjacent = {m: self.adjacent_proj[m](dense_adjacent_features(
                adjacency['adjacent'], normed[m])).reshape(B * N, d) for m in MODALITIES}
            rows = valid.reshape(-1).nonzero(as_tuple=True)[0]
            pieces = {m: [] for m in MODALITIES}
            for chunk in rows.split(self.chunk_size):
                if not chunk.numel():
                    continue
                contexts = self._fused_context_rows(chunk, queries, keys, values,
                    diff_mask, same_mask, {m: adjacent[m][chunk] for m in MODALITIES},
                    adjacent_mask, read)
                for m in MODALITIES:
                    pieces[m].append(contexts[m])
            result = {}
            for m in MODALITIES:
                context = H[m].new_zeros(B * N, d)
                if pieces[m]:
                    context = context.index_copy(0, rows, torch.cat(pieces[m]).to(context.dtype))
                result[m] = self.update[m](H[m], context.reshape(B, N, d), valid)
            return result

        previous = {m: H[m].new_zeros(B, d) for m in MODALITIES}
        nodes = {m: [] for m in MODALITIES}
        offsets = torch.arange(B, device=valid.device) * N
        for time in range(N):
            all_rows = offsets + time
            rows = all_rows[valid[:, time]]
            adjacent = {m: self.adjacent_proj[m](self.pre_norm[m](previous[m]))
                        for m in MODALITIES}
            pieces = {m: [] for m in MODALITIES}
            for chunk in rows.split(self.chunk_size):
                if not chunk.numel():
                    continue
                b = torch.div(chunk, N, rounding_mode='floor')
                contexts = self._fused_context_rows(chunk, queries, keys, values,
                    diff_mask, same_mask, {m: adjacent[m][b] for m in MODALITIES},
                    adjacent_mask, read)
                for m in MODALITIES:
                    pieces[m].append(contexts[m])
            current = {}
            b = torch.div(rows, N, rounding_mode='floor')
            for m in MODALITIES:
                context = H[m].new_zeros(B, d)
                if pieces[m]:
                    context = context.index_copy(0, b, torch.cat(pieces[m]).to(context.dtype))
                current[m] = self.update[m](H[m][:, time], context, valid[:, time])
                nodes[m].append(current[m])
            previous = current  # Keep gradients; all m advance after this target.
        return {m: torch.stack(nodes[m], dim=1) for m in MODALITIES}

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
                chunks.append(self._scan_chunk(H, inputs))
            dialogue_rows = torch.div(rows, N, rounding_mode='floor')
            current = self._finish_scan_rows(H[:, time], valid[:, time], dialogue_rows,
                                             chunks, modality)
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
        if self.joint_ssm_fusion != 'off':
            queries = {m: self.query[m](normed[m]).reshape(B * N, d) for m in MODALITIES}
            read = {m: (self.read_proj[m](normed[m]) + self.read_embedding[m]).reshape(B * N, d)
                    for m in MODALITIES}
            result = self._forward_fused(H, valid, normed, queries, keys, values,
                                        diff_mask, same_mask, adjacent_mask, read, adjacency)
            return result, {
                'diff': {'neighbor_count': diff_mask.sum(-1).reshape(B, N).detach()},
                'same': {'neighbor_count': same_mask.sum(-1).reshape(B, N).detach()},
                'adjacent_present': adjacent_mask.reshape(B, N).detach(),
                'scan_token_count': ((diff_mask.any(-1).long() + same_mask.any(-1).long()
                    + adjacent_mask.long() + 1) * active.long()).reshape(B, N).detach(),
                'shared_scan_token_count': ((diff_mask.any(-1).long() + same_mask.any(-1).long()
                    + adjacent_mask.long() + 1) * valid.reshape(-1).long()).reshape(B, N).detach()}
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
                chunks.append(self._scan_chunk(H[m], inputs))
            result[m] = self._finish_scan_rows(H[m], valid, rows, chunks, m)
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
