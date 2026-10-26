"""Shape-based forward arithmetic accounting, including opaque CUDA selective scan.

Convention: one multiply-accumulate = two FLOPs. Count Linear/Conv1d (including
bias), dense adjacency aggregation, attention arithmetic, and the sequential-equivalent
SSM recurrence. Exclude normalization, activations/softmax/transcendentals,
masking, data movement and miscellaneous vector additions. This is an explicit
major-arithmetic estimate, not a hardware instruction count or training FLOPs.
"""
from collections import Counter

import torch.nn as nn

from . import attention_graph
from .mamba import MambaMixer, SelectiveStateSpace
from .gated_local import GatedLocalConv
from .hypergraph import UtteranceHypergraph, UtteranceHypergraphFusion


class ForwardCost:
    def __init__(self, model):
        self.model = model
        self.counts = Counter()
        self.handles = []

    def linear(self, module, inputs, output):
        macs = output.numel() * module.in_features
        self.counts['linear_macs'] += macs
        self.counts['linear_flops'] += 2 * macs + (output.numel() if module.bias is not None else 0)

    def conv(self, module, inputs, output):
        # Count the complete convolution output, including the causal tail that
        # Mamba subsequently crops. Those output elements are actually computed.
        macs = output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        self.counts['conv1d_macs'] += macs
        self.counts['conv1d_flops'] += 2 * macs + (output.numel() if module.bias is not None else 0)

    def scan(self, module, inputs, output):
        rows, length = inputs[0].shape[:2]
        tokens = rows * length * module.d_inner
        state = module.d_state
        # Per channel/token: delta*A (S); delta*B*u (2S); state update
        # (2S); state*C reduction (2S-1); D*u+readout (2): 7S+1.
        self.counts['scan_arithmetic_flops'] += tokens * (7 * state + 1)
        self.counts['scan_exp_evaluations_excluded'] += tokens * state + module.d_inner * state
        self.counts['scan_channel_tokens'] += tokens
        self.counts['scan_calls'] += 1

    def gated_local(self, module, inputs, output):
        rows, length, channels = inputs[0].shape
        width = module.kernel_size
        # Actual vectorized arithmetic includes zero-padded source slots.
        # Gates: two C-length dot products + bias + combine = 4C FLOPs/slot.
        # Candidate multiply, gate multiply, reduce K terms, output bias = 3KC.
        # sigmoid/softmax/tanh/SiLU and masks stay excluded by this convention.
        slots = rows * length * width * channels
        # Removing k_l also removes one candidate multiplication per slot.
        self.counts['gated_local_flops'] += (6 if module.kernel is None else 7) * slots
        self.counts['gated_local_calls'] += 1

    def hypergraph(self, module, inputs, output):
        contextual, valid = inputs
        batch, length = valid.shape
        modalities, hidden, edges = len(contextual), module.hidden, module.num_hyperedges
        # Routing Linear layers are counted by their own hooks. Count anchor
        # scores (+scale), writes, degree mean and reads here, including
        # padded slots that vectorized kernels really compute. Norm/softmax excluded.
        scores = batch * length * modalities * edges * (2 * hidden)
        writes = batch * length * edges * hidden * (2 * modalities - 1)
        # Sum M incidence weights, add epsilon, then divide each of H channels.
        mean = batch * length * edges * (modalities + hidden)
        # Fusion's score Linear layers are counted by their own hooks. It
        # pools K edges once, rather than reading them back to M nodes.
        receivers = 1 if isinstance(module, UtteranceHypergraphFusion) else modalities
        reads = batch * length * receivers * hidden * (2 * edges - 1)
        self.counts['hypergraph_flops'] += scores + writes + mean + reads
        self.counts['hypergraph_calls'] += 1

    def __enter__(self):
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                self.handles.append(module.register_forward_hook(self.linear))
            elif isinstance(module, nn.Conv1d):
                self.handles.append(module.register_forward_hook(self.conv))
            elif isinstance(module, (MambaMixer, SelectiveStateSpace)):
                self.handles.append(module.register_forward_hook(self.scan))
            elif isinstance(module, GatedLocalConv):
                self.handles.append(module.register_forward_hook(self.gated_local))
            elif isinstance(module, UtteranceHypergraph):
                self.handles.append(module.register_forward_hook(self.hypergraph))
        self.original_attention = attention_graph.attention_messages

        def counted_attention(query, keys, values, mask, normalization='joint'):
            rows, width, modalities, hidden = keys.shape
            # Dot product (2H-1) plus scale (1); weighted values and their sum.
            self.counts['attention_flops'] += (
                rows * width * modalities * 2 * hidden
                + rows * width * hidden * (2 * modalities - 1))
            self.counts['attention_calls'] += 1
            return self.original_attention(query, keys, values, mask, normalization)

        attention_graph.attention_messages = counted_attention
        self.original_pool = attention_graph.attention_pool

        def counted_pool(query, keys, values, mask, reduction='mean'):
            # attention_messages counts modality reduction; pooling also reduces time.
            rows, width, _, hidden = keys.shape
            self.counts['attention_flops'] += rows * hidden * max(0, width - 1)
            if reduction == 'mean':
                self.counts['attention_flops'] += rows * hidden  # divide pooled vector by valid count
            self.counts['attention_pool_calls'] += 1
            return self.original_pool(query, keys, values, mask, reduction)

        attention_graph.attention_pool = counted_pool
        self.original_adjacent = attention_graph.dense_adjacent_features

        def counted_adjacent(adjacency, features):
            batch, targets, sources = adjacency.shape
            self.counts['adjacency_flops'] += 2 * batch * targets * sources * features.shape[-1]
            return self.original_adjacent(adjacency, features)

        attention_graph.dense_adjacent_features = counted_adjacent
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        attention_graph.attention_messages = self.original_attention
        attention_graph.attention_pool = self.original_pool
        attention_graph.dense_adjacent_features = self.original_adjacent

    def result(self):
        result = dict(self.counts)
        result['total_major_arithmetic_flops'] = sum(self.counts[k] for k in (
            'linear_flops', 'conv1d_flops', 'attention_flops', 'scan_arithmetic_flops',
            'gated_local_flops', 'adjacency_flops', 'hypergraph_flops'))
        result['gflops'] = result['total_major_arithmetic_flops'] / 1e9
        return result
