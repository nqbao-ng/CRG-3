"""Kernel-free sigmoid-gated tanh mixing followed by a width-preserving SSM."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import SelectiveStateSpace, RMSNorm


class GatedLocalMixer(nn.Module):
    """Mix causal role tokens with one input-dependent scalar gate per lag."""
    def __init__(self, channels, kernel_size, bias):
        super().__init__()
        self.channels, self.kernel_size = channels, kernel_size
        self.bias = nn.Parameter(bias.detach().clone())
        # Isolate gate initialization so all common weights match old checkpoints.
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Linear(2 * channels, kernel_size)
            nn.init.zeros_(self.gate.bias)

    def windows(self, u):
        # [B,L,C] -> [B,L,K,C], lag0 current. All positions vectorized.
        return F.pad(u.transpose(1, 2), (self.kernel_size - 1, 0)).unfold(
            2, self.kernel_size, 1).permute(0, 2, 3, 1).flip(2)

    def source_gates(self, u, sources):
        # Same calculation as dot(w_l, concat(u_p, u_(p-l))) + b_l,
        # without materializing a separate concat for every lag.
        current = F.linear(u, self.gate.weight[:, :self.channels], self.gate.bias)
        source = torch.einsum("blkc,kc->blk", sources, self.gate.weight[:, self.channels:])
        logits = (current + source).float()
        present = (torch.arange(u.shape[1], device=u.device)[:, None]
                   >= torch.arange(self.kernel_size, device=u.device)[None, :])
        gates = logits.sigmoid().masked_fill(~present, 0)
        return gates.to(u.dtype)

    def forward(self, u):
        sources = self.windows(u)
        gates = self.source_gates(u, sources)
        features = self.windows(u.tanh())
        return (gates.unsqueeze(-1) * features).sum(2) + self.bias.to(u.dtype)


class DirectGatedRoleSSMMixer(SelectiveStateSpace):
    """Normalized roles -> gated tanh mixing -> selective scan -> output projection.

    Only the target is projected when read_position is supplied. There is no
    split input projection, output z gate, learned local kernel or expansion.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, *, output_projection=True):
        nn.Module.__init__(self)
        self.d_inner, self.d_state = d_model, d_state
        self.dt_rank = math.ceil(d_model / 16)
        # Preserve historical seeded initialization. This temporary tensor only
        # advances RNG; it is never a parameter or part of the forward graph.
        nn.init.kaiming_uniform_(torch.empty(2 * d_model, d_model), a=math.sqrt(5))
        self.out_proj = (nn.Linear(d_model, d_model, bias=False)
                         if output_projection else nn.Identity())
        # Preserve the original bias draw and RNG order without registering a
        # convolution. Its weights are discarded; no convolution runs at runtime.
        initial_local = nn.Conv1d(d_model, d_model, d_conv, groups=d_model, padding=d_conv - 1)
        self._init_state_parameters()
        self.local = GatedLocalMixer(d_model, d_conv, initial_local.bias)

    def forward(self, x, read_position=None, *, project=True):
        y = self.selective_scan(self.local(x))
        if read_position is not None:
            y = y[torch.arange(y.shape[0], device=y.device), read_position]
        return self.out_proj(y) if project else y


class DirectGatedRoleSSMBlock(nn.Module):
    """Keep pre-norm and residual; optionally project only the target output."""
    def __init__(self, d_model, d_state=16, d_conv=4):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = DirectGatedRoleSSMMixer(d_model, d_state, d_conv)

    def forward(self, x, read_position=None):
        residual = x
        if read_position is not None:
            residual = x[torch.arange(x.shape[0], device=x.device), read_position]
        return residual + self.mixer(self.norm(x), read_position)
