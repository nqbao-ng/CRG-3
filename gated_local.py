"""Causal scalar-gated local mixers for the pooled-role attention graph.

gated_silu: SiLU(sum_l g[p,l] * k[l] * u[p-l] + bias).
Without k_l: SiLU(sum_l g[p,l] * u[p-l] + bias); all gates/norms/SSM stay unchanged.
gmu_tanh:   sum_l g[p,l] * tanh(k[l] * u[p-l]) + bias.
Without k_l: sum_l g[p,l] * tanh(u[p-l]) + bias; no post-sum SiLU.

One scalar gate per lag, conditioned on the current and source vectors. Sigmoid
is the matched default; softmax optionally makes valid sources compete. These
are GMU-inspired role mixers, not a reproduction of the original multimodal GMU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba import MambaMixer, RMSNorm


class GatedLocalConv(nn.Module):
    def __init__(self, conv, mode, gate_normalization="sigmoid", use_kernel=True):
        super().__init__()
        if mode not in ("gated_silu", "gmu_tanh"):
            raise ValueError("GatedLocalConv requires gated_silu or gmu_tanh")
        if gate_normalization not in ("sigmoid", "softmax"):
            raise ValueError("gate_normalization must be sigmoid or softmax")
        if conv.groups != conv.in_channels or conv.out_channels != conv.in_channels:
            raise ValueError("Expected a depthwise Conv1d")
        self.mode, self.gate_normalization = mode, gate_normalization
        self.channels, self.kernel_size = conv.in_channels, conv.kernel_size[0]
        # PyTorch Conv1d stores oldest -> current; our lag0 is the current token.
        if use_kernel:
            self.kernel = nn.Parameter(conv.weight.detach()[:, 0].flip(-1).T.contiguous())
        else:
            self.register_parameter("kernel", None)
        self.bias = nn.Parameter(conv.bias.detach().clone())
        # Keep all original/common parameter initializations identical across
        # local-mixer choices under the same seed. Only gate params are new.
        with torch.random.fork_rng(devices=[]):
            self.gate = nn.Linear(2 * self.channels, self.kernel_size)
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
        if self.gate_normalization == "softmax":
            gates = logits.masked_fill(~present, -torch.inf).softmax(-1)
        else:
            gates = logits.sigmoid().masked_fill(~present, 0)
        return gates.to(u.dtype)

    def forward(self, u):
        sources = self.windows(u)
        gates = self.source_gates(u, sources)
        if self.kernel is None:
            # Gates above always read raw u. Without lag-specific k_l, tanh
            # can be computed once per token and reused in all causal windows.
            features = self.windows(u.tanh()) if self.mode == "gmu_tanh" else sources
        else:
            features = sources * self.kernel.to(u.dtype)[None, None]
            if self.mode == "gmu_tanh":
                features = features.tanh()
        result = (gates.unsqueeze(-1) * features).sum(2) + self.bias.to(u.dtype)
        return F.silu(result) if self.mode == "gated_silu" else result


class GatedRoleMambaMixer(MambaMixer):
    """Original projections/SSM/output gating; replace Conv1d + its SiLU only."""
    def __init__(self, d_model, d_state=16, d_conv=4, *,
                 mode="gated_silu", gate_normalization="sigmoid", use_kernel=True):
        super().__init__(d_model, d_state, d_conv)
        self.local = GatedLocalConv(self.conv1d, mode, gate_normalization, use_kernel)
        del self.conv1d  # no unused original convolution parameters or second conv

    def forward(self, x):
        u, z = self.in_proj(x).chunk(2, dim=-1)
        y = self.selective_scan(self.local(u))
        return self.out_proj(y * F.silu(z))


class GatedRoleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, *,
                 mode="gated_silu", gate_normalization="sigmoid", use_kernel=True):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = GatedRoleMambaMixer(d_model, d_state, d_conv,
                                        mode=mode, gate_normalization=gate_normalization,
                                        use_kernel=use_kernel)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
