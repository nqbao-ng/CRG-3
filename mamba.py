"""Mamba with fused CUDA selective scan and an equivalent CPU/reference path.

CUDA requires mamba_ssm: fail explicitly rather than silently use the high-memory path.
Projection, convolution and recurrence are shared by both RMSNorm ablations.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    SCAN_IMPORT_ERROR = None
except (ImportError, OSError) as exc:
    selective_scan_fn = None
    SCAN_IMPORT_ERROR = str(exc)


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(x.dtype)


class MambaMixer(nn.Module):
    """in_proj -> causal depthwise conv -> SiLU -> selective SSM, gated by SiLU(z) -> out_proj.
    """

    def __init__(self, d_model, d_state=16, d_conv=4,
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        d_inner = d_model  # width-preserving: no expansion hyperparameter
        self.d_inner, self.d_state = d_inner, d_state
        self.scan_backend = "auto"  # reference is only for CPU checks/parity tests
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, groups=d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_inner, bias=True)

        # Mamba initialisation of dt and A
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(d_inner))

    def forward(self, x):
        # x: [batch, L, d_model]; causal along L
        L = x.shape[1]
        u, z = self.in_proj(x).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        y = self.selective_scan(u)
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan(self, u):
        dtype = u.dtype
        with torch.autocast(device_type=u.device.type, enabled=False):
            u = u.float()
            dt, Bm, Cm = self.x_proj(u).split([self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = F.softplus(self.dt_proj(dt))                            # [b, L, d_inner]
            A = -torch.exp(self.A_log.float())                           # [d_inner, n]
            if self.scan_backend not in ("auto", "cuda", "reference"):
                raise ValueError(f"Unknown scan backend: {self.scan_backend}")
            use_cuda = self.scan_backend == "cuda" or (u.is_cuda and self.scan_backend == "auto")
            if use_cuda:
                if not u.is_cuda or selective_scan_fn is None:
                    raise RuntimeError(f"CUDA selective scan requires working mamba_ssm. {SCAN_IMPORT_ERROR}")
                # dt already includes the bias AND softplus: do not apply either twice.
                y = selective_scan_fn(
                    u.transpose(1, 2).contiguous(), dt.transpose(1, 2).contiguous(), A,
                    Bm.transpose(1, 2).contiguous(), Cm.transpose(1, 2).contiguous(),
                    self.D.float(), z=None, delta_bias=None, delta_softplus=False,
                    return_last_state=False)
                return y.transpose(1, 2).to(dtype)
            dA = torch.exp(dt.unsqueeze(-1) * A)                         # [b, L, d_inner, n]
            dBu = dt.unsqueeze(-1) * Bm.unsqueeze(2) * u.unsqueeze(-1)   # [b, L, d_inner, n]
            h = u.new_zeros(u.shape[0], self.d_inner, self.d_state)
            ys = []
            for t in range(u.shape[1]):
                h = dA[:, t] * h + dBu[:, t]
                ys.append((h * Cm[:, t].unsqueeze(1)).sum(-1))
            y = torch.stack(ys, dim=1) + u * self.D
        return y.to(dtype)


class SelectiveStateSpace(nn.Module):
    """Selective SSM for transformed inputs, with no local convolution.

    Reuses the exact CUDA/reference recurrence of MambaMixer. Input projection,
    local interaction, output gating and projection are owned by the caller.
    """

    def __init__(self, d_model, d_state=16,
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        self.d_inner, self.d_state = d_model, d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.scan_backend = "auto"
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        nn.init.uniform_(self.dt_proj.weight, -self.dt_rank ** -0.5, self.dt_rank ** -0.5)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        dt = dt.clamp(min=dt_init_floor)
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, v):
        return MambaMixer.selective_scan(self, v)


class MambaBlock(nn.Module):
    """Residual Mamba block with optional pre-Mixer RMSNorm."""

    def __init__(self, d_model, d_state=16, d_conv=4, use_rmsnorm=True):
        super().__init__()
        self.norm = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.mixer = MambaMixer(d_model, d_state, d_conv)

    def forward(self, x):
        return x + self.mixer(self.norm(x))
