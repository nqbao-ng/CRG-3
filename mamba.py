"""Mamba with fused CUDA selective scan and an equivalent CPU/reference path.

CUDA requires mamba_ssm: fail explicitly rather than silently use the high-memory path.
Weights, projections, convolution, recurrence and modality experts remain unchanged.
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


class ExpertLinear(nn.Module):
    """Deterministic modality-aware projection: shared expert + expert of the token's modality.

    Both experts run in parallel and are summed; routing is fixed by the modality id (no router).
    """

    def __init__(self, in_dim, out_dim, num_modalities=3, bias=False):
        super().__init__()
        self.shared = nn.Linear(in_dim, out_dim, bias=bias)
        self.specific = nn.ModuleList([nn.Linear(in_dim, out_dim, bias=bias) for _ in range(num_modalities)])

    def forward(self, x, modality_ids):
        # x: [..., L, in_dim], modality_ids: [..., L]
        all_specific = torch.stack([expert(x) for expert in self.specific], dim=-2)   # [...,L,M,out]
        route = F.one_hot(modality_ids, len(self.specific)).to(all_specific.dtype)[..., None]
        return self.shared(x) + (all_specific * route).sum(-2)


class MambaMixer(nn.Module):
    """in_proj -> causal depthwise conv -> SiLU -> selective SSM, gated by SiLU(z) -> out_proj.

    With `num_modalities > 0`, in_proj/out_proj become ExpertLinear (Modality-Aware Mamba);
    conv and SSM core stay shared.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_modalities=0,
                 dt_min=1e-3, dt_max=1e-1, dt_init_floor=1e-4):
        super().__init__()
        d_inner = expand * d_model
        self.d_inner, self.d_state = d_inner, d_state
        self.scan_backend = "auto"  # reference is only for CPU checks/parity tests
        self.dt_rank = math.ceil(d_model / 16)
        if num_modalities:
            self.in_proj = ExpertLinear(d_model, 2 * d_inner, num_modalities)
            self.out_proj = ExpertLinear(d_inner, d_model, num_modalities)
        else:
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

    def _proj(self, layer, x, modality_ids):
        return layer(x, modality_ids) if isinstance(layer, ExpertLinear) else layer(x)

    def forward(self, x, modality_ids=None):
        # x: [batch, L, d_model]; causal along L
        L = x.shape[1]
        u, z = self._proj(self.in_proj, x, modality_ids).chunk(2, dim=-1)
        u = self.conv1d(u.transpose(1, 2))[..., :L].transpose(1, 2)
        u = F.silu(u)
        y = self.selective_scan(u)
        y = y * F.silu(z)
        return self._proj(self.out_proj, y, modality_ids)

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


class MambaBlock(nn.Module):
    """Standard Mamba residual block: x + Mixer(RMSNorm(x))."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, num_modalities=0):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = MambaMixer(d_model, d_state, d_conv, expand, num_modalities)

    def forward(self, x, modality_ids=None, token_weights=None):
        normalized = self.norm(x)
        if token_weights is not None:
            # Weight AFTER RMSNorm so normalization cannot cancel edge magnitude.
            # The read token has weight 1; the residual path stays unchanged.
            normalized = normalized * token_weights.unsqueeze(-1).to(normalized.dtype)
        return x + self.mixer(normalized, modality_ids)
