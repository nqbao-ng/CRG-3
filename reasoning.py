

import torch
from torch import nn


class ReasoningFusion(nn.Module):
    """Fuse graph evidence using a learned or text-initialized summary token.

    Args:
        hidden_dim: Common feature size D.
        num_heads: Number of attention heads; must divide D.
        dropout: Attention, residual and feed-forward dropout.
        token_mode: ``learned`` for a shared trainable c; ``text`` for feature_t.
        modals: Evidence order, matching h_list: tva, tv, ta or va.
        use_gate: Apply feature-wise source gating before attention.

    forward(h_list, query=None) returns a tensor of shape (N, D).
    h_list contains M tensors of shape (N, D); query is (N, D) in text mode.
    An optional return_attention=True returns (fused, weights), with weights
    shaped (N, heads, M+1, M+1). Weights are diagnostics, not causal evidence.
    """

    def __init__(self, hidden_dim, num_heads=8, dropout=0.2,
                 token_mode="learned", modals="tva", use_gate=True):
        super().__init__()
        if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
            raise ValueError("hidden_dim must be positive and divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if token_mode not in ("learned", "text"):
            raise ValueError("token_mode must be 'learned' or 'text'")
        if modals not in ("tva", "tv", "ta", "va"):
            raise ValueError("modals must be tva, tv, ta or va")
        if token_mode == "text" and "t" not in modals:
            raise ValueError("text token requires an active text modality; use learned for va")

        self.hidden_dim = hidden_dim
        self.token_mode = token_mode
        self.modals = modals
        self.source_norm = nn.LayerNorm(hidden_dim)
        self.anchor_norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Linear(hidden_dim, hidden_dim, bias=False) if use_gate else None

        # Fixed semantic roles: summary=0, text=1, visual=2, audio=3.
        self.type_embeddings = nn.Parameter(torch.empty(1, 4, hidden_dim))
        role_ids = {"t": 1, "v": 2, "a": 3}
        self.register_buffer(
            "token_type_ids",
            torch.tensor([0] + [role_ids[m] for m in modals], dtype=torch.long),
            persistent=False,
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Initialize common parameters before the mode-specific parameter so
        # common head weights can start identically under the same random seed.
        nn.init.normal_(self.type_embeddings, std=0.02)
        if token_mode == "learned":
            self.cls_token = nn.Parameter(torch.empty(1, 1, hidden_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        else:
            self.register_parameter("cls_token", None)

    def forward(self, h_list, query=None, return_attention=False):
        if len(h_list) != len(self.modals):
            raise ValueError("h_list must contain one (N, D) tensor per active modality")
        reference = h_list[0]
        if reference.ndim != 2 or reference.shape[-1] != self.hidden_dim:
            raise ValueError("Each evidence tensor must have shape (N, hidden_dim)")
        for h in h_list[1:]:
            if h.shape != reference.shape or h.device != reference.device or h.dtype != reference.dtype:
                raise ValueError("All evidence tensors must have matching shape, device and dtype")
        n_utterances = reference.shape[0]

        # N is the attention batch dimension; M is the token dimension.
        evidence = self.source_norm(torch.stack(h_list, dim=1))  # (N, M, D)
        if self.gate is not None:
            source_weights = torch.softmax(self.gate(evidence), dim=1)
            evidence = source_weights * evidence  # Keep M tokens; do NOT sum.

        if self.token_mode == "learned":
            anchor = self.cls_token.expand(n_utterances, -1, -1)
        else:
            if query is None or query.shape != reference.shape:
                raise ValueError("text mode requires query=feature_t with shape (N, D)")
            if query.device != reference.device or query.dtype != reference.dtype:
                raise ValueError("query and evidence must have matching device and dtype")
            anchor = query.unsqueeze(1)
        anchor = self.anchor_norm(anchor)

        tokens = torch.cat((anchor, evidence), dim=1)  # (N, 1+M, D)
        tokens = tokens + self.type_embeddings.index_select(1, self.token_type_ids)
        attended, weights = self.attention(
            tokens, tokens, tokens,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        # Post-attention normalization preserves gating effects in Q/K/V.
        encoded = self.attention_norm(tokens + self.attention_dropout(attended))
        encoded = self.output_norm(
            encoded + self.output_dropout(self.feed_forward(encoded))
        )
        fused_feature = encoded[:, 0, :]  # (N, D), one prediction per utterance.
        if return_attention:
            return fused_feature, weights
        return fused_feature
