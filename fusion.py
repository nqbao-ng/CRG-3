"""Post-graph plain Bidirectional Modality-Aware Mamba (spec section 10)."""
import torch
import torch.nn as nn

from .mamba import MambaBlock

TEXT, AUDIO, VISUAL = 0, 1, 2


class BiMAMamba(nn.Module):
    """Scans the 3-token sequence [text, audio, visual] of EACH utterance, forward and backward.

    Utterances are never concatenated along time ([B, 3N, d] is never formed), so no state flows
    between utterances; temporal context comes only from the causal graph.
    """

    def __init__(self, d, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.forward_block = MambaBlock(d, d_state, d_conv, expand, num_modalities=3)
        self.backward_block = MambaBlock(d, d_state, d_conv, expand, num_modalities=3)

    def forward(self, q_text, q_audio, q_visual, valid_mask):
        B, N, d = q_text.shape
        x = torch.stack([q_text, q_audio, q_visual], dim=2)[valid_mask]      # [N_valid, 3, d]
        ids = torch.tensor([TEXT, AUDIO, VISUAL], device=x.device).expand(x.shape[0], 3)

        y_forward = self.forward_block(x, ids)
        # flip tokens AND modality ids, then flip the output back to canonical [t, a, v]
        y_backward = self.backward_block(x.flip(1), ids.flip(1)).flip(1)

        fused = torch.cat([y_forward, y_backward], dim=-1).reshape(x.shape[0], 6 * d)
        out = fused.new_zeros(B, N, 6 * d)
        return out.index_put(valid_mask.nonzero(as_tuple=True), fused)       # [B, N, 6d]
