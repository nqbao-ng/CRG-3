"""CRG-3: causal graph encoding with Gated Role Scan for emotion recognition."""
from math import log

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_graph import AttentionGraphEncoder
from .graph import MODALITY_FEATURES


class SpeakerEmbedding(nn.Module):
    """Share dialogue-local speaker embeddings across modalities."""

    def __init__(self, hidden, speaker_count, *, legacy_positions=False):
        super().__init__()
        if speaker_count < 1:
            raise ValueError("Speaker embeddings require a positive training speaker count")
        self.hidden = hidden
        self.speakers = nn.Embedding(speaker_count + 1, hidden, padding_idx=speaker_count)
        nn.init.normal_(self.speakers.weight[:-1], std=0.02)
        frequencies = None
        if legacy_positions:
            frequencies = torch.exp(-log(10000.0) * torch.arange(0, hidden, 2, dtype=torch.float32) / hidden)
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, speaker_ids, valid):
        padding_idx = self.speakers.padding_idx
        known = valid & speaker_ids.ge(0) & speaker_ids.lt(padding_idx)
        speakers = self.speakers(speaker_ids.masked_fill(~known, padding_idx))
        if self.frequencies is None:
            return speakers
        positions = torch.arange(valid.shape[1], device=valid.device, dtype=torch.float32)
        angles = positions[:, None] * self.frequencies.float()
        positional = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)[:, :self.hidden]
        return speakers + positional.to(speakers.dtype)


class CRG3(nn.Module):
    def __init__(self, cfg, feature_dims, num_classes):
        super().__init__()
        self.cfg = cfg
        self.modalities = tuple(cfg.modalities)
        d = cfg.hidden
        self.input_proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(feature_dims[MODALITY_FEATURES[m]], d),
                             nn.LayerNorm(d), nn.Dropout(cfg.dropout))
            for m in self.modalities
        })
        self.graph = AttentionGraphEncoder(cfg)
        # Preserve the original concat classifier initialization and RNG state.
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, num_classes))
        with torch.random.fork_rng(devices=[]):
            self.classifier[1] = nn.Linear(len(self.modalities) * d, num_classes)
        self.utterance_embeddings = None
        if cfg.speaker_embeddings or cfg.speaker_position_embeddings:
            with torch.random.fork_rng(devices=[]):
                self.utterance_embeddings = SpeakerEmbedding(
                    d, cfg.speaker_embedding_count, legacy_positions=cfg.speaker_position_embeddings)
        self.modality_heads = None
        if cfg.modality_loss_weight:
            with torch.random.fork_rng(devices=[]):
                self.modality_heads = nn.ModuleList(nn.Linear(d, num_classes) for _ in self.modalities)

    def modality_features(self, batch):
        """Return one contextual CRG output per active modality, each [B,N,H]."""
        return self._encode(batch)

    def _encode(self, batch):
        valid = batch["mask"]
        mask = valid.unsqueeze(-1)
        h = {m: self.input_proj[m](batch[MODALITY_FEATURES[m]]) for m in self.modalities}
        if self.utterance_embeddings is not None:
            context = self.utterance_embeddings(batch["speaker"], valid)
            h = {m: features + context.to(features.dtype) for m, features in h.items()}
        h = {m: features * mask for m, features in h.items()}
        q, diagnostics = self.graph(h, batch.get("diff"), batch.get("same"), valid,
                                    speakers=batch.get("speaker"),
                                    adjacency=batch.get("adjacency"))
        return q, diagnostics

    def _fuse(self, features):
        return torch.cat([features[m] for m in self.modalities], dim=-1)

    def fused_features(self, batch):
        """Return the features read by the primary classifier."""
        features, diagnostics = self._encode(batch)
        return self._fuse(features), diagnostics

    def forward(self, batch, return_aux=False):
        q, _ = self._encode(batch)
        logits = self.classifier(self._fuse(q))
        if return_aux:
            # Tensor-only dictionary is compatible with DataParallel gather.
            output = {"logits": logits}
            if self.modality_heads is not None:
                output["modality_logits"] = torch.stack([
                    head(q[m]) for m, head in zip(self.modalities, self.modality_heads)
                ], dim=-2)
            return output
        return logits

    def loss(self, batch):
        output = self(batch, return_aux=True)
        return self.classification_losses(output, batch["label"], batch["mask"])["loss"]

    def classification_losses(self, output, labels, valid, loss_function=F.cross_entropy):
        """Masked classification loss and optional auxiliary supervision."""
        class_ids = labels[valid]
        main = loss_function(output["logits"][valid].float(), class_ids)
        losses = {"loss": main, "classification_loss": main}
        if "modality_logits" in output:
            logits = output["modality_logits"][valid].float()
            targets = class_ids[:, None].expand(-1, logits.shape[1])
            modality_ce = loss_function(logits.flatten(0, 1), targets.reshape(-1))
            losses.update(loss=main + self.cfg.modality_loss_weight * modality_ce, modality_loss=modality_ce)
        return losses

    @torch.no_grad()
    def predict(self, batch):
        result = self(batch).argmax(-1)
        return result.masked_fill(~batch["mask"], -100)

    def encoder_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(("input_proj.", "graph.", "utterance_embeddings."))}

    def load_encoder_state_dict(self, state):
        modules = [("input_proj.", self.input_proj), ("graph.", self.graph)]
        if self.utterance_embeddings is not None:
            modules.append(("utterance_embeddings.", self.utterance_embeddings))
        for prefix, module in modules:
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                    if k.startswith(prefix)}, strict=True)


def build_model(cfg, feature_dims, num_classes):
    """Build the CRG encoder and classifier."""
    return CRG3(cfg, feature_dims, num_classes)
