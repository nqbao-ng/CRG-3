"""CRG-3: GraphSmile textf0/audio/visual -> causal Graph Mamba -> ERC."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention_graph import AttentionGraphEncoder
from .affect_fusion import PairwiseGMUFusion, TextAnchoredGMUFusion
from .config import TEXT_ANCHORED_GMU_MODES
from .graph import MODALITY_FEATURES


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
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, num_classes))
        # Keep common encoder weights and the RNG state matched across fusion modes.
        with torch.random.fork_rng(devices=[]):
            if cfg.fusion_mode != "sum":
                self.classifier[1] = nn.Linear(len(self.modalities) * d, num_classes)
                if cfg.fusion_mode == "gmu_pairwise_concat":
                    self.pairwise_gmu = PairwiseGMUFusion(d, use_product=cfg.gmu_pairwise_product == "on")
                elif cfg.fusion_mode in TEXT_ANCHORED_GMU_MODES:
                    self.text_anchored_gmu = TextAnchoredGMUFusion(
                        d, pair_interaction=cfg.fusion_mode == "gmu_text_anchor_interaction_concat")
        self.modality_heads = None
        if cfg.modality_loss_weight:
            with torch.random.fork_rng(devices=[]):
                self.modality_heads = nn.ModuleList(nn.Linear(d, num_classes) for _ in self.modalities)

    def modality_features(self, batch):
        """Return one contextual CRG output per active modality, each [B,N,H]."""
        valid = batch["mask"]
        mask = valid.unsqueeze(-1)
        h = {m: self.input_proj[m](batch[MODALITY_FEATURES[m]]) * mask for m in self.modalities}
        q, diagnostics = self.graph(h, batch.get("diff"), batch.get("same"), valid,
                                    speakers=batch.get("speaker"),
                                    adjacency=batch.get("adjacency"))
        return q, diagnostics

    def _fuse(self, features, valid):
        if self.cfg.fusion_mode == "gmu_pairwise_concat":
            return self.pairwise_gmu(features).masked_fill(~valid.unsqueeze(-1), 0)
        if self.cfg.fusion_mode in TEXT_ANCHORED_GMU_MODES:
            return self.text_anchored_gmu(features).masked_fill(~valid.unsqueeze(-1), 0)
        # Plain readouts support saved baseline checkpoints.
        private = tuple(features[m] for m in self.modalities)
        if self.cfg.fusion_mode == "sum":
            return sum(private[1:], private[0])
        return torch.cat(private, dim=-1)

    def fused_features(self, batch):
        features, diagnostics = self.modality_features(batch)
        return self._fuse(features, batch["mask"]), diagnostics

    def forward(self, batch, return_aux=False):
        q, _ = self.modality_features(batch)
        logits = self.classifier(self._fuse(q, batch["mask"]))
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
                if name.startswith(("input_proj.", "graph."))}

    def load_encoder_state_dict(self, state):
        modules = [("input_proj.", self.input_proj), ("graph.", self.graph)]
        for prefix, module in modules:
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                    if k.startswith(prefix)}, strict=True)


def build_model(cfg, feature_dims, num_classes):
    """Build the CRG encoder and classifier."""
    return CRG3(cfg, feature_dims, num_classes)
