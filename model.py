"""CRG-3: GraphSmile textf0/audio/visual -> causal Graph Mamba -> ERC."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fusion import BiMAMamba
from .graph import GraphEncoder


class CRG3(nn.Module):
    def __init__(self, cfg, feature_dims, num_classes):
        super().__init__()
        d = cfg.hidden
        self.input_proj = nn.ModuleDict({
            "t": nn.Sequential(nn.Linear(feature_dims["text"], d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)),
            "a": nn.Sequential(nn.Linear(feature_dims["audio"], d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)),
            "v": nn.Sequential(nn.Linear(feature_dims["visual"], d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)),
        })
        self.graph = GraphEncoder(cfg)
        if cfg.post_graph_fusion == "mamba":
            self.fusion = BiMAMamba(d, cfg.d_state, cfg.d_conv, cfg.expand)
            classifier_dim = 6 * d
        elif cfg.post_graph_fusion == "sum":
            self.fusion = None
            classifier_dim = d
        else:
            raise ValueError(f"Unknown post-graph fusion: {cfg.post_graph_fusion}")
        self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(classifier_dim, num_classes))

    def fused_features(self, batch):
        valid = batch["mask"]
        mask = valid.unsqueeze(-1)
        h = {"t": self.input_proj["t"](batch["text"]) * mask,
             "a": self.input_proj["a"](batch["audio"]) * mask,
             "v": self.input_proj["v"](batch["visual"]) * mask}
        q, diagnostics = self.graph(h, batch["diff"], batch["same"], valid)
        if self.fusion is None:
            return q["t"] + q["a"] + q["v"], diagnostics
        return self.fusion(q["t"], q["a"], q["v"], valid), diagnostics

    def forward(self, batch):
        fused, _ = self.fused_features(batch)
        return self.classifier(fused)

    def loss(self, batch):
        valid = batch["mask"]
        return F.cross_entropy(self(batch)[valid].float(), batch["label"][valid])
