"""CRG-3: GraphSmile textf0/audio/visual -> causal Graph Mamba -> ERC."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GraphEncoder
from .attention_graph import AttentionGraphEncoder
from .config import LLMConfig
from .llm import MultimodalLLMDecoder


class CRG3(nn.Module):
    graph_encoder_class = GraphEncoder

    def __init__(self, cfg, feature_dims, num_classes, *, llm_config=None,
                 label_names=None, language_model=None, tokenizer=None):
        super().__init__()
        self.cfg = cfg
        self.decoder_mode = cfg.decoder_mode
        d = cfg.hidden
        self.input_proj = nn.ModuleDict({
            "t": nn.Sequential(nn.Linear(feature_dims["text"], d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)),
            "a": nn.Sequential(nn.Linear(feature_dims["audio"], d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)),
            "v": nn.Sequential(nn.Linear(feature_dims["visual"], d), nn.LayerNorm(d), nn.Dropout(cfg.dropout)),
        })
        encoder = self.graph_encoder_class if cfg.graph_update_mode == "baseline" else AttentionGraphEncoder
        self.graph = encoder(cfg)
        if self.decoder_mode == "classifier":
            self.classifier = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(d, num_classes))
        else:
            if label_names is None or len(label_names) != num_classes:
                raise ValueError("LLM mode requires label_names in dataset class-ID order")
            self.classifier = None
            self.llm_config = llm_config or LLMConfig()
            self.llm_decoder = MultimodalLLMDecoder(
                d, label_names, self.llm_config, language_model=language_model, tokenizer=tokenizer)
            if self.llm_config.freeze_graph:
                self.input_proj.requires_grad_(False)
                self.graph.requires_grad_(False)
                self.input_proj.eval()
                self.graph.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.decoder_mode == "llm" and self.llm_config.freeze_graph:
            self.input_proj.eval()
            self.graph.eval()
        return self

    def modality_features(self, batch):
        """Return separate contextual T/A/V features, optionally from a frozen cache."""
        if "graph_features" in batch:
            if self.decoder_mode != "llm" or not self.llm_config.freeze_graph:
                raise ValueError("Cached graph features require frozen-graph LLM mode")
            return batch["graph_features"], {}
        valid = batch["mask"]
        mask = valid.unsqueeze(-1)
        h = {"t": self.input_proj["t"](batch["text"]) * mask,
             "a": self.input_proj["a"](batch["audio"]) * mask,
             "v": self.input_proj["v"](batch["visual"]) * mask}
        q, diagnostics = self.graph(h, batch["diff"], batch["same"], valid)
        return q, diagnostics

    def fused_features(self, batch):
        q, diagnostics = self.modality_features(batch)
        return q["t"] + q["a"] + q["v"], diagnostics

    def forward(self, batch, target_indices=None):
        if self.decoder_mode == "llm":
            q, _ = self.modality_features(batch)
            return self.llm_decoder(q, batch, target_indices=target_indices)
        fused, _ = self.fused_features(batch)
        return self.classifier(fused)

    def loss(self, batch, target_indices=None):
        if self.decoder_mode == "llm":
            return self(batch, target_indices=target_indices).loss
        valid = batch["mask"]
        return F.cross_entropy(self(batch)[valid].float(), batch["label"][valid])

    @torch.no_grad()
    def predict(self, batch, target_indices=None):
        if self.decoder_mode == "classifier":
            result = self(batch).argmax(-1)
            return result.masked_fill(~batch["mask"], -100)
        q, _ = self.modality_features(batch)
        return self.llm_decoder.generate_labels(q, batch, target_indices=target_indices)

    def encoder_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(("input_proj.", "graph."))}

    def load_encoder_state_dict(self, state):
        for prefix, module in (("input_proj.", self.input_proj), ("graph.", self.graph)):
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                    if k.startswith(prefix)}, strict=True)


def build_model(cfg, feature_dims, num_classes, **kwargs):
    """Single model factory; variants are selected by graph_update_mode."""
    return CRG3(cfg, feature_dims, num_classes, **kwargs)
