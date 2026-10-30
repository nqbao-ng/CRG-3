"""CRG-3: GraphSmile textf0/audio/visual -> causal Graph Mamba -> ERC."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .graph import GraphEncoder
from .attention_graph import AttentionGraphEncoder
from .fusion import HyperedgeFusion
from .hypergraph import UtteranceHypergraph, UtteranceHypergraphFusion
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
            if cfg.fusion_mode in ("concat", "hypergraph_nodes"):
                # Match the original sum model's encoder initialization and
                # post-construction RNG, keeping only a new standard 3d head.
                with torch.random.fork_rng(devices=[]):
                    self.classifier[1] = nn.Linear(3 * d, num_classes)
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
        # Instantiate after the existing classifier/decoder to preserve old sum initialization.
        if cfg.fusion_mode == "hypergraph_nodes":
            self.fusion = UtteranceHypergraph(d, cfg.hypergraph_edges,
                confidence=cfg.hypergraph_confidence, num_classes=num_classes)
        elif cfg.fusion_mode == "hypergraph_fusion":
            self.fusion = UtteranceHypergraphFusion(d, cfg.hypergraph_edges,
                cfg.hypergraph_fusion_temperature, confidence=cfg.hypergraph_confidence,
                num_classes=num_classes)
        else:
            self.fusion = HyperedgeFusion(d) if cfg.fusion_mode == "hyperedge" else None
        if self.decoder_mode == "llm" and self.llm_config.freeze_graph and self.fusion is not None:
            self.fusion.requires_grad_(False)
            self.fusion.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.decoder_mode == "llm" and self.llm_config.freeze_graph:
            self.input_proj.eval()
            self.graph.eval()
            if self.fusion is not None:
                self.fusion.eval()
        return self

    def modality_features(self, batch):
        """Return T/A/V [B,N,d]; direction-2 pooling occurs in fused_features.

        hypergraph_fusion returns post-CRG nodes here, not updated node outputs.
        """
        if "graph_features" in batch:
            if self.decoder_mode != "llm" or not self.llm_config.freeze_graph:
                raise ValueError("Cached graph features require frozen-graph LLM mode")
            return batch["graph_features"], {}
        valid = batch["mask"]
        mask = valid.unsqueeze(-1)
        h = {"t": self.input_proj["t"](batch["text"]) * mask,
             "a": self.input_proj["a"](batch["audio"]) * mask,
             "v": self.input_proj["v"](batch["visual"]) * mask}
        if self.cfg.graph_update_mode == "attention_joint":
            q, diagnostics = self.graph(h, batch.get("diff"), batch.get("same"), valid,
                                        speakers=batch.get("speaker"),
                                        adjacency=batch.get("adjacency"))
        else:
            q, diagnostics = self.graph(h, batch["diff"], batch["same"], valid)
        if self.fusion is not None and self.cfg.fusion_mode != "hypergraph_fusion":
            q, fusion_info = self.fusion(q, valid)
            diagnostics["fusion"] = fusion_info
        return q, diagnostics

    def fused_features(self, batch):
        q, diagnostics = self.modality_features(batch)
        if self.cfg.fusion_mode == "hypergraph_fusion":
            fused, info = self.fusion(q, batch["mask"])
            diagnostics["fusion"] = info
            return fused, diagnostics
        if self.cfg.fusion_mode in ("concat", "hypergraph_nodes"):
            return torch.cat([q["t"], q["a"], q["v"]], dim=-1), diagnostics
        return q["t"] + q["a"] + q["v"], diagnostics

    def forward(self, batch, target_indices=None, return_aux=False):
        if self.decoder_mode == "llm":
            q, _ = self.modality_features(batch)
            return self.llm_decoder(q, batch, target_indices=target_indices)
        fused, diagnostics = self.fused_features(batch)
        logits = self.classifier(fused)
        if return_aux:
            # Tensor-only dictionary is compatible with DataParallel gather.
            return {"logits": logits, **diagnostics.get("fusion", {})}
        return logits

    def loss(self, batch, target_indices=None):
        if self.decoder_mode == "llm":
            return self(batch, target_indices=target_indices).loss
        output = self(batch, return_aux=True)
        return self.classification_losses(output, batch["label"], batch["mask"])["loss"]

    def classification_losses(self, output, labels, valid, loss_function=F.cross_entropy):
        """Same masked objective for model.loss, CLI/notebooks and DataParallel.

        Confidence never reads labels in forward. Only auxiliary CE uses labels;
        the detached routing score prevents main CE from inflating confidence.
        Apply the caller's CE class weighting/smoothing to all three heads too.
        """
        main = loss_function(output["logits"][valid].float(), labels[valid])
        if not self.cfg.hypergraph_confidence:
            return {"loss": main, "classification_loss": main}
        logits = output["hypergraph_confidence_logits"][valid].float()
        auxiliary = torch.stack([loss_function(logits[:, m], labels[valid])
                                 for m in range(3)]).mean()
        return {"loss": main + self.cfg.hypergraph_confidence_aux_weight * auxiliary,
                "classification_loss": main, "confidence_aux_loss": auxiliary}

    @torch.no_grad()
    def predict(self, batch, target_indices=None):
        if self.decoder_mode == "classifier":
            result = self(batch).argmax(-1)
            return result.masked_fill(~batch["mask"], -100)
        q, _ = self.modality_features(batch)
        return self.llm_decoder.generate_labels(q, batch, target_indices=target_indices)

    def encoder_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if name.startswith(("input_proj.", "graph.", "fusion."))}

    def load_encoder_state_dict(self, state):
        modules = [("input_proj.", self.input_proj), ("graph.", self.graph)]
        if self.fusion is not None:
            modules.append(("fusion.", self.fusion))
        for prefix, module in modules:
            module.load_state_dict({k[len(prefix):]: v for k, v in state.items()
                                    if k.startswith(prefix)}, strict=True)


def build_model(cfg, feature_dims, num_classes, **kwargs):
    """Single model factory; variants are selected by graph_update_mode."""
    return CRG3(cfg, feature_dims, num_classes, **kwargs)
