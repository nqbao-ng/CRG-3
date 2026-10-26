from dataclasses import dataclass


GRAPH_UPDATE_MODES = ("baseline", "attention_joint")
LEARNED_HYPERGRAPH_MODES = ("hypergraph_nodes", "hypergraph_fusion")
FUSION_MODES = ("sum", "concat", "hyperedge", *LEARNED_HYPERGRAPH_MODES)


@dataclass
class ModelConfig:
    model_type: str = "crg3"
    hidden: int = 256
    graph_layers: int = 2
    local_window: int = 20
    d_state: int = 16
    d_conv: int = 4
    dropout: float = 0.2
    message_mode: str = "embedding"  # retained in checkpoint schema; fixed in active experiments
    graph_rmsnorm: str = "on"
    graph_update_mode: str = "baseline"  # see GRAPH_UPDATE_MODES
    graph_gate: str = "on"  # off: joint only, retain context projection/dropout/residual
    attention_chunk_size: int = 64  # target rows processed together, per modality
    attention_checkpoint: bool = True
    attention_layout: str = "pooled_roles_v1"  # Diff/Same pooling -> Adjacent -> Target
    attention_adjacent_source: str = "updated"  # input: A, updated: recurrent B
    attention_pool_reduction: str = "mean"  # arithmetic mean after weighting
    joint_local_mixer: str = "conv"  # conv | gated_silu | gmu_tanh (joint only)
    joint_gate_normalization: str = "sigmoid"  # same scalar gates for both variants
    joint_local_kernel: str = "on"  # off: gated_silu/gmu_tanh without the learned k_l
    fusion_mode: str = "hyperedge"
    hypergraph_edges: int = 4  # shared anchors; independent mean hyperedges per utterance
    decoder_mode: str = "classifier"  # classifier | llm

    def __post_init__(self):
        if self.model_type != "crg3":
            raise ValueError("Only model_type='crg3' is supported")
        if self.graph_update_mode not in GRAPH_UPDATE_MODES:
            raise ValueError(f"graph_update_mode must be one of {GRAPH_UPDATE_MODES}")
        if self.graph_rmsnorm != "on" or self.message_mode != "embedding":
            raise ValueError("Active experiments keep embedding messages and RMSNorm on")
        if self.graph_gate not in ("on", "off"):
            raise ValueError("graph_gate must be on or off")
        if self.graph_gate == "off" and self.graph_update_mode != "attention_joint":
            raise ValueError("GateUpdate ablation is only supported in attention_joint; baseline stays unchanged")
        if self.attention_chunk_size < 1:
            raise ValueError("attention_chunk_size must be positive")
        if self.attention_layout != "pooled_roles_v1":
            raise ValueError("Only attention_layout='pooled_roles_v1' is supported")
        if self.attention_adjacent_source not in ("input", "updated"):
            raise ValueError("attention_adjacent_source must be 'input' or 'updated'")
        if self.attention_pool_reduction not in ("sum", "mean"):
            raise ValueError("attention_pool_reduction must be 'sum' or 'mean'")
        if self.joint_local_mixer not in ("conv", "gated_silu", "gmu_tanh"):
            raise ValueError("joint_local_mixer must be conv, gated_silu or gmu_tanh")
        if self.joint_gate_normalization not in ("sigmoid", "softmax"):
            raise ValueError("joint_gate_normalization must be sigmoid or softmax")
        if self.joint_local_kernel not in ("on", "off"):
            raise ValueError("joint_local_kernel must be on or off")
        if self.joint_local_kernel == "off" and (
                self.graph_update_mode != "attention_joint"
                or self.joint_local_mixer not in ("gated_silu", "gmu_tanh")):
            raise ValueError("Kernel ablation requires attention_joint with gated_silu or gmu_tanh")
        if self.graph_update_mode == "baseline" and self.joint_local_mixer != "conv":
            raise ValueError("Gated local mixers are only supported in attention_joint")
        if min(self.hidden, self.graph_layers, self.d_state, self.d_conv) < 1:
            raise ValueError("Model dimensions and graph_layers must be positive")
        if self.local_window < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid window or dropout")
        if self.decoder_mode not in ("classifier", "llm"):
            raise ValueError("decoder_mode must be 'classifier' or 'llm'")
        if self.fusion_mode in ("hypergraph_concat", "hypergraph_sum", "hypergraph_only_sum"):
            raise ValueError("This is a retired causal-prefix hypergraph mode. Use its archived source "
                             "to load old checkpoints; hypergraph_nodes is the new within-utterance "
                             "mean node-update architecture and requires new training.")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        if self.hypergraph_edges < 1:
            raise ValueError("hypergraph_edges must be positive")
        if self.fusion_mode in LEARNED_HYPERGRAPH_MODES and self.decoder_mode != "classifier":
            raise ValueError("Learned hypergraph readouts currently support the classifier decoder only")


def model_config_from_dict(values):
    """Load active configs and original baseline checkpoints without renaming weights."""
    values = dict(values)
    # New models always use d_inner=hidden. Accept old width-preserving configs,
    # but never silently load an expanded checkpoint as a different architecture.
    legacy_expand = values.pop("expand", 1)
    if legacy_expand != 1:
        raise ValueError("This checkpoint uses expand != 1. The current model fixes d_inner=hidden; "
                         "use the checkpoint's archived source to load its original architecture.")
    # Existing pooled-role checkpoints used the input snapshot (A).
    values.setdefault("attention_adjacent_source", "input")
    # Before explicit mean pooling, pooled roles used sum(alpha * value).
    values.setdefault("attention_pool_reduction", "sum")
    values.setdefault("joint_local_mixer", "conv")
    values.setdefault("joint_gate_normalization", "sigmoid")
    values.setdefault("joint_local_kernel", "on")
    if (values.get("graph_update_mode", "baseline") == "attention_joint"
            and values.get("attention_layout") != "pooled_roles_v1"):
        raise ValueError("This is an older attention_joint checkpoint with two sequential stages. "
                         "Use its archived source; pooled_roles_v1 requires a new trained checkpoint.")
    # A checkpoint written before fusion has no fusion weights and used plain sum.
    values.setdefault("fusion_mode", "sum")
    if values["fusion_mode"] == "confidence_hyperedge":
        raise ValueError("This checkpoint uses retired confidence fusion. Use its archived source; "
                         "it cannot be reinterpreted as the new private/shared hyperedge fusion.")
    # Sum checkpoints may carry the former, unused confidence-fusion defaults.
    if values["fusion_mode"] == "sum":
        for field in ("fusion_aux_weight", "fusion_alpha_init", "fusion_beta_init"):
            values.pop(field, None)
    # Historical baseline checkpoints carried this unused option.
    if values.get("graph_update_mode", "baseline") == "baseline" and not values.get("context_first_dropout", 0):
        values.pop("context_first_dropout", None)
    return ModelConfig(**values)


@dataclass
class LLMConfig:
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    device: str = "cuda:0"
    compute_dtype: str = "float16"
    lora_rank: int = 8  # zero: train projectors only
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True
    max_prompt_tokens: int = 512  # includes the three feature tokens
    freeze_graph: bool = True

    def __post_init__(self):
        if self.compute_dtype not in ("float16", "bfloat16", "float32"):
            raise ValueError("Unsupported LLM compute_dtype")
        if self.lora_rank < 0 or self.lora_alpha < 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("Invalid LoRA settings")
        if self.max_prompt_tokens < 32:
            raise ValueError("max_prompt_tokens must be at least 32")
