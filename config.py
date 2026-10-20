from dataclasses import dataclass


GRAPH_UPDATE_MODES = ("baseline", "attention_joint", "attention_per_modality", "attention_history")


@dataclass
class ModelConfig:
    model_type: str = "crg3"
    hidden: int = 256
    graph_layers: int = 2
    local_window: int = 20
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.2
    message_mode: str = "embedding"  # retained in checkpoint schema; fixed in active experiments
    graph_rmsnorm: str = "on"
    graph_update_mode: str = "baseline"  # see GRAPH_UPDATE_MODES
    graph_gate: str = "on"
    attention_chunk_size: int = 64  # target rows processed together, per modality
    attention_checkpoint: bool = True
    decoder_mode: str = "classifier"  # classifier | llm

    def __post_init__(self):
        if self.model_type != "crg3":
            raise ValueError("Only model_type='crg3' is supported")
        if self.graph_update_mode not in GRAPH_UPDATE_MODES:
            raise ValueError(f"graph_update_mode must be one of {GRAPH_UPDATE_MODES}")
        if self.graph_gate != "on" or self.graph_rmsnorm != "on" or self.message_mode != "embedding":
            raise ValueError("Active experiments keep embedding messages, RMSNorm and GateUpdate on")
        if self.attention_chunk_size < 1:
            raise ValueError("attention_chunk_size must be positive")
        if min(self.hidden, self.graph_layers, self.d_state, self.d_conv, self.expand) < 1:
            raise ValueError("Model dimensions and graph_layers must be positive")
        if self.local_window < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid window or dropout")
        if self.decoder_mode not in ("classifier", "llm"):
            raise ValueError("decoder_mode must be 'classifier' or 'llm'")


def model_config_from_dict(values):
    """Load active configs and original baseline checkpoints without renaming weights."""
    values = dict(values)
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
