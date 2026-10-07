from dataclasses import dataclass


@dataclass
class ModelConfig:
    hidden: int = 256
    graph_layers: int = 2
    local_window: int = 20
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    dropout: float = 0.2
    post_graph_fusion: str = "mamba"  # "mamba" or "sum" ablation
    message_mode: str = "projection"  # "projection" (per relation) or "embedding" (shared projection + relation vector)
    edge_weight_mode: str = "none"  # none, dot_tanh, dot_softmax, vuemo_tanh
    edge_weight_dim: int = 32  # query/key width for dot-product modes
