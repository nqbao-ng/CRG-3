from dataclasses import InitVar, dataclass
from math import isfinite


GRAPH_UPDATE_MODES = ("attention_joint",)
FUSION_MODES = ("concat", "text_anchor_spatial_concat", "tav_anchor_spatial_concat")
ATTENTION_HISTORY_SCOPES = ("joint", "intra")
INPUT_MODALITY_MODES = ("tav", "ta")


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
    graph_update_mode: str = "attention_joint"  # see GRAPH_UPDATE_MODES
    graph_gate: str = "on"  # off: retain context projection/dropout/residual
    attention_chunk_size: int = 64  # target rows processed together, per modality
    attention_layout: str = "pooled_roles_v1"  # Diff/Same pooling -> Adjacent -> Target
    attention_adjacent_source: str = "updated"  # input: A, updated: recurrent B
    attention_pool_reduction: str = "mean"  # legacy mean | joint sum | modality_mean (separate softmax + mean)
    attention_score: str = "dot_product"  # dot_product | gat (additive scorer only)
    joint_local_mixer: str = "gmu_tanh_no_uz"
    joint_gate_normalization: str = "sigmoid"  # fixed scalar local gates
    joint_local_kernel: str = "off"  # fixed: no learned lag kernel
    joint_ssm_fusion: str = "off"  # fixed: private SSMs
    fusion_mode: str = "concat"
    tav_anchor_all_peers: bool = False  # checkpoint metadata for TAVASC topology
    decoder_mode: InitVar[str] = "classifier"  # accepted only for legacy classifier metadata
    attention_history_scope: str | None = None  # defaults to joint for concat and intra for anchors
    role_scan_mode: str = "forward"  # fixed: scan in role order
    modality_loss_weight: float = 0.0  # legacy concat checkpoint with modality heads
    modalities: str = "tav"  # ta: legacy Text + Audio baselines
    speaker_embeddings: bool = False
    speaker_position_embeddings: bool = False  # legacy speaker + position checkpoints only
    speaker_embedding_count: int = 0  # resolved from the training dialogues

    def __post_init__(self, decoder_mode):
        if self.model_type != "crg3":
            raise ValueError("Only model_type='crg3' is supported")
        if self.graph_update_mode not in GRAPH_UPDATE_MODES:
            raise ValueError(f"graph_update_mode must be one of {GRAPH_UPDATE_MODES}")
        if self.graph_rmsnorm != "on" or self.message_mode != "embedding":
            raise ValueError("Active experiments keep embedding messages and RMSNorm on")
        if self.graph_gate not in ("on", "off"):
            raise ValueError("graph_gate must be on or off")
        if self.attention_chunk_size < 1:
            raise ValueError("attention_chunk_size must be positive")
        if self.attention_layout != "pooled_roles_v1":
            raise ValueError("Only attention_layout='pooled_roles_v1' is supported")
        if self.attention_adjacent_source not in ("input", "updated"):
            raise ValueError("attention_adjacent_source must be 'input' or 'updated'")
        if self.attention_pool_reduction not in ("sum", "mean", "modality_mean"):
            raise ValueError("attention_pool_reduction must be 'sum', 'mean' or 'modality_mean'")
        if self.attention_score not in ("dot_product", "gat"):
            raise ValueError("attention_score must be 'dot_product' or 'gat'")
        if self.fusion_mode not in FUSION_MODES:
            raise ValueError(f"fusion_mode must be one of {FUSION_MODES}")
        if self.attention_history_scope is None:
            self.attention_history_scope = "joint" if self.fusion_mode == "concat" else "intra"
        if self.attention_history_scope not in ATTENTION_HISTORY_SCOPES:
            raise ValueError(f"attention_history_scope must be one of {ATTENTION_HISTORY_SCOPES}")
        if self.fusion_mode != "concat" and self.attention_history_scope != "intra":
            raise ValueError("TASC and TAVASC require intra history. Use the original source for joint-history anchors.")
        if self.joint_ssm_fusion != "off":
            raise ValueError("Shared SSM is retired. Use the original experiment source.")
        if self.role_scan_mode != "forward":
            raise ValueError("Bidirectional role scan is retired. Use the original experiment source.")
        if self.tav_anchor_all_peers:
            raise ValueError("TAVASC with all peers is retired. Use the original experiment source.")
        if self.joint_local_mixer != "gmu_tanh_no_uz":
            raise ValueError("Only the gmu_tanh_no_uz mixer is supported")
        if self.joint_local_kernel != "off" or self.joint_gate_normalization != "sigmoid":
            raise ValueError("The retained mixer requires no learned kernel and sigmoid local gates")
        if min(self.hidden, self.graph_layers, self.d_state, self.d_conv) < 1:
            raise ValueError("Model dimensions and graph_layers must be positive")
        if self.local_window < 0 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid window or dropout")
        if decoder_mode != "classifier":
            raise ValueError("Only classifier checkpoints are supported")
        if self.modalities not in INPUT_MODALITY_MODES:
            raise ValueError(f"modalities must be one of {INPUT_MODALITY_MODES}")
        if self.modalities == "ta" and (self.attention_history_scope != "intra"
                or self.fusion_mode != "concat"):
            raise ValueError("modalities='ta' requires intra history and a concat classifier")
        if not isfinite(self.modality_loss_weight) or self.modality_loss_weight < 0:
            raise ValueError("modality_loss_weight must be finite and nonnegative")
        if self.modality_loss_weight and self.fusion_mode != "concat":
            raise ValueError("Joint modality heads require the concat classifier")
        if self.speaker_embedding_count < 0:
            raise ValueError("speaker_embedding_count must be nonnegative")


def model_config_from_dict(values):
    """Load retained no-u/z checkpoints without reinterpreting retired architectures."""
    values = dict(values)
    if values.pop("role_order", "diff_same_adjacent_target") != "diff_same_adjacent_target":
        raise ValueError("This checkpoint uses a retired role order. Load it with its original experiment source.")
    # Retired runtime fields do not affect weights or recurrence. All restored
    # models now use PyTorch without CRG checkpointing. Use archived source to
    # reproduce historical fused-kernel execution exactly.
    old_backend = values.pop('scan_backend', 'reference')
    if old_backend not in ('auto', 'cuda', 'reference'):
        raise ValueError('Unrecognized historical scan backend')
    old_checkpoint = values.pop('attention_checkpoint', False)
    if not isinstance(old_checkpoint, bool):
        raise ValueError('Historical attention_checkpoint must be boolean')
    # Adjacent is always enabled. Never reinterpret the retired ablation as
    # the retained architecture when restoring a historical checkpoint.
    if values.pop('attention_adjacent', True) is not True:
        raise ValueError('The no-Adjacent ablation is retired; load its archived source.')
    # Old CRG-only checkpoints contain unused fields. Accept those configs, but
    # never reinterpret a trained removed architecture as plain CRG.
    if values.get('input_hypergraph', False) or values.get('hypergraph_confidence', False):
        raise ValueError('This checkpoint uses a removed HG architecture; load its archived source.')
    for field in ('input_hypergraph', 'input_hypergraph_layout', 'input_hypergraph_edges',
                  'hypergraph_edges', 'hypergraph_fusion_temperature',
                  'hypergraph_confidence', 'hypergraph_confidence_aux_weight'):
        values.pop(field, None)
    # Accept the retired flag only for checkpoints using the retained behavior.
    relation_placement = values.pop("attention_relation_embedding", "group")
    if relation_placement != "group":
        raise ValueError("Source relation embedding is retired. Use the archived source "
                         "to load its checkpoint; current CRG adds relation embeddings after pooling.")
    if (values.get('graph_update_mode', 'baseline') != 'attention_joint'
            or values.get('joint_local_mixer', 'conv') != 'gmu_tanh_no_uz'):
        raise ValueError('This checkpoint uses a retired u/z architecture or mixer; load its original source.')
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
    values.setdefault("attention_score", "dot_product")
    values.setdefault("attention_history_scope", "joint")
    values.setdefault("joint_ssm_fusion", "off")

    values.setdefault("joint_gate_normalization", "sigmoid")
    values.setdefault("joint_local_kernel", "off")
    if (values.get("graph_update_mode", "baseline") == "attention_joint"
            and values.get("attention_layout") != "pooled_roles_v1"):
        raise ValueError("This is an older attention_joint checkpoint with two sequential stages. "
                         "Use its archived source; pooled_roles_v1 requires a new trained checkpoint.")
    # A checkpoint written before fusion has no fusion weights and used plain sum.
    if values.get("fusion_mode", "sum") not in FUSION_MODES:
        raise ValueError('This checkpoint uses a removed readout; load its archived source.')
    if values.get("fusion_mode") == "tav_anchor_spatial_concat":
        # Older checkpoints have two incoming cross-modal messages per target.
        values.setdefault("tav_anchor_all_peers", True)
    # These settings were serialized even on checkpoints whose readout never used them.
    for field in ("cross_modal_reduction", "va_loss_weight", "va_targets", "raven_beta",
                  "symmetric_shift_tau", "digemo_hard_weight", "digemo_distill_weight", "digemo_temperature",
                  "gmu_pairwise_product", "fafm_residual", "pairwise_ct_candidate_only",
                  "tav_anchor_skip_all", "tav_anchor_av_skip_count"):
        values.pop(field, None)
    return ModelConfig(**values)
