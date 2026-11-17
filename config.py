from dataclasses import InitVar, dataclass
from math import isfinite


GRAPH_UPDATE_MODES = ("attention_joint",)
TEXT_ANCHORED_GMU_MODES = ("gmu_text_anchor_concat", "gmu_text_anchor_interaction_concat")
FUSION_MODES = ("gmu_pairwise_concat", *TEXT_ANCHORED_GMU_MODES)
_BASELINE_READOUTS = ("sum", "concat")  # Saved baseline checkpoints.
JOINT_SSM_FUSION_MODES = ("off", "gated")
ATTENTION_HISTORY_SCOPES = ("joint", "intra")
ROLE_SCAN_MODES = ("forward", "sequential", "cagmamba")
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
    joint_ssm_fusion: str = "off"  # off | gated; gated_concat is retained for checkpoint loading
    fusion_mode: str = "gmu_pairwise_concat"
    gmu_pairwise_product: str = "on"
    decoder_mode: InitVar[str] = "classifier"  # accepted only for legacy classifier metadata
    attention_history_scope: str = "joint"  # joint T/A/V or intra-modality Same/Diff sources
    role_scan_mode: str = "forward"  # forward | sequential | cagmamba
    modality_loss_weight: float = 0.0  # legacy concat checkpoint with modality heads
    modalities: str = "tav"  # ta: legacy Text + Audio baselines

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
        if self.joint_ssm_fusion not in (*JOINT_SSM_FUSION_MODES, 'gated_concat'):
            raise ValueError(f"joint_ssm_fusion must be one of {JOINT_SSM_FUSION_MODES}")
        if self.attention_history_scope not in ATTENTION_HISTORY_SCOPES:
            raise ValueError(f"attention_history_scope must be one of {ATTENTION_HISTORY_SCOPES}")
        if self.role_scan_mode not in ROLE_SCAN_MODES:
            raise ValueError(f"role_scan_mode must be one of {ROLE_SCAN_MODES}")
        if self.role_scan_mode != "forward" and self.joint_ssm_fusion != "off":
            raise ValueError("Bidirectional role scans require joint_ssm_fusion='off'")
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
        if self.fusion_mode not in (*FUSION_MODES, *_BASELINE_READOUTS):
            raise ValueError(f"Unsupported readout: {self.fusion_mode}")
        if self.gmu_pairwise_product not in ("on", "off"):
            raise ValueError("gmu_pairwise_product must be on or off")
        if self.modalities not in INPUT_MODALITY_MODES:
            raise ValueError(f"modalities must be one of {INPUT_MODALITY_MODES}")
        if self.modalities == "ta" and (self.attention_history_scope != "intra"
                or self.joint_ssm_fusion != "off"
                or self.fusion_mode not in ("sum", "concat")):
            raise ValueError("modalities='ta' requires intra history, shared SSM off, "
                             "and a sum or concat classifier")
        if not isfinite(self.modality_loss_weight) or self.modality_loss_weight < 0:
            raise ValueError("modality_loss_weight must be finite and nonnegative")
        if self.modality_loss_weight and self.fusion_mode != "concat":
            raise ValueError("Joint modality heads require the concat classifier")


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
    values.setdefault("fusion_mode", "sum")
    if values["fusion_mode"] not in (*FUSION_MODES, *_BASELINE_READOUTS):
        raise ValueError('This checkpoint uses a removed readout; load its archived source.')
    # These settings were serialized even on checkpoints whose readout never used them.
    for field in ("cross_modal_reduction", "va_loss_weight", "va_targets", "raven_beta",
                  "symmetric_shift_tau", "digemo_hard_weight", "digemo_distill_weight", "digemo_temperature"):
        values.pop(field, None)
    # Sum checkpoints may carry the former, unused confidence-fusion defaults.
    if values["fusion_mode"] == "sum":
        for field in ("fusion_aux_weight", "fusion_alpha_init", "fusion_beta_init"):
            values.pop(field, None)
    return ModelConfig(**values)
