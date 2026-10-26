"""Dense speaker-aware causal adjacency; rows are targets, columns are sources.

Same/Diff [B,N,N] matrices apply to all T/A/V source-target modality pairs.
Adjacent [B,N,N] connects i-1 to i within the receiving modality only.
Binary adjacency defines connectivity; attention computes learned edge weights.
"""
import torch


def causal_window_mask(valid, window):
    """Reject future, self, outside-window and padded source/target nodes."""
    if window < 0:
        raise ValueError("window must be nonnegative")
    positions = torch.arange(valid.shape[1], device=valid.device)
    lag = positions[:, None] - positions[None, :]
    allowed = (lag > 0) & (lag <= window)
    return allowed[None] & valid[:, :, None] & valid[:, None, :]


def mask_dense_adjacency(adjacency, valid, window):
    """Apply causal constraints even to explicitly supplied connectivity masks."""
    allowed = causal_window_mask(valid, window)
    positions = torch.arange(valid.shape[1], device=valid.device)
    preceding = (positions[:, None] - positions[None, :]) == 1
    result = {}
    for relation in ('diff', 'same', 'adjacent'):
        matrix = adjacency[relation]
        if matrix.shape != allowed.shape:
            raise ValueError(f"{relation} adjacency must have shape {tuple(allowed.shape)}")
        result[relation] = matrix.to(device=valid.device, dtype=torch.bool) & allowed
    result['adjacent'] = result['adjacent'] & preceding[None]
    return result


def build_speaker_adjacency(speakers, valid, window):
    """Construct actual dense matrices directly from speaker IDs and time."""
    if speakers.shape != valid.shape:
        raise ValueError("speakers and valid must both have shape [B,N]")
    same_speaker = speakers[:, :, None] == speakers[:, None, :]
    allowed = causal_window_mask(valid, window)
    positions = torch.arange(valid.shape[1], device=valid.device)
    preceding = (positions[:, None] - positions[None, :]) == 1
    return {'same': allowed & same_speaker,
            'diff': allowed & ~same_speaker,
            'adjacent': allowed & preceding[None]}


def adjacency_from_neighbors(diff, same, valid, window):
    """Compatibility for old/manual batches: materialize adjacency before attention.

    Historical neighbor tables remain supported by the baseline and callers,
    but the joint layer itself attends over dense adjacency rows, not index lists.
    """
    B, N = valid.shape
    result = {}
    for relation, table in (('diff', diff), ('same', same)):
        indices = table['idx']
        present = table['mask'].bool() & (indices >= 0) & (indices < N)
        # scatter_add, rather than scatter, handles repeated padding index0
        # without overwriting a real edge to source0 with a false padding slot.
        matrix = torch.zeros(B, N, N, device=valid.device, dtype=torch.long)
        matrix.scatter_add_(2, indices.clamp(0, N - 1), present.long())
        result[relation] = matrix > 0
    result['adjacent'] = torch.ones(B, N, N, device=valid.device, dtype=torch.bool)
    return mask_dense_adjacency(result, valid, window)
