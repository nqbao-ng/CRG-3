"""
predict.py

Utilities for the DiGemo model loaded from a saved checkpoint.

Layout:
    1. Shared checkpoint loading (rebuild model + dummy inputs).
    2. Task functions:
        - calculate_complexity(): params, GFLOPs, inference time, peak memory.
        - predict():              run a forward pass and return predicted labels.
        - plot_confusion_from_checkpoint(): evaluate the test set and plot a
          confusion matrix.

Quick start:
    python predict.py --checkpoint ./checkpoints/best_model_IEMOCAP_260.pth
    python predict.py --checkpoint ./checkpoints/best_model_IEMOCAP_260.pth --task confusion
    python predict.py --checkpoint <path> --task confusion --feature_path ./features/iemocap_multi_features.pkl
"""

import os
import time
import argparse
import statistics

import torch

from model import DiGemo

try:
    from thop import profile
    _HAS_THOP = True
except Exception:  # thop not installed
    _HAS_THOP = False


# ---------------------------------------------------------------------------
# Dataset config: embedding_dims = [text, visual, audio]
# n_speakers follows model.py (n_classes in {4, 6} -> 2, else 9).
# default_feature: default .pkl path used to build the test set.
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    "IEMOCAP":   {"embedding_dims": [1024, 342, 1582], "n_classes": 6, "n_speakers": 2,
                  "default_feature": "./features/iemocap_multi_features.pkl"},
    "IEMOCAP4":  {"embedding_dims": [1024, 512, 100],  "n_classes": 4, "n_speakers": 2,
                  "default_feature": None},
    "MELD":      {"embedding_dims": [1024, 342, 300],  "n_classes": 7, "n_speakers": 9,
                  "default_feature": "./features/meld_multi_features.pkl"},
    "CMUMOSEI7": {"embedding_dims": [1024, 35, 384],   "n_classes": 7, "n_speakers": 9,
                  "default_feature": None},
}


# ===========================================================================
# 1. SHARED CHECKPOINT LOADING
# ===========================================================================
def _safe_torch_load(path, map_location):
    """torch.load compatible across versions (checkpoint holds an argparse.Namespace)."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict):
    """Strip the 'module.' prefix if the checkpoint was saved from DDP."""
    if any(k.startswith("module.") for k in state_dict):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def resolve_dataset_config(args):
    """Get embedding_dims / n_classes / n_speakers from args.dataset."""
    dataset = getattr(args, "dataset", None)
    if dataset not in DATASET_CONFIG:
        raise ValueError(
            f"Dataset '{dataset}' is not supported. "
            f"Valid options: {list(DATASET_CONFIG.keys())}"
        )
    cfg = DATASET_CONFIG[dataset]
    return cfg["embedding_dims"], cfg["n_classes"], cfg["n_speakers"]


def load_model_from_checkpoint(checkpoint_path, device=None):
    """
    Load checkpoint -> rebuild DiGemo -> load weights -> eval().

    Returns:
        model, args, embedding_dims, n_classes, n_speakers, device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path, map_location=device)
    args = checkpoint["args"]

    # Keep no_cuda in sync with the real device to avoid internal .cuda() errors.
    args.no_cuda = (device.type == "cpu")

    embedding_dims, n_classes, n_speakers = resolve_dataset_config(args)

    model = DiGemo(args, embedding_dims, n_classes).to(device)
    state_dict = _strip_module_prefix(checkpoint["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    return model, args, embedding_dims, n_classes, n_speakers, device


def build_dummy_inputs(embedding_dims, n_speakers, seq_len=50, batch_size=1, device="cpu"):
    """
    Build dummy inputs matching DiGemo.forward:
        forward(feature_t, feature_v, feature_a, umask, qmask, dia_lengths)
    """
    device = torch.device(device)
    t_feat = torch.randn(seq_len, batch_size, embedding_dims[0], device=device)
    v_feat = torch.randn(seq_len, batch_size, embedding_dims[1], device=device)
    a_feat = torch.randn(seq_len, batch_size, embedding_dims[2], device=device)

    umask = torch.ones(seq_len, batch_size, device=device)

    qmask = torch.zeros(seq_len, batch_size, n_speakers, device=device)
    qmask[:, :, 0] = 1.0

    dia_lengths = [seq_len] * batch_size

    return (t_feat, v_feat, a_feat, umask, qmask, dia_lengths)


# ===========================================================================
# 2. COMPLEXITY MEASUREMENT
# ===========================================================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops_params(model, inputs):
    if not _HAS_THOP:
        print("[!] 'thop' not installed (pip install thop) -> skipping FLOPs.")
        return None, None
    with torch.no_grad():
        flops, params = profile(model, inputs=inputs, verbose=False)
    return flops, params


@torch.no_grad()
def measure_inference_time(model, inputs, warmup=20, runs=100, device="cpu"):
    device = torch.device(device)
    is_cuda = device.type == "cuda"

    for _ in range(warmup):
        model(*inputs)
    if is_cuda:
        torch.cuda.synchronize()

    timings = []
    for _ in range(runs):
        if is_cuda:
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            model(*inputs)
            ender.record()
            torch.cuda.synchronize()
            timings.append(starter.elapsed_time(ender))  # ms
        else:
            t0 = time.perf_counter()
            model(*inputs)
            timings.append((time.perf_counter() - t0) * 1000.0)  # ms

    mean_ms = statistics.mean(timings)
    std_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
    return {
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "fps": 1000.0 / mean_ms if mean_ms > 0 else float("inf"),
    }


@torch.no_grad()
def measure_peak_memory(model, inputs, device="cpu"):
    device = torch.device(device)
    if device.type != "cuda":
        return None

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    model(*inputs)
    torch.cuda.synchronize()

    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    return {"peak_allocated_mb": peak_alloc, "peak_reserved_mb": peak_reserved}


def calculate_complexity(checkpoint_path, seq_len=50, batch_size=1,
                         warmup=20, runs=100, device=None, verbose=True):
    """TASK 1: params, GFLOPs, inference time, peak memory."""
    model, args, embedding_dims, n_classes, n_speakers, device = \
        load_model_from_checkpoint(checkpoint_path, device)

    inputs = build_dummy_inputs(embedding_dims, n_speakers,
                                seq_len=seq_len, batch_size=batch_size, device=device)

    total_params, trainable_params = count_parameters(model)
    flops, _ = compute_flops_params(model, inputs)
    timing = measure_inference_time(model, inputs, warmup=warmup, runs=runs, device=device)
    mem = measure_peak_memory(model, inputs, device=device)

    result = {
        "checkpoint": checkpoint_path,
        "dataset": getattr(args, "dataset", "unknown"),
        "device": str(device),
        "seq_len": seq_len,
        "batch_size": batch_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops,
        "gflops": (flops / 1e9) if flops is not None else None,
        "timing": timing,
        "memory": mem,
    }

    if verbose:
        _print_report(result)
    return result


def _print_report(r):
    print("=" * 60)
    print(" DiGemo - Model Complexity Report")
    print("=" * 60)
    print(f" Checkpoint   : {r['checkpoint']}")
    print(f" Dataset      : {r['dataset']}")
    print(f" Device       : {r['device']}")
    print(f" Input shape  : seq_len={r['seq_len']}, batch_size={r['batch_size']}")
    print("-" * 60)
    print(f" Params (total)     : {r['total_params'] / 1e6:.3f} M")
    print(f" Params (trainable) : {r['trainable_params'] / 1e6:.3f} M")
    if r["gflops"] is not None:
        print(f" FLOPs              : {r['gflops']:.3f} GFLOPs")
    else:
        print(f" FLOPs              : N/A (install 'thop' to compute)")
    t = r["timing"]
    print(f" Inference time     : {t['mean_ms']:.3f} ± {t['std_ms']:.3f} ms "
          f"(min {t['min_ms']:.3f} / max {t['max_ms']:.3f})")
    print(f" Throughput         : {t['fps']:.2f} forward/s")
    if r["memory"] is not None:
        m = r["memory"]
        print(f" Peak memory (alloc): {m['peak_allocated_mb']:.2f} MB")
        print(f" Peak memory (resv) : {m['peak_reserved_mb']:.2f} MB")
    else:
        print(f" Peak memory        : N/A (only measurable on CUDA)")
    print("=" * 60)


# ===========================================================================
# 3. PREDICT - run a forward pass and return predicted labels
# ===========================================================================
@torch.no_grad()
def predict(checkpoint_path, inputs=None, seq_len=50, batch_size=1, device=None):
    """Load checkpoint and run prediction. inputs=None -> use dummy inputs."""
    model, args, embedding_dims, n_classes, n_speakers, device = \
        load_model_from_checkpoint(checkpoint_path, device)

    if inputs is None:
        inputs = build_dummy_inputs(embedding_dims, n_speakers,
                                    seq_len=seq_len, batch_size=batch_size, device=device)
    else:
        inputs = tuple(x.to(device) if torch.is_tensor(x) else x for x in inputs)

    fused_logit, t_logit, v_logit, a_logit, fused_feature = model(*inputs)
    pred_labels = torch.argmax(fused_logit, dim=-1)
    return pred_labels, fused_logit


# ===========================================================================
# 4. CONFUSION MATRIX - evaluate the test set and plot
# ===========================================================================
def _build_test_loader(args, feature_path=None, batch_size=16):
    """Build the test DataLoader based on args.dataset."""
    from torch.utils.data import DataLoader
    from dataloader import IEMOCAPDataset_BERT, MELDDataset_BERT

    dataset = args.dataset
    cfg = DATASET_CONFIG.get(dataset, {})
    path = feature_path or cfg.get("default_feature")

    if path is None:
        raise ValueError(
            f"No default feature path for dataset '{dataset}'. "
            f"Please pass --feature_path."
        )
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature file not found: {path}")

    if dataset == "IEMOCAP":
        testset = IEMOCAPDataset_BERT(path, train=False)
    elif dataset == "MELD":
        testset = MELDDataset_BERT(path, train=False)
    else:
        raise ValueError(
            f"Confusion matrix currently supports IEMOCAP and MELD only "
            f"(current dataset: '{dataset}')."
        )

    loader = DataLoader(
        testset,
        batch_size=batch_size,
        collate_fn=testset.collate_fn,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return loader


@torch.no_grad()
def evaluate_testset(model, dataloader, device):
    """
    Run the model over the whole test set and collect true/predicted labels.
    The dia_lengths / label handling mirrors trainer.train_or_eval_model (eval mode).
    Returns (labels, preds) as 1-D numpy arrays at the utterance level.
    """
    import numpy as np
    import torch.nn.functional as F

    model.eval()
    all_preds, all_labels = [], []

    for data in dataloader:
        # data = [textf, visuf, acouf, qmask, umask, label_emotion, vids]
        textf, visuf, acouf, qmask, umask, label_emotion = \
            [d.to(device) for d in data[:-1]]

        dia_lengths, label_emotions = [], []
        for j in range(umask.size(1)):
            length = (umask[:, j] == 1).nonzero().tolist()[-1][0] + 1
            dia_lengths.append(length)
            label_emotions.append(label_emotion[:length, j])
        label_emo = torch.cat(label_emotions)

        fused_logit, t_logit, v_logit, a_logit, fused_feature = \
            model(textf, visuf, acouf, umask, qmask, dia_lengths)

        fused_prob = F.log_softmax(fused_logit, -1)
        preds = torch.argmax(fused_prob, 1)

        all_preds.append(preds.cpu().numpy())
        all_labels.append(label_emo.cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    return labels, preds


# Short class labels + x-axis suffix per dataset (matching the reference figure).
CM_LABELS = {
    "IEMOCAP": (["Hap", "Sad", "Neu", "Ang", "Exc", "Fru"], "IEMOCAP"),
    "MELD":    (["Neu", "Sur", "Fea", "Sad", "Joy", "Dis", "Ang"], "MELD"),
}


def plot_confusion_matrix_styled(matrix, dataset, file_name,
                                 save_dir="results/confusion_matrix",
                                 img_format="pdf"):
    """
    Plot the confusion matrix in the reference style:
        - GnBu colormap (white -> green -> blue)
        - color encodes the row-normalized fraction (0..1)
        - cells display PERCENTAGES (2 decimal places)
        - short labels, horizontal x-ticks, no title, no grid lines, serif font

    matrix: raw count matrix, shape (C, C).
    Returns the saved file path.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs(save_dir, exist_ok=True)

    n = matrix.shape[0]
    label_names, suffix = CM_LABELS.get(
        dataset, ([str(i) for i in range(n)], dataset)
    )

    matrix = np.asarray(matrix, dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    frac = matrix / (row_sums + 1e-9)   # 0..1  -> used for COLOR
    annot = frac * 100.0                # %     -> used for TEXT

    # Serif font like the reference (falls back if Times New Roman is missing).
    try:
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    except Exception:
        pass

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(
        frac,                 # color by 0..1 fraction
        annot=annot,          # but print percentages
        fmt=".2f",
        cmap="GnBu",
        cbar=True,
        vmin=0.0,
        linewidths=0,
        xticklabels=label_names,
        yticklabels=label_names,
        square=True,
        annot_kws={"fontsize": 11},
    )
    ax.set_xlabel(f"Predict Label ({suffix})", fontsize=13)
    ax.set_ylabel("True Label", fontsize=13)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{file_name}.{img_format}")
    plt.savefig(save_path, format=img_format, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path


def plot_confusion_from_checkpoint(checkpoint_path, feature_path=None, batch_size=16,
                                   device=None, file_name=None,
                                   save_dir="results/confusion_matrix",
                                   img_format="pdf", verbose=True):
    """
    TASK 2: load checkpoint -> evaluate test set -> plot confusion matrix.

    Returns (confusion_matrix, labels, preds).
    File saved at: {save_dir}/{file_name}.{img_format}
    """
    from sklearn.metrics import confusion_matrix, classification_report

    model, args, embedding_dims, n_classes, n_speakers, device = \
        load_model_from_checkpoint(checkpoint_path, device)

    loader = _build_test_loader(args, feature_path=feature_path, batch_size=batch_size)
    labels, preds = evaluate_testset(model, loader, device)

    cm = confusion_matrix(labels, preds, labels=list(range(n_classes)))

    if verbose:
        print(classification_report(labels, preds, digits=4, zero_division=0))
        print("Confusion matrix (counts):")
        print(cm)

    if file_name is None:
        seed = getattr(args, "seed", "x")
        file_name = f"conf_{args.dataset}_{seed}"

    out_path = plot_confusion_matrix_styled(
        cm, args.dataset, file_name, save_dir=save_dir, img_format=img_format
    )
    print(f"[+] Saved confusion matrix: {out_path}")

    return cm, labels, preds


# ===========================================================================
# CLI
# ===========================================================================
def parse_args():
    p = argparse.ArgumentParser(description="DiGemo predict / complexity / confusion-matrix utility")
    p.add_argument("--checkpoint", type=str,
                   default="./checkpoints/best_model_IEMOCAP_260.pth",
                   help="Path to the checkpoint .pth file")
    p.add_argument("--task", choices=["complexity", "predict", "confusion"],
                   default="complexity", help="Task to run")
    p.add_argument("--seq_len", type=int, default=50, help="Dummy sequence length (complexity/predict)")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size")
    p.add_argument("--warmup", type=int, default=20, help="Warmup iterations for timing")
    p.add_argument("--runs", type=int, default=100, help="Measured iterations for inference timing")
    p.add_argument("--feature_path", type=str, default=None,
                   help="Path to the test feature .pkl (task=confusion)")
    p.add_argument("--save_dir", type=str, default="results/confusion_matrix",
                   help="Directory to save the confusion matrix")
    p.add_argument("--file_name", type=str, default=None,
                   help="Confusion matrix file name (default conf_{dataset}_{seed})")
    p.add_argument("--img_format", choices=["pdf", "png"], default="pdf",
                   help="Confusion matrix image format")
    p.add_argument("--cpu", action="store_true", help="Force running on CPU")
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    dev = "cpu" if cli.cpu else None

    if cli.task == "complexity":
        calculate_complexity(
            cli.checkpoint,
            seq_len=cli.seq_len,
            batch_size=cli.batch_size,
            warmup=cli.warmup,
            runs=cli.runs,
            device=dev,
        )

    elif cli.task == "predict":
        preds, logits = predict(
            cli.checkpoint,
            seq_len=cli.seq_len,
            batch_size=cli.batch_size,
            device=dev,
        )
        print(f"Predicted labels shape: {tuple(preds.shape)}")
        print(f"Predicted labels: {preds.cpu().tolist()}")

    elif cli.task == "confusion":
        # confusion matrix should run with batch_size > 1 for speed
        bs = cli.batch_size if cli.batch_size > 1 else 16
        plot_confusion_from_checkpoint(
            cli.checkpoint,
            feature_path=cli.feature_path,
            batch_size=bs,
            device=dev,
            file_name=cli.file_name,
            save_dir=cli.save_dir,
            img_format=cli.img_format,
        )