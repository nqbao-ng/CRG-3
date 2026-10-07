"""Train CRG-3 on GraphSmile pickles; select the best epoch by test weighted F1."""
import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from crg3.config import ModelConfig
from crg3.data import Collator, GraphSmileDataset, NUM_CLASSES, to_device
from crg3.model import CRG3


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="CRG-3 on GraphSmile textf0 features")
    parser.add_argument("--data_path", required=True, help="GraphSmile dataset pickle")
    parser.add_argument("--dataset", choices=NUM_CLASSES, default="IEMOCAP")
    parser.add_argument("--out_dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--class_weighting", choices=["none", "inverse_sqrt"], default="none",
                        help="Training CE class weights computed only from the selected train dialogues")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--limit_dialogs", type=int, default=0, help="First N dialogues per split for a smoke run")
    parser.add_argument("--valid_ratio", type=float, default=0.0,
                        help="Fraction of official train dialogues for validation; 0 uses all trainVid without validation")
    parser.add_argument("--patience", type=int, default=0,
                        help="Stop after this many epochs without validation weighted-F1 improvement; 0 disables")
    parser.add_argument("--min_epochs", type=int, default=1)
    parser.add_argument("--lr_patience", type=int, default=0,
                        help="Reduce LR after this many stagnant validation epochs; 0 disables")
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--amp", action="store_true", help="Use bfloat16 autocast on supported CUDA GPUs")
    parser.add_argument("--multi_gpu", action="store_true", help="Use all visible CUDA GPUs with DataParallel")
    for name, field in ModelConfig.__dataclass_fields__.items():
        parser.add_argument(f"--{name}", type=type(field.default), default=field.default)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 1 or args.graph_layers < 1:
        parser.error("epochs, batch_size, hidden and graph_layers must be positive")
    if args.local_window < 0:
        parser.error("local_window must be nonnegative")
    if args.post_graph_fusion not in ("mamba", "sum"):
        parser.error("post_graph_fusion must be 'mamba' or 'sum'")
    if args.message_mode not in ("projection", "embedding"):
        parser.error("message_mode must be 'projection' or 'embedding'")
    if args.edge_weight_mode not in ("none", "dot_tanh", "dot_softmax", "vuemo_tanh"):
        parser.error("edge_weight_mode must be none, dot_tanh, dot_softmax or vuemo_tanh")
    if args.edge_weight_dim < 1:
        parser.error("edge_weight_dim must be positive")
    if not 0 <= args.valid_ratio < 1:
        parser.error("valid_ratio must be in [0, 1)")
    if args.patience < 0 or args.min_epochs < 1:
        parser.error("patience must be nonnegative and min_epochs positive")
    if args.lr_patience < 0 or not 0 < args.lr_factor < 1 or args.min_lr <= 0:
        parser.error("lr_patience must be nonnegative, lr_factor in (0,1), min_lr positive")
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_valid_sampler(trainset, valid=0.0, dataset="IEMOCAP", limit_dialogs=0):
    """Split official train IDs by dialogue index."""
    idx = list(range(len(trainset)))
    split = int(valid * len(trainset))
    train_indices, valid_indices = idx[split:], idx[:split]
    if limit_dialogs > 0:
        train_indices = train_indices[:limit_dialogs]
        valid_indices = valid_indices[:limit_dialogs]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices) if valid_indices else None
    return train_sampler, valid_sampler


def get_IEMOCAP_loaders(batch_size=32, valid=0.0, num_workers=0, pin_memory=False,
                        data_path=None, local_window=20, limit_dialogs=0,
                        dataset="IEMOCAP"):
    """Create train/validation/test loaders for CRG-3 from official dialogue IDs."""
    trainset = GraphSmileDataset(data_path, dataset, "train")
    train_sampler, valid_sampler = get_train_valid_sampler(
        trainset, valid, dataset, limit_dialogs)
    collate = Collator(local_window)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=collate,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = None
    if valid_sampler is not None:
        valid_loader = DataLoader(trainset,
                                  batch_size=batch_size,
                                  sampler=valid_sampler,
                                  collate_fn=collate,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    testset = GraphSmileDataset(data_path, dataset, "test")
    if trainset.feature_dims != testset.feature_dims:
        raise ValueError("Train/test feature dimensions differ")
    if limit_dialogs > 0:
        testset = Subset(testset, range(min(limit_dialogs, len(testset))))
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=collate,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def get_MELD_loaders(batch_size=32, valid=0.0, num_workers=0, pin_memory=False,
                     data_path=None, local_window=20, limit_dialogs=0):
    return get_IEMOCAP_loaders(batch_size, valid, num_workers, pin_memory,
                               data_path, local_window, limit_dialogs, dataset="MELD")


def get_class_weights(train_loader, num_classes, device, weighting="none"):
    """Count labels from the train sampler only; never inspect validation or test."""
    trainset = train_loader.dataset
    labels = np.concatenate([trainset.records[i]["label"] for i in train_loader.sampler.indices])
    counts = np.bincount(labels, minlength=num_classes)
    if weighting == "none":
        return None, counts.tolist()
    if weighting != "inverse_sqrt":
        raise ValueError(f"Unknown class weighting: {weighting}")
    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float64)
    weights[present] = 1.0 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return torch.tensor(weights, dtype=torch.float32, device=device), counts.tolist()


def train_or_eval_graph_model(model, loss_function, dataloader, epoch, device,
                              autocast, optimizer=None, train=False, max_grad_norm=1.0):
    """One CRG-3 pass; report utterance-level emotion metrics."""
    if train and optimizer is None:
        raise ValueError("Training requires an optimizer")
    if train:
        model.train()
    else:
        model.eval()
    loss_sum, loss_denominator, n_utterances = 0.0, 0.0, 0
    labels, preds = [], []

    with torch.set_grad_enabled(train):
        for data in dataloader:
            batch = to_device(data, device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast():
                logits = model(batch)
                mask = batch["mask"]
                loss = loss_function(logits[mask].float(), batch["label"][mask])

            labels.extend(batch["label"][mask].detach().cpu().tolist())
            preds.extend(logits[mask].detach().argmax(-1).cpu().tolist())
            count = int(mask.sum())
            class_weights = getattr(loss_function, "weight", None)
            batch_denominator = (float(class_weights[batch["label"][mask]].sum().item())
                                 if class_weights is not None else count)
            loss_sum += float(loss.item()) * batch_denominator
            loss_denominator += batch_denominator
            n_utterances += count

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

    if n_utterances == 0:
        raise ValueError("Empty dataloader")
    return {"loss": loss_sum / loss_denominator,
            "accuracy": 100 * accuracy_score(labels, preds),
            "weighted_f1": 100 * f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_f1": 100 * f1_score(labels, preds, average="macro", zero_division=0),
            "n_utterances": n_utterances}


def main(argv=None):
    args = parse_args(argv)
    seed_everything(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if (args.device == "cuda" or
                                    args.device == "auto" and torch.cuda.is_available()) else "cpu")
    use_amp = args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.amp and not use_amp:
        print("bfloat16 AMP unavailable; using float32", flush=True)

    def autocast():
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp)

    loader_kwargs = dict(batch_size=args.batch_size, valid=args.valid_ratio,
                         num_workers=args.num_workers, pin_memory=False,
                         data_path=args.data_path, local_window=args.local_window,
                         limit_dialogs=args.limit_dialogs)
    if args.dataset == "IEMOCAP":
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(**loader_kwargs)
    elif args.dataset == "MELD":
        train_loader, valid_loader, test_loader = get_MELD_loaders(**loader_kwargs)
    else:
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
            **loader_kwargs, dataset=args.dataset)
    train_data = train_loader.dataset
    split_sizes = (len(train_loader.sampler),
                   len(valid_loader.sampler) if valid_loader is not None else 0,
                   len(test_loader.dataset))
    class_weights, class_counts = get_class_weights(
        train_loader, train_data.num_classes, device, args.class_weighting)
    train_loss_function = (torch.nn.CrossEntropyLoss(weight=class_weights)
                           if class_weights is not None else F.cross_entropy)
    cfg = ModelConfig(**{name: getattr(args, name) for name in ModelConfig.__dataclass_fields__})
    model = CRG3(cfg, train_data.feature_dims, train_data.num_classes).to(device)
    if args.multi_gpu:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--multi_gpu requires at least two visible CUDA GPUs")
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr) if args.lr_patience else None)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    best = None
    best_val = None
    stale = 0
    print(f"CRG-3 | {args.dataset} | textf0 | device={device} | "
          f"train={split_sizes[0]} val={split_sizes[1]} "
          f"test={split_sizes[2]} | dims={train_data.feature_dims}", flush=True)
    print(f"Train class counts={class_counts} | CE weights="
          f"{class_weights.tolist() if class_weights is not None else 'none'}", flush=True)
    for epoch in range(1, args.epochs + 1):
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_or_eval_graph_model(
            model, train_loss_function, train_loader, epoch, device, autocast,
            optimizer=optimizer, train=True, max_grad_norm=args.max_grad_norm)
        val_metrics = (train_or_eval_graph_model(
            model, F.cross_entropy, valid_loader, epoch, device, autocast)
            if valid_loader is not None else None)
        if scheduler is not None and val_metrics is not None:
            scheduler.step(val_metrics["weighted_f1"])
        metrics = train_or_eval_graph_model(
            model, F.cross_entropy, test_loader, epoch, device, autocast)
        record = {"epoch": epoch, "train_loss": train_metrics["loss"],
                  "train_accuracy": train_metrics["accuracy"],
                  "train_weighted_f1": train_metrics["weighted_f1"],
                  "train_macro_f1": train_metrics["macro_f1"],
                  "lr": epoch_lr, "test_loss": metrics["loss"],
                  **{key: value for key, value in metrics.items() if key != "loss"}}
        if val_metrics is not None:
            record["validation"] = val_metrics
            if best_val is None or val_metrics["weighted_f1"] > best_val["weighted_f1"] + 1e-4:
                best_val = {"epoch": epoch, **val_metrics}
                stale = 0
            else:
                stale += 1
        history.append(record)
        if best is None or record["weighted_f1"] > best["weighted_f1"]:
            best = dict(record)
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({"model": state, "model_config": asdict(cfg),
                        "feature_dims": train_data.feature_dims,
                        "num_classes": train_data.num_classes, "dataset": args.dataset,
                        "class_weighting": args.class_weighting,
                        "class_weights": class_weights.tolist() if class_weights is not None else None,
                        "epoch": epoch, "test_metrics": metrics}, out / "best_test.pt")
        with (out / "results.json").open("w", encoding="utf-8") as handle:
            json.dump({"name": "CRG-3", "selection": "best test weighted F1",
                       "dataset": args.dataset, "text_feature": "textf0",
                       "config": vars(args), "feature_dims": train_data.feature_dims,
                       "train_class_counts": class_counts,
                       "class_weights": class_weights.tolist() if class_weights is not None else None,
                       "best_test": best, "best_validation": best_val,
                       "epochs": history}, handle, indent=2)
        message = (f"epoch {epoch:03d} train_loss={record['train_loss']:.4f} "
                   f"train_acc={record['train_accuracy']:.2f} train_wF1={record['train_weighted_f1']:.2f} "
                   f"train_macroF1={record['train_macro_f1']:.2f} lr={record['lr']:.2e} ")
        if val_metrics is not None:
            message += f"val_wF1={val_metrics['weighted_f1']:.2f} "
        message += (f"test_acc={record['accuracy']:.2f} "
                    f"test_wF1={record['weighted_f1']:.2f} best_epoch={best['epoch']}")
        print(message, flush=True)
        if val_metrics is not None and args.patience and epoch >= args.min_epochs and stale >= args.patience:
            print(f"Early stop: validation weighted F1 did not improve for {stale} epochs", flush=True)
            break
    print(f"Best test weighted F1: {best['weighted_f1']:.2f} at epoch {best['epoch']}", flush=True)


if __name__ == "__main__":
    main()
