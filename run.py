"""Train CRG-3: best validation when enabled; valid_ratio=0 uses train/test and best test."""
import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler

from CRG3.config import (ModelConfig, GRAPH_UPDATE_MODES, FUSION_MODES,
                         LEARNED_HYPERGRAPH_MODES, model_config_from_dict)
from CRG3.data import (Collator, GraphSmileDataset, NUM_CLASSES, to_device,
                       split_train_validation, session_id)
from CRG3.model import CRG3, build_model


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
    parser.add_argument("--valid_ratio", type=float, default=0.1,
                        help="Fraction of official train dialogues reserved for validation; 0 disables validation")
    parser.add_argument("--validation_split", choices=("ratio", "session"), default="ratio",
                        help="ratio: first valid_ratio of train dialogue IDs; session: held-out IEMOCAP session")
    parser.add_argument("--validation_session", type=int, choices=(1, 2, 3, 4), default=1,
                        help="Held-out IEMOCAP session, used only with --validation_split session")
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
        options = {}
        if name == "graph_update_mode":
            options["choices"] = GRAPH_UPDATE_MODES
        if name == "fusion_mode":
            options["choices"] = FUSION_MODES
        if name == "attention_adjacent_source":
            options["choices"] = ("input", "updated")
        if name == "attention_pool_reduction":
            options["choices"] = ("sum", "mean")
        if name == "joint_local_mixer":
            options["choices"] = ("conv", "gated_silu", "gmu_tanh")
        if name == "joint_gate_normalization":
            options["choices"] = ("sigmoid", "softmax")
        if name in ("graph_gate", "joint_local_kernel"):
            options["choices"] = ("on", "off")
        if name in ("model_type", "message_mode", "graph_rmsnorm"):
            options["choices"] = (field.default,)
        if isinstance(field.default, bool):
            parser.add_argument(f"--{name}", action=argparse.BooleanOptionalAction, default=field.default)
        else:
            parser.add_argument(f"--{name}", type=type(field.default), default=field.default, **options)
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch_size < 1 or args.hidden < 1 or args.graph_layers < 1:
        parser.error("epochs, batch_size, hidden and graph_layers must be positive")
    if args.local_window < 0:
        parser.error("local_window must be nonnegative")
    if args.message_mode not in ("projection", "embedding"):
        parser.error("message_mode must be 'projection' or 'embedding'")
    if args.graph_rmsnorm not in ("on", "off"):
        parser.error("graph_rmsnorm must be 'on' or 'off'")
    try:
        config_from_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.decoder_mode != "classifier":
        parser.error("Use python -m CRG3.run_llm for the generative LLM decoder")
    if not 0 <= args.valid_ratio < 1:
        parser.error("valid_ratio must be in [0, 1)")
    if args.validation_split == "session" and args.dataset not in ("IEMOCAP", "IEMOCAP4"):
        parser.error("session validation is only supported for IEMOCAP")
    if args.patience < 0 or args.min_epochs < 1:
        parser.error("patience must be nonnegative and min_epochs positive")
    if args.lr_patience < 0 or not 0 < args.lr_factor < 1 or args.min_lr <= 0:
        parser.error("lr_patience must be nonnegative, lr_factor in (0,1), min_lr positive")
    return args


def config_from_args(args):
    return ModelConfig(**{name: getattr(args, name) for name in ModelConfig.__dataclass_fields__})


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_train_valid_sampler(trainset, valid=0.1, dataset="IEMOCAP", limit_dialogs=0,
                            validation_session=1, validation_split="ratio"):
    train, val = split_train_validation(trainset, dataset, valid, validation_session, limit_dialogs,
                                        validation_split)
    return SubsetRandomSampler(train), SubsetRandomSampler(val) if val else None


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False,
                        data_path=None, local_window=20, limit_dialogs=0,
                        dataset="IEMOCAP", validation_session=1, validation_split="ratio"):
    trainset = GraphSmileDataset(data_path, dataset, "train")
    testset = GraphSmileDataset(data_path, dataset, "test")
    if trainset.feature_dims != testset.feature_dims:
        raise ValueError("Train/test feature dimensions differ")
    if set(trainset.keys) & set(testset.keys):
        raise ValueError("Official train/test dialogue IDs overlap")
    if dataset in ("IEMOCAP", "IEMOCAP4") and any(session_id(k) != 5 for k in testset.keys):
        raise ValueError("IEMOCAP test must contain only session 5")
    train_indices, val_indices = split_train_validation(
        trainset, dataset, valid, validation_session, limit_dialogs, validation_split)
    kwargs = dict(batch_size=batch_size, collate_fn=Collator(local_window),
                  num_workers=num_workers, pin_memory=pin_memory)
    train_loader = DataLoader(trainset, sampler=SubsetRandomSampler(train_indices), **kwargs)
    # Sequential validation batches make checkpoint verification deterministic.
    valid_loader = DataLoader(Subset(trainset, val_indices), **kwargs) if val_indices else None
    if limit_dialogs:
        testset = Subset(testset, range(min(limit_dialogs, len(testset))))
    return train_loader, valid_loader, DataLoader(testset, **kwargs)


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False,
                     data_path=None, local_window=20, limit_dialogs=0, validation_session=1,
                     validation_split="ratio"):
    return get_IEMOCAP_loaders(batch_size, valid, num_workers, pin_memory,
                              data_path, local_window, limit_dialogs, "MELD", validation_session,
                              validation_split)


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
    shared_norm_sum = 0.0
    modality_sum = torch.zeros(3, dtype=torch.float64)
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    use_fusion = core.cfg.fusion_mode == "hyperedge"
    use_hypergraph = core.cfg.fusion_mode in LEARNED_HYPERGRAPH_MODES
    use_edge_fusion = core.cfg.fusion_mode == "hypergraph_fusion"
    use_confidence = core.cfg.hypergraph_confidence
    main_loss_sum = aux_loss_sum = 0.0
    confidence_sum = torch.zeros(3, dtype=torch.float64)
    confidence_correct = torch.zeros(3, dtype=torch.float64)
    edge_sum = torch.zeros(core.cfg.hypergraph_edges, dtype=torch.float64) if use_hypergraph else None
    beta_sum = torch.zeros(core.cfg.hypergraph_edges, dtype=torch.float64) if use_edge_fusion else None
    hypergraph_norm_sum = 0.0
    labels, preds = [], []

    with torch.set_grad_enabled(train):
        for data in dataloader:
            batch = to_device(data, device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            with autocast():
                output = model(batch, return_aux=True) if (use_fusion or use_hypergraph) else {"logits": model(batch)}
                logits = output["logits"]
                mask = batch["mask"]
                losses = core.classification_losses(output, batch["label"], mask, loss_function)
                loss = losses["loss"]

            labels.extend(batch["label"][mask].detach().cpu().tolist())
            preds.extend(logits[mask].detach().argmax(-1).cpu().tolist())
            count = int(mask.sum())
            class_weights = getattr(loss_function, "weight", None)
            batch_denominator = (float(class_weights[batch["label"][mask]].sum().item())
                                 if class_weights is not None else count)
            loss_sum += float(loss.item()) * batch_denominator
            if use_confidence:
                main_loss_sum += float(losses["classification_loss"].item()) * batch_denominator
                aux_loss_sum += float(losses["confidence_aux_loss"].item()) * batch_denominator
                confidence_sum += output["hypergraph_confidence"][mask].detach().double().sum(0).cpu()
                confidence_correct += (output["hypergraph_confidence_logits"][mask].detach().argmax(-1)
                                       == batch["label"][mask].unsqueeze(-1)).double().sum(0).cpu()
            if use_fusion:
                shared_norm_sum += float(output["shared_update"][mask].detach().float().norm(dim=-1).sum())
                modality_sum += output["modality_weights"][mask].detach().double().sum(0).cpu()
            if use_hypergraph:
                edge_sum += output["hypergraph_membership"][mask].detach().double().sum((0, 1)).cpu()
                value = output["hypergraph_fused"] if use_edge_fusion else output["hypergraph_raw"]
                hypergraph_norm_sum += float(value[mask].detach().float().norm(dim=-1).sum())
                if use_edge_fusion:
                    beta_sum += output["hypergraph_attention"][mask].detach().double().sum(0).cpu()
            loss_denominator += batch_denominator
            n_utterances += count

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

    if n_utterances == 0:
        raise ValueError("Empty dataloader")
    metrics = {"loss": loss_sum / loss_denominator,
            "accuracy": 100 * accuracy_score(labels, preds),
            "weighted_f1": 100 * f1_score(labels, preds, average="weighted", zero_division=0),
            "macro_f1": 100 * f1_score(labels, preds, average="macro", zero_division=0),
            "n_utterances": n_utterances}
    if use_fusion:
        metrics.update(mean_shared_update_norm=shared_norm_sum / n_utterances,
                       mean_modality_weights=dict(zip(("t", "a", "v"), (modality_sum / n_utterances).tolist())))
    if use_hypergraph:
        metrics.update(mean_hyperedge_membership=(edge_sum / (3 * n_utterances)).tolist())
        if use_edge_fusion:
            metrics.update(mean_hyperedge_attention=(beta_sum / n_utterances).tolist(),
                           mean_hypergraph_fused_norm=hypergraph_norm_sum / n_utterances)
        else:
            metrics.update(mean_hypergraph_raw_norm=hypergraph_norm_sum / (3 * n_utterances))
    if use_confidence:
        metrics.update(classification_loss=main_loss_sum / loss_denominator,
                       confidence_aux_loss=aux_loss_sum / loss_denominator,
                       mean_modality_confidence=dict(zip(("t", "a", "v"),
                           (confidence_sum / n_utterances).tolist())),
                       confidence_head_accuracy=dict(zip(("t", "a", "v"),
                           (100 * confidence_correct / n_utterances).tolist())))
    return metrics


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
                         limit_dialogs=args.limit_dialogs, validation_session=args.validation_session,
                         validation_split=args.validation_split)
    if args.dataset == "IEMOCAP":
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(**loader_kwargs)
    elif args.dataset == "MELD":
        train_loader, valid_loader, test_loader = get_MELD_loaders(**loader_kwargs)
    else:
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(
            **loader_kwargs, dataset=args.dataset)
    train_data = train_loader.dataset
    has_validation = valid_loader is not None
    split_sizes = (len(train_loader.sampler),
                   len(valid_loader.dataset) if has_validation else 0,
                   len(test_loader.dataset))
    class_weights, class_counts = get_class_weights(
        train_loader, train_data.num_classes, device, args.class_weighting)
    train_loss_function = (torch.nn.CrossEntropyLoss(weight=class_weights)
                           if class_weights is not None else F.cross_entropy)
    cfg = config_from_args(args)
    model = build_model(cfg, train_data.feature_dims, train_data.num_classes).to(device)
    if args.multi_gpu:
        if device.type != "cuda" or torch.cuda.device_count() < 2:
            raise RuntimeError("--multi_gpu requires at least two visible CUDA GPUs")
        model = torch.nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=args.lr_factor, patience=args.lr_patience,
        min_lr=args.min_lr) if args.lr_patience and has_validation else None)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("best_val.pt", "best_test.pt", "results.json")):
        raise ValueError("Choose a new out_dir; a completed or interrupted run already exists here")
    history, best, stale = [], None, 0
    split_ids = {"train": [str(train_data.keys[i]) for i in train_loader.sampler.indices],
                 "validation": ([str(train_data.keys[i]) for i in valid_loader.dataset.indices]
                                if has_validation else [])}
    test_data = test_loader.dataset
    test_indices = test_data.indices if isinstance(test_data, Subset) else range(len(test_data))
    test_base = test_data.dataset if isinstance(test_data, Subset) else test_data
    split_ids["test"] = [str(test_base.keys[i]) for i in test_indices]
    selection = "best validation weighted F1" if has_validation else "best test weighted F1"
    checkpoint_name = "best_val.pt" if has_validation else "best_test.pt"
    phase = "validation" if has_validation else "test"
    report = {"name": "CRG-3", "selection": selection, "dataset": args.dataset,
              "text_feature": "textf0", "config": vars(args), "model_config": asdict(cfg),
              "feature_dims": train_data.feature_dims, "train_class_counts": class_counts,
              "split_dialogue_ids": split_ids,
              "parameter_count": sum(p.numel() for p in model.parameters()),
              "class_weights": class_weights.tolist() if class_weights is not None else None,
              "validation_enabled": has_validation,
              "lr_scheduler_active": scheduler is not None,
              "early_stopping_active": bool(args.patience and has_validation),
              "epochs": history}
    print(f"CRG3 | {args.dataset} | {cfg.graph_update_mode} | fusion={cfg.fusion_mode} | device={device} | "
          f"train={split_sizes[0]} val={split_sizes[1]} test={split_sizes[2]} | selection={selection}", flush=True)
    def synchronize():
        if device.type == "cuda":
            for index in range(torch.cuda.device_count() if args.multi_gpu else 1):
                torch.cuda.synchronize(index)

    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        synchronize()
        tick = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]["lr"]
        train_metrics = train_or_eval_graph_model(
            model, train_loss_function, train_loader, epoch, device, autocast,
            optimizer=optimizer, train=True, max_grad_norm=args.max_grad_norm)
        synchronize()
        train_seconds = time.perf_counter() - tick
        eval_start = time.perf_counter()
        eval_metrics = train_or_eval_graph_model(
            model, F.cross_entropy, valid_loader if has_validation else test_loader, epoch, device, autocast)
        synchronize()
        evaluation_seconds = time.perf_counter() - eval_start
        if scheduler is not None:
            scheduler.step(eval_metrics["weighted_f1"])
        record = {"epoch": epoch, "train": train_metrics, phase: eval_metrics,
                  "lr": epoch_lr, "next_lr": optimizer.param_groups[0]["lr"],
                  "train_seconds": train_seconds, "evaluation_seconds": evaluation_seconds}
        history.append(record)
        save_start = time.perf_counter()
        if best is None or eval_metrics["weighted_f1"] > best["weighted_f1"]:
            best = {"epoch": epoch, **eval_metrics}
            stale = 0
            state = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
            torch.save({"model": state, "model_config": asdict(cfg),
                        "feature_dims": train_data.feature_dims, "num_classes": train_data.num_classes,
                        "dataset": args.dataset, "epoch": epoch, "selection": selection,
                        phase + "_metrics": eval_metrics, "split_dialogue_ids": split_ids,
                        "training_config": vars(args)}, out / checkpoint_name)
        else:
            stale += 1
        record["checkpoint_seconds"] = time.perf_counter() - save_start
        record["epoch_seconds"] = time.perf_counter() - tick
        report["best_validation"] = best if has_validation else None
        if not has_validation:
            report["best_test"] = best
        (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"epoch {epoch:03d} train_loss={train_metrics['loss']:.4f} "
              f"train_acc={train_metrics['accuracy']:.2f} train_wF1={train_metrics['weighted_f1']:.2f} "
              f"train_macroF1={train_metrics['macro_f1']:.2f} "
              f"{phase}_acc={eval_metrics['accuracy']:.2f} {phase}_wF1={eval_metrics['weighted_f1']:.2f} "
              f"lr={epoch_lr:.2e} best_epoch={best['epoch']} "
              f"train_seconds={train_seconds:.2f} seconds={record['epoch_seconds']:.2f}", flush=True)
        if has_validation and args.patience and epoch >= args.min_epochs and stale >= args.patience:
            break
    report["training_evaluation_seconds"] = time.perf_counter() - started
    # With validation, test is evaluated only after restoring the selected checkpoint.
    saved = torch.load(out / checkpoint_name, map_location=device, weights_only=False)
    core = model.module if isinstance(model, torch.nn.DataParallel) else model
    core.load_state_dict(saved["model"], strict=True)
    test_metrics = train_or_eval_graph_model(model, F.cross_entropy, test_loader, best["epoch"], device, autocast)
    report["test_at_best_validation" if has_validation else "test_at_best_test"] = {
        "epoch": best["epoch"], **test_metrics}
    report["test_evaluations_during_training"] = 0 if has_validation else len(history)
    report["test_evaluations_after_selection"] = 1
    (out / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selection={selection} | epoch={best['epoch']} | selected_wF1={best['weighted_f1']:.2f} | "
          f"test_wF1={test_metrics['weighted_f1']:.2f}", flush=True)


if __name__ == "__main__":
    main()
