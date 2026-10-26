"""Prepare frozen CRG3 feature caches, then train/evaluate its generative decoder.

Run from CRG-3: python -m CRG3.run_llm {prepare,train,evaluate} --help.
No large model is loaded during prepare. Training uses one GPU and selects by
validation weighted F1; the official test split is evaluated after selection.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import LLMConfig, ModelConfig, LEARNED_HYPERGRAPH_MODES, model_config_from_dict
from .data import (Collator, GraphSmileDataset, NUM_CLASSES, to_device,
                   split_train_validation, session_id)
from .model import CRG3
from .model import build_model


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_names_from_json(path, count):
    names = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(names, list) or len(names) != count or
            any(not isinstance(x, str) or not x.strip() for x in names) or len(set(names)) != count):
        raise ValueError("Label JSON must be a unique string list ordered by the dataset's integer label IDs")
    return names


def file_digest(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cached_collate(items):
    return {"graph_features": {m: torch.stack([r["features"][m] for r in items])[:, None]
                               for m in ("t", "a", "v")},
            "mask": torch.ones(len(items), 1, dtype=torch.bool),
            "label": torch.tensor([r["label"] for r in items])[:, None],
            "speaker": torch.tensor([r["speaker"] for r in items])[:, None],
            "transcript": [[r["transcript"]] for r in items],
            "ids": [r["dialogue_id"] for r in items],
            "utterance_indices": [r["utterance_index"] for r in items]}


def prepare_cache(args):
    target = Path(args.cache_path)
    if target.exists():
        raise ValueError("Cache already exists; use a new cache_path to preserve the existing run")
    seed_all(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    checkpoint = torch.load(args.graph_checkpoint, map_location="cpu", weights_only=False)
    raw_cfg = dict(checkpoint["model_config"])
    # A historical default convolution config can still describe the same baseline.
    if raw_cfg.get("graph_conv_mode", "baseline") != "baseline":
        raise ValueError("Use a supported graph checkpoint, not a removed convolution variant")
    for field in ("graph_conv_mode", "kernel_experts", "kernel_router_dim"):
        raw_cfg.pop(field, None)
    cfg = model_config_from_dict(raw_cfg)
    if cfg.decoder_mode != "classifier":
        raise ValueError("prepare expects an existing classifier graph checkpoint")
    if cfg.fusion_mode in LEARNED_HYPERGRAPH_MODES:
        raise ValueError("Learned hypergraph readouts currently support classifier experiments, not this LLM cache")
    if checkpoint.get("dataset", args.dataset) != args.dataset:
        raise ValueError("Graph checkpoint and requested dataset differ")
    trainset = GraphSmileDataset(args.data_path, args.dataset, "train", include_transcript=True,
                                transcripts_path=args.transcripts_path)
    testset = GraphSmileDataset(args.data_path, args.dataset, "test", include_transcript=True,
                               transcripts_path=args.transcripts_path)
    if checkpoint["feature_dims"] != trainset.feature_dims or trainset.feature_dims != testset.feature_dims:
        raise ValueError("Graph checkpoint/data feature dimensions differ")
    if checkpoint["num_classes"] != trainset.num_classes:
        raise ValueError("Graph checkpoint/data label counts differ")
    names = label_names_from_json(args.label_names, trainset.num_classes)
    train_indices, val_indices = split_train_validation(
        trainset, args.dataset, args.valid_ratio, args.validation_session, args.limit_dialogs,
        validation_split="session" if args.dataset in ("IEMOCAP", "IEMOCAP4") else "ratio")
    train_records = [trainset.records[i] for i in train_indices]
    val_records = [trainset.records[i] for i in val_indices]
    if args.dataset in ("IEMOCAP", "IEMOCAP4") and any(session_id(k) != 5 for k in testset.keys):
        raise ValueError("IEMOCAP test must contain only session 5")
    test_records = testset.records[:args.limit_dialogs] if args.limit_dialogs else testset.records
    split_ids = {name: {str(r["id"]) for r in values} for name, values in
                 (("train", train_records), ("validation", val_records), ("test", test_records))}
    if any(split_ids[a] & split_ids[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Dialogue IDs overlap across train/validation/test")
    model = build_model(cfg, trainset.feature_dims, trainset.num_classes).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    model.requires_grad_(False)
    cache = {"schema_version": 1, "metadata": {
        "dataset": args.dataset, "model_config": asdict(cfg), "feature_dims": trainset.feature_dims,
        "num_classes": trainset.num_classes, "label_names": names, "split_seed": args.seed,
        "graph_checkpoint": str(args.graph_checkpoint), "graph_sha256": file_digest(args.graph_checkpoint),
        "graph_checkpoint_selection": checkpoint.get("selection", "not_recorded"),
        "graph_training_split_verified": False,
        "split_dialogue_ids": {k: sorted(v) for k, v in split_ids.items()}},
        "encoder_state": {k: v.cpu() for k, v in model.encoder_state_dict().items()}}
    with torch.no_grad():
        for name, records in (("train", train_records), ("validation", val_records), ("test", test_records)):
            rows = []
            loader = DataLoader(records, batch_size=args.batch_size, shuffle=False,
                                collate_fn=Collator(cfg.local_window))
            for batch in loader:
                q, _ = model.modality_features(to_device(batch, device))
                for b, i in batch["mask"].nonzero().tolist():
                    rows.append({"features": {m: q[m][b, i].float().cpu().clone() for m in q},
                                 "label": int(batch["label"][b, i]), "speaker": int(batch["speaker"][b, i]),
                                 "transcript": batch["transcript"][b][i],
                                 "dialogue_id": str(batch["ids"][b]), "utterance_index": i})
            cache[name] = rows
            print(f"Cached {name}: {len(records)} dialogues, {len(rows)} utterances", flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, target)
    print(f"Cache saved: {target}", flush=True)


def load_cache(path):
    cache = torch.load(path, map_location="cpu", weights_only=False)
    if cache.get("schema_version") != 1:
        raise ValueError("Unsupported graph feature cache")
    d = cache["metadata"]["model_config"]["hidden"]
    classes = cache["metadata"]["num_classes"]
    for split in ("train", "validation", "test"):
        if not cache.get(split):
            raise ValueError(f"Empty cache split: {split}")
        for row in cache[split]:
            if not 0 <= row["label"] < classes or not row["transcript"].strip():
                raise ValueError("Invalid cached label/transcript")
            for m in ("t", "a", "v"):
                value = row["features"][m]
                if value.shape != (d,) or not torch.isfinite(value).all():
                    raise ValueError("Invalid cached modality feature")
    return cache


def make_model(cache, llm_config):
    cfg = model_config_from_dict({**cache["metadata"]["model_config"], "decoder_mode": "llm"})
    model = build_model(cfg, cache["metadata"]["feature_dims"], cache["metadata"]["num_classes"],
                 llm_config=llm_config, label_names=cache["metadata"]["label_names"])
    model.load_encoder_state_dict(cache["encoder_state"])
    return model


@torch.no_grad()
def evaluate_model(model, rows, batch_size):
    from sklearn.metrics import accuracy_score, f1_score
    model.eval()
    gold, predicted, records = [], [], []
    for batch in DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=cached_collate):
        result = model.predict(batch)
        predictions = result["label_ids"][:, 0].cpu().tolist()
        labels = batch["label"][:, 0].tolist()
        gold.extend(labels)
        predicted.extend(predictions)
        for i, pred in enumerate(predictions):
            records.append({"dialogue_id": batch["ids"][i], "utterance_index": batch["utterance_indices"][i],
                            "gold_id": labels[i], "predicted_id": pred,
                            "generated_label": result["labels"][i], "generated_token_ids": result["token_ids"][i]})
    classes = list(range(len(model.llm_decoder.label_names)))
    metrics = {"accuracy": 100 * accuracy_score(gold, predicted),
               "weighted_f1": 100 * f1_score(gold, predicted, labels=classes, average="weighted", zero_division=0),
               "macro_f1": 100 * f1_score(gold, predicted, labels=classes, average="macro", zero_division=0),
               "per_class_f1": (100 * f1_score(gold, predicted, labels=classes, average=None, zero_division=0)).tolist(),
               "n_utterances": len(gold), "invalid_outputs": sum(p not in classes for p in predicted)}
    return metrics, records


def save_bundle(model, cache, target):
    target = Path(target)
    model.llm_decoder.save_pretrained(target)
    torch.save(cache["encoder_state"], target / "encoder.pt")
    (target / "model_config.json").write_text(json.dumps(
        {"model_config": asdict(model.cfg), "feature_dims": cache["metadata"]["feature_dims"],
         "num_classes": cache["metadata"]["num_classes"], "dataset": cache["metadata"]["dataset"],
         "graph_sha256": cache["metadata"]["graph_sha256"]}, indent=2), encoding="utf-8")


def load_bundle_weights(model, cache, target):
    target = Path(target)
    meta = json.loads((target / "model_config.json").read_text(encoding="utf-8"))
    if meta["graph_sha256"] != cache["metadata"]["graph_sha256"] or meta["dataset"] != cache["metadata"]["dataset"]:
        raise ValueError("Decoder checkpoint and feature cache disagree")
    model.load_encoder_state_dict(torch.load(target / "encoder.pt", map_location="cpu", weights_only=True))
    model.llm_decoder.load_decoder_weights(target)


def train_decoder(args):
    out = Path(args.out_dir)
    if (out / "results.json").exists() and not args.resume:
        raise ValueError("Output already contains results; choose a new directory or --resume")
    seed_all(args.seed)
    cache = load_cache(args.cache_path)
    cfg = LLMConfig(model_name=args.llm_model, load_in_4bit=not args.no_4bit,
                    device=args.device, compute_dtype=args.compute_dtype, lora_rank=args.lora_rank,
                    lora_alpha=args.lora_alpha, max_prompt_tokens=args.max_prompt_tokens,
                    gradient_checkpointing=not args.no_gradient_checkpointing, freeze_graph=True)
    model = make_model(cache, cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    device = model.llm_decoder.language_model.get_input_embeddings().weight.device
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg.compute_dtype == "float16")
    out.mkdir(parents=True, exist_ok=True)
    history, best, start_epoch = [], None, 1
    if args.resume:
        load_bundle_weights(model, cache, out / "last")
        state = torch.load(out / "last/training_state.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        history, best, start_epoch = state["history"], state["best"], state["epoch"] + 1
        if state["gradient_accumulation"] != args.gradient_accumulation or state["batch_size"] != args.batch_size:
            raise ValueError("Resume must retain batch_size and gradient_accumulation")
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        if device.type == "cuda" and state["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    print(f"Training projector/LoRA parameters: {sum(p.numel() for p in trainable):,}; device={device}", flush=True)
    loader = DataLoader(cache["train"], batch_size=args.batch_size, shuffle=True, collate_fn=cached_collate)
    if not loader:
        raise ValueError("Empty training loader")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        token_loss, n_tokens = 0.0, 0
        # Token-weighted accumulation, including a short final accumulation group.
        group_tokens = 0
        for step, batch in enumerate(loader):
            count = sum(len(model.llm_decoder.label_tokens[int(label)]) + 1 for label in batch["label"].flatten())
            with torch.autocast(device_type=device.type, dtype=getattr(torch, cfg.compute_dtype),
                                enabled=device.type == "cuda" and cfg.compute_dtype != "float32"):
                loss = model.loss(batch)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite generation loss at epoch {epoch}, batch {step}")
            scaler.scale(loss * count).backward()
            group_tokens += count
            token_loss += float(loss.detach()) * count
            n_tokens += count
            if (step + 1) % args.gradient_accumulation == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                for p in trainable:
                    if p.grad is not None:
                        p.grad.div_(group_tokens)
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                group_tokens = 0
        validation, _ = evaluate_model(model, cache["validation"], args.batch_size)
        record = {"epoch": epoch, "train_token_loss": token_loss / n_tokens,
                  "validation": validation, "epoch_seconds": time.monotonic() - started}
        history.append(record)
        if best is None or validation["weighted_f1"] > best["validation"]["weighted_f1"]:
            best = record
            save_bundle(model, cache, out / "best")
        save_bundle(model, cache, out / "last")
        torch.save({"epoch": epoch, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                    "gradient_accumulation": args.gradient_accumulation, "batch_size": args.batch_size,
                    "history": history, "best": best, "torch_rng": torch.get_rng_state(),
                    "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None},
                   out / "last/training_state.pt")
        (out / "results.json").write_text(json.dumps({"selection": "best validation weighted F1",
            "arguments": vars(args), "cache_metadata": cache["metadata"], "llm_config": asdict(cfg),
            "best_validation": best, "epochs": history}, indent=2), encoding="utf-8")
        print(f"epoch={epoch} loss={record['train_token_loss']:.4f} val_wF1={validation['weighted_f1']:.2f}", flush=True)
    if best is None:
        raise ValueError("No completed epoch/checkpoint")
    load_bundle_weights(model, cache, out / "best")
    metrics, predictions = evaluate_model(model, cache["test"], args.batch_size)
    (out / "test_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")
    results = json.loads((out / "results.json").read_text(encoding="utf-8"))
    results["test_at_selected_checkpoint"] = metrics
    (out / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Test at selected validation checkpoint:", metrics, flush=True)


def evaluate_checkpoint(args):
    cache = load_cache(args.cache_path)
    saved = json.loads((Path(args.checkpoint) / "decoder_config.json").read_text(encoding="utf-8"))
    cfg = LLMConfig(**{**saved["llm_config"], "device": args.device})
    model = make_model(cache, cfg)
    load_bundle_weights(model, cache, args.checkpoint)
    metrics, predictions = evaluate_model(model, cache[args.split], args.batch_size)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"split": args.split, "metrics": metrics, "predictions": predictions}, indent=2), encoding="utf-8")
    print(metrics, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare", help="Cache separate graph features without loading an LLM")
    prepare.add_argument("--data_path", required=True)
    prepare.add_argument("--graph_checkpoint", required=True)
    prepare.add_argument("--dataset", choices=NUM_CLASSES, default="IEMOCAP")
    prepare.add_argument("--label_names", required=True, help="JSON list in integer class-ID order")
    prepare.add_argument("--transcripts_path", default=None, help="Optional JSON override for embedded transcripts")
    prepare.add_argument("--valid_ratio", type=float, default=0.1)
    prepare.add_argument("--validation_session", type=int, choices=(1, 2, 3, 4), default=1)
    prepare.add_argument("--limit_dialogs", type=int, default=0)
    prepare.add_argument("--seed", type=int, default=1475)
    train = sub.add_parser("train", help="Train three projectors and optionally LoRA on cached features")
    train.add_argument("--out_dir", required=True)
    train.add_argument("--llm_model", default=LLMConfig.model_name)
    train.add_argument("--no_4bit", action="store_true")
    train.add_argument("--compute_dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    train.add_argument("--lora_rank", type=int, default=8)
    train.add_argument("--lora_alpha", type=int, default=16)
    train.add_argument("--max_prompt_tokens", type=int, default=512)
    train.add_argument("--no_gradient_checkpointing", action="store_true")
    train.add_argument("--epochs", type=int, default=3)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--gradient_accumulation", type=int, default=16)
    train.add_argument("--max_grad_norm", type=float, default=1.0)
    train.add_argument("--seed", type=int, default=1475)
    train.add_argument("--resume", action="store_true")
    evaluate = sub.add_parser("evaluate", help="Reload a decoder bundle and generate labels")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output", default="llm_predictions.json")
    for command in (prepare, train, evaluate):
        command.add_argument("--cache_path", required=True)
        command.add_argument("--batch_size", type=int, default=1)
        command.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("batch_size must be positive")
    if args.command == "train" and (args.epochs < 1 or args.gradient_accumulation < 1 or args.lr <= 0 or args.max_grad_norm <= 0):
        parser.error("Invalid training settings")
    if args.command == "prepare" and args.limit_dialogs < 0:
        parser.error("limit_dialogs must be nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    {"prepare": prepare_cache, "train": train_decoder, "evaluate": evaluate_checkpoint}[args.command](args)


if __name__ == "__main__":
    main()
