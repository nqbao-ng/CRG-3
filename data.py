"""GraphSmile pickle adapter and shared causal graph support for CRG-3.

Only the precomputed textf0, audio, visual, speakers, emotion labels and
official train/test dialogue IDs are read. GraphSmile model modules are unused.
"""
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_CLASSES = {"IEMOCAP": 6, "IEMOCAP4": 4, "MELD": 7, "CMUMOSEI7": 7}


def build_neighbors(speakers, local_window):
    """Split only the preceding local_window utterances into Diff and Same."""
    if local_window < 0:
        raise ValueError("local_window must be nonnegative")
    diff, same = [], []
    for i, speaker in enumerate(speakers):
        candidates = range(max(0, i - local_window), i)
        diff.append([j for j in candidates if speakers[j] != speaker])
        same.append([j for j in candidates if speakers[j] == speaker])
    return diff, same


def pack_neighbors(neighborhoods, max_length):
    batch_size = len(neighborhoods)
    width = max([1] + [len(src) for conv in neighborhoods for src in conv])
    idx = torch.full((batch_size, max_length, width), -1, dtype=torch.long)
    for b, conv in enumerate(neighborhoods):
        for i, src in enumerate(conv):
            if src:
                idx[b, i, :len(src)] = torch.tensor(src)
    return {"idx": idx, "mask": idx.ge(0)}


def _speaker_ids(values):
    ids, lookup = [], {}
    for value in values:
        arr = np.asarray(value)
        if arr.ndim == 0:
            key = arr.item()
        else:
            key = tuple(arr.reshape(-1).tolist())
        if key not in lookup:
            lookup[key] = len(lookup)
        ids.append(lookup[key])
    return ids


def _mosei_class(values):
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.select([x < -2, x < -1, x < 0, x == 0, x <= 1, x <= 2],
                     [0, 1, 2, 3, 4, 5], default=6).astype(np.int64)


class GraphSmileDataset(Dataset):
    """Official GraphSmile dialogue split, with textf0 as the only text stream."""

    def __init__(self, path, dataset="IEMOCAP", split="train"):
        if dataset not in NUM_CLASSES:
            raise ValueError(f"Unsupported dataset: {dataset}")
        if split not in ("train", "test"):
            raise ValueError("GraphSmile pickle adapter supports train/test")
        with open(path, "rb") as handle:
            raw = pickle.load(handle, encoding="latin1")
        if dataset == "MELD":
            if len(raw) < 13:
                raise ValueError("Expected MELD GraphSmile pickle layout")
            ids, speakers, labels, text, audio, visual = raw[0], raw[1], raw[2], raw[4], raw[8], raw[9]
            train_ids, test_ids = raw[11], raw[12]
        elif dataset == "IEMOCAP":
            if len(raw) < 12:
                raise ValueError("Expected IEMOCAP GraphSmile pickle layout")
            ids, speakers, labels, text, audio, visual = raw[0], raw[1], raw[2], raw[3], raw[7], raw[8]
            train_ids, test_ids = raw[10], raw[11]
        else:  # IEMOCAP4 / CMUMOSEI7: one available text feature
            if len(raw) < 9:
                raise ValueError("Expected GraphSmile single-text pickle layout")
            ids, speakers, labels, text, audio, visual = raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]
            train_ids, test_ids = raw[7], raw[8]
        self.keys = list(train_ids if split == "train" else test_ids)
        if not self.keys:
            raise ValueError(f"Empty {split} split")
        self.records = []
        for key in self.keys:
            t = np.asarray(text[key], dtype=np.float32)
            a = np.asarray(audio[key], dtype=np.float32)
            v = np.asarray(visual[key], dtype=np.float32)
            y = (_mosei_class(labels[key]) if dataset == "CMUMOSEI7"
                 else np.asarray(labels[key], dtype=np.int64).reshape(-1))
            n = len(y)
            if n == 0 or any(x.ndim != 2 or x.shape[0] != n for x in (t, a, v)):
                raise ValueError(f"Feature/label length or rank mismatch for dialogue {key}")
            if len(speakers[key]) != n or y.min() < 0 or y.max() >= NUM_CLASSES[dataset]:
                raise ValueError(f"Invalid speaker/label data for dialogue {key}")
            if not all(np.isfinite(x).all() for x in (t, a, v)):
                raise ValueError(f"Non-finite feature in dialogue {key}")
            self.records.append({"id": key, "text": t, "audio": a, "visual": v,
                                 "speaker": _speaker_ids(speakers[key]), "label": y})
        first = self.records[0]
        self.feature_dims = {m: first[m].shape[1] for m in ("text", "audio", "visual")}
        for record in self.records:
            if any(record[m].shape[1] != self.feature_dims[m] for m in self.feature_dims):
                raise ValueError("Inconsistent feature dimensions across dialogues")
        self.num_classes = NUM_CLASSES[dataset]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class Collator:
    def __init__(self, local_window=20):
        self.local_window = local_window

    def __call__(self, items):
        batch_size = len(items)
        max_length = max(len(item["label"]) for item in items)
        mask = torch.zeros(batch_size, max_length, dtype=torch.bool)
        labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        features = {m: torch.zeros(batch_size, max_length, items[0][m].shape[1])
                    for m in ("text", "audio", "visual")}
        diff, same = [], []
        for b, item in enumerate(items):
            n = len(item["label"])
            mask[b, :n] = True
            labels[b, :n] = torch.as_tensor(item["label"], dtype=torch.long)
            for m in features:
                features[m][b, :n] = torch.as_tensor(item[m], dtype=torch.float32)
            d, s = build_neighbors(item["speaker"], self.local_window)
            diff.append(d)
            same.append(s)
        return {**features, "mask": mask, "label": labels,
                "diff": pack_neighbors(diff, max_length),
                "same": pack_neighbors(same, max_length),
                "ids": [item["id"] for item in items]}


def to_device(batch, device):
    return {k: ({name: tensor.to(device) for name, tensor in value.items()}
                if isinstance(value, dict) else value.to(device) if torch.is_tensor(value) else value)
            for k, value in batch.items()}
