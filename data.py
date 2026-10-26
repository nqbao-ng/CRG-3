"""GraphSmile pickle adapter and shared causal graph support for CRG-3.

Only the precomputed textf0, audio, visual, speakers, emotion labels and
official train/test dialogue IDs are read. GraphSmile model modules are unused.
"""
import pickle
import json
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


NUM_CLASSES = {"IEMOCAP": 6, "IEMOCAP4": 4, "MELD": 7, "CMUMOSEI7": 7}


def session_id(dialogue_id):
    match = re.match(r"^Ses0([1-5])[FM]_", str(dialogue_id))
    if match is None:
        raise ValueError(f"Unrecognized IEMOCAP session ID: {dialogue_id}")
    return int(match.group(1))


def split_train_validation(dataset, dataset_name, valid_ratio=0.1, validation_session=1,
                           limit_dialogs=0, validation_split="ratio"):
    """Split official train dialogues before any truncation or model training."""
    if validation_split not in ("ratio", "session"):
        raise ValueError("validation_split must be 'ratio' or 'session'")
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must be in [0, 1)")
    if dataset_name in ("IEMOCAP", "IEMOCAP4"):
        sessions = [session_id(key) for key in dataset.keys]
        if any(s == 5 for s in sessions):
            raise ValueError("Session 5 must be held out for test")
    elif validation_split == "session":
        raise ValueError("Session validation is only supported for IEMOCAP")
    if valid_ratio == 0:
        train, valid = list(range(len(dataset))), []
    elif validation_split == "session":
        train = [i for i, s in enumerate(sessions) if s != validation_session]
        valid = [i for i, s in enumerate(sessions) if s == validation_session]
    else:
        n_valid = max(1, int(valid_ratio * len(dataset)))
        valid = list(range(n_valid))
        train = list(range(n_valid, len(dataset)))
    if limit_dialogs:
        train, valid = train[:limit_dialogs], valid[:limit_dialogs]
    if not train or (valid_ratio > 0 and not valid):
        raise ValueError("Training must be nonempty; positive valid_ratio also requires validation")
    if set(train) & set(valid):
        raise ValueError("Training/validation overlap")
    return train, valid


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

    def __init__(self, path, dataset="IEMOCAP", split="train", *,
                 include_transcript=False, transcripts_path=None):
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
        transcripts = None
        if include_transcript:
            if transcripts_path:
                transcripts = json.loads(Path(transcripts_path).read_text(encoding="utf-8"))
            elif dataset in ("IEMOCAP", "MELD"):
                transcripts = raw[9 if dataset == "IEMOCAP" else 10]
            if not isinstance(transcripts, dict):
                raise ValueError("Supply transcript JSON {dialogue_id: [utterance strings]} for this dataset")
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
            if include_transcript:
                sentences = transcripts.get(key, transcripts.get(str(key)))
                if (not isinstance(sentences, (list, tuple)) or len(sentences) != n or
                        any(not isinstance(x, str) or not x.strip() for x in sentences)):
                    raise ValueError(f"Missing/invalid transcripts for dialogue {key}")
                self.records[-1]["transcript"] = list(sentences)
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
        speaker = torch.full((batch_size, max_length), -1, dtype=torch.long)
        features = {m: torch.zeros(batch_size, max_length, items[0][m].shape[1])
                    for m in ("text", "audio", "visual")}
        diff, same = [], []
        for b, item in enumerate(items):
            n = len(item["label"])
            mask[b, :n] = True
            labels[b, :n] = torch.as_tensor(item["label"], dtype=torch.long)
            speaker[b, :n] = torch.as_tensor(item["speaker"], dtype=torch.long)
            for m in features:
                features[m][b, :n] = torch.as_tensor(item[m], dtype=torch.float32)
            d, s = build_neighbors(item["speaker"], self.local_window)
            diff.append(d)
            same.append(s)
        batch = {**features, "mask": mask, "label": labels, "speaker": speaker,
                "diff": pack_neighbors(diff, max_length),
                "same": pack_neighbors(same, max_length),
                "ids": [item["id"] for item in items]}
        if any("transcript" in item for item in items):
            if not all("transcript" in item for item in items):
                raise ValueError("Cannot mix records with and without transcripts")
            batch["transcript"] = [item["transcript"] for item in items]
        return batch


def to_device(batch, device):
    return {k: ({name: tensor.to(device) for name, tensor in value.items()}
                if isinstance(value, dict) else value.to(device) if torch.is_tensor(value) else value)
            for k, value in batch.items()}
