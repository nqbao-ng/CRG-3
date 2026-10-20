"""Three contextual feature tokens -> causal LLM -> constrained emotion generation.

Projectors remain independent. Cross-modal interaction happens in the LLM's
causal self-attention, not in a sum or an extra fusion block. Dependencies are
lazy: the classifier baseline does not require transformers/PEFT/bitsandbytes.
"""
from dataclasses import asdict
import json
from pathlib import Path

import torch
import torch.nn as nn


FEATURE_MARKERS = ("[[CRG3_T]]", "[[CRG3_A]]", "[[CRG3_V]]")
LLM_MODALITIES = ("t", "a", "v")


def load_language_model(config):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("Install CRG3/requirements-llm.txt to use decoder_mode='llm'") from exc
    device = torch.device(config.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LLM CUDA requested but unavailable; use CPU + float32 without 4-bit for small tests")
    if config.load_in_4bit and device.type != "cuda":
        raise ValueError("This runner supports 4-bit LLM training on CUDA only")
    if config.compute_dtype == "bfloat16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise ValueError("bfloat16 is unsupported on this GPU; use float16 on T4")
    dtype = getattr(torch, config.compute_dtype)
    kwargs = dict(torch_dtype=dtype, attn_implementation="sdpa")
    if config.load_in_4bit:
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype), device_map={"": str(device)})
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)
    if not config.load_in_4bit:
        model.to(device)
    model.requires_grad_(False)
    if config.load_in_4bit or config.lora_rank:
        try:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        except ImportError as exc:
            raise ImportError("PEFT is required for QLoRA/LoRA") from exc
        if config.load_in_4bit:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=config.gradient_checkpointing,
                gradient_checkpointing_kwargs={"use_reentrant": False})
        if config.lora_rank:
            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=config.lora_rank, lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout, target_modules=["q_proj", "v_proj"], bias="none"))
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False  # generation explicitly enables its KV cache
    return model, tokenizer


class MultimodalLLMDecoder(nn.Module):
    def __init__(self, graph_dim, label_names, config, *, language_model=None, tokenizer=None):
        super().__init__()
        self.config = config
        if (language_model is None) != (tokenizer is None):
            raise ValueError("Provide both language_model and tokenizer, or neither")
        if not label_names or any(not isinstance(x, str) or not x.strip() for x in label_names):
            raise ValueError("label_names must contain nonempty strings in class-ID order")
        self.label_names = tuple(label_names)
        if len(set(self.label_names)) != len(self.label_names):
            raise ValueError("Emotion names must be unique")
        if language_model is None:
            language_model, tokenizer = load_language_model(config)
        else:
            # Injection is for offline tests or an already constructed frozen backbone.
            # Production LoRA is configured by load_language_model.
            language_model.requires_grad_(False)
        self.language_model, self.tokenizer = language_model, tokenizer
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define an EOS token")
        self.eos_id = int(tokenizer.eos_token_id)
        self.pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else self.eos_id)
        self.label_tokens = tuple(tuple(self._encode(name)) for name in self.label_names)
        if any(not ids or self.eos_id in ids or self.pad_id in ids for ids in self.label_tokens):
            raise ValueError("Emotion names must tokenize into nonempty ordinary token sequences")
        if len(set(self.label_tokens)) != len(self.label_tokens):
            raise ValueError("Emotion names have duplicate tokenizations")
        self.token_to_label = {ids: i for i, ids in enumerate(self.label_tokens)}
        embedding = self.language_model.get_input_embeddings().weight
        self.projectors = nn.ModuleDict({
            m: nn.Linear(graph_dim, embedding.shape[1], device=embedding.device, dtype=torch.float32)
            for m in LLM_MODALITIES})
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if not any(p.requires_grad for p in self.language_model.parameters()):
            self.language_model.eval()
        return self

    def _encode(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _prompt_segments(self, transcript, speaker):
        if not isinstance(transcript, str) or not transcript.strip():
            raise ValueError("LLM mode requires a nonempty raw transcript for each target")
        if any(marker in transcript for marker in FEATURE_MARKERS):
            raise ValueError("Transcript contains a reserved feature marker")
        choices = ", ".join(self.label_names)
        def render(text):
            content = (f"Classify the emotion of the target utterance using its contextual multimodal features.\n"
                       f"Text feature: {FEATURE_MARKERS[0]}\nAudio feature: {FEATURE_MARKERS[1]}\n"
                       f"Visual feature: {FEATURE_MARKERS[2]}\nTarget speaker: {speaker}\n"
                       f"Target utterance: {text}\nReturn exactly one label from: {choices}.")
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            pieces = [prompt]
            segments = []
            for marker in FEATURE_MARKERS:
                left, right = pieces[-1].split(marker, 1)
                segments.append(self._encode(left))
                pieces = [right]
            segments.append(self._encode(pieces[-1]))
            return segments
        segments = render(transcript)
        if sum(map(len, segments)) + 3 > self.config.max_prompt_tokens:
            # Truncate only transcript content. Keep instruction, modalities and answer cue.
            tokens = self._encode(transcript)
            low, high, best = 0, len(tokens), None
            while low <= high:
                middle = (low + high) // 2
                candidate = render(self.tokenizer.decode(tokens[:middle], skip_special_tokens=True))
                if sum(map(len, candidate)) + 3 <= self.config.max_prompt_tokens:
                    best, low = candidate, middle + 1
                else:
                    high = middle - 1
            if best is None or low <= 1:
                raise ValueError("max_prompt_tokens is too small for instruction, labels and target text")
            segments = best
        return segments

    def _indices(self, batch, target_indices):
        valid = batch["mask"]
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("mask must be a boolean [dialogue, utterance] tensor")
        indices = valid.nonzero() if target_indices is None else torch.as_tensor(
            target_indices, device=valid.device, dtype=torch.long)
        if indices.ndim != 2 or indices.shape[1] != 2 or not len(indices):
            raise ValueError("Select at least one target using [row, utterance] pairs")
        if (indices < 0).any() or (indices[:, 0] >= valid.shape[0]).any() or (indices[:, 1] >= valid.shape[1]).any():
            raise ValueError("Target indices are out of bounds")
        if not valid[indices[:, 0], indices[:, 1]].all():
            raise ValueError("Cannot decode padding targets")
        return indices

    def build_inputs(self, q, batch, *, with_answers, target_indices=None):
        """Insert one soft token per modality; supervise only answer/EOS tokens.

        Prompts contain the target transcript and speaker, never future text or
        ground-truth history labels. q supplies the graph's contextual history.
        """
        indices = self._indices(batch, target_indices)
        if "transcript" not in batch or "speaker" not in batch:
            raise ValueError("LLM batches require transcript and speaker metadata")
        for m in LLM_MODALITIES:
            if q[m].shape[:2] != batch["mask"].shape or q[m].ndim != 3:
                raise ValueError("Each graph feature must have shape [B,N,graph_dim]")
        embedding = self.language_model.get_input_embeddings()
        device, dtype = embedding.weight.device, embedding.weight.dtype
        selected = {m: self.projectors[m](q[m][indices[:, 0].to(q[m].device),
                                                      indices[:, 1].to(q[m].device)].to(
            device=device, dtype=self.projectors[m].weight.dtype)).to(dtype) for m in LLM_MODALITIES}
        rows, supervision = [], []
        for row, (b, i) in enumerate(indices.detach().cpu().tolist()):
            speaker = int(batch["speaker"][b, i])
            segments = self._prompt_segments(batch["transcript"][b][i], speaker)
            pieces = []
            for k, ids in enumerate(segments):
                pieces.append(embedding(torch.tensor(ids, device=device, dtype=torch.long)))
                if k < 3:
                    pieces.append(selected[LLM_MODALITIES[k]][row:row + 1])
            prompt = torch.cat(pieces, dim=0)
            labels = torch.full((len(prompt),), -100, device=device, dtype=torch.long)
            if with_answers:
                target = int(batch["label"][b, i])
                if not 0 <= target < len(self.label_names):
                    raise ValueError("Target emotion ID is outside label_names")
                answer = torch.tensor((*self.label_tokens[target], self.eos_id), device=device)
                prompt = torch.cat((prompt, embedding(answer)), dim=0)
                labels = torch.cat((labels, answer))
            rows.append(prompt)
            supervision.append(labels)
        width = max(map(len, rows))
        padded, masks, labels = [], [], []
        for row, targets in zip(rows, supervision):
            padding = width - len(row)
            # Left padding supports decoder generation; cumsum positions ignore padding.
            padded.append(torch.cat((row.new_zeros(padding, row.shape[-1]), row), dim=0))
            masks.append(torch.cat((torch.zeros(padding, device=device, dtype=torch.long),
                                    torch.ones(len(row), device=device, dtype=torch.long))))
            labels.append(torch.cat((torch.full((padding,), -100, device=device), targets)))
        mask = torch.stack(masks)
        inputs = dict(inputs_embeds=torch.stack(padded), attention_mask=mask,
                      position_ids=(mask.cumsum(-1) - 1).clamp_min(0))
        if with_answers:
            inputs["labels"] = torch.stack(labels)
        return inputs, indices

    def forward(self, q, batch, target_indices=None):
        inputs, _ = self.build_inputs(q, batch, with_answers=True, target_indices=target_indices)
        # Do not use no_grad here: even a frozen LLM must backpropagate to projectors.
        return self.language_model(**inputs, use_cache=False, return_dict=True)

    def allowed_next_tokens(self, prefix):
        allowed = set()
        for label in self.label_tokens:
            if label[:len(prefix)] == tuple(prefix):
                allowed.add(self.eos_id if len(label) == len(prefix) else label[len(prefix)])
        if not allowed:
            raise RuntimeError("Generated prefix is outside the emotion vocabulary")
        return sorted(allowed)

    @torch.no_grad()
    def generate_labels(self, q, batch, target_indices=None):
        """Greedy autoregressive decoding constrained to label tokens + EOS.

        This generates tokens; it does not substitute a classifier or score all
        complete label strings. KV cache avoids recomputing the full prompt.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before deterministic label generation")
        inputs, indices = self.build_inputs(q, batch, with_answers=False, target_indices=target_indices)
        output = self.language_model(**inputs, use_cache=True, return_dict=True)
        mask = inputs["attention_mask"]
        prefixes = [[] for _ in range(len(indices))]
        finished = [False] * len(indices)
        for _ in range(max(map(len, self.label_tokens)) + 1):
            chosen = []
            active = torch.tensor([not done for done in finished], device=mask.device, dtype=mask.dtype)
            for row, prefix in enumerate(prefixes):
                if finished[row]:
                    chosen.append(self.pad_id)
                    continue
                allowed = self.allowed_next_tokens(prefix)
                values = output.logits[row, -1, allowed]
                token = allowed[int(values.argmax())]
                chosen.append(token)
                if token == self.eos_id:
                    finished[row] = True
                else:
                    prefix.append(token)
            if all(finished):
                break
            mask = torch.cat((mask, active[:, None]), dim=1)
            output = self.language_model(
                input_ids=torch.tensor(chosen, device=mask.device)[:, None], attention_mask=mask,
                position_ids=(mask.sum(-1) - 1).clamp_min(0)[:, None],
                past_key_values=output.past_key_values, use_cache=True, return_dict=True)
        if not all(finished):
            raise RuntimeError("Constrained generation did not terminate")
        ids = [self.token_to_label[tuple(prefix)] for prefix in prefixes]
        result = torch.full(batch["mask"].shape, -100, device=mask.device, dtype=torch.long)
        result[indices[:, 0].to(mask.device), indices[:, 1].to(mask.device)] = torch.tensor(ids, device=mask.device)
        return {"label_ids": result, "labels": [self.label_names[i] for i in ids],
                "token_ids": prefixes, "target_indices": indices}

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.projectors.state_dict(), path / "projectors.pt")
        (path / "decoder_config.json").write_text(json.dumps(
            {"llm_config": asdict(self.config), "label_names": self.label_names}, indent=2), encoding="utf-8")
        if hasattr(self.language_model, "peft_config"):
            self.language_model.save_pretrained(path / "lora")
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path / "tokenizer")

    def load_decoder_weights(self, path):
        path = Path(path)
        saved = json.loads((path / "decoder_config.json").read_text(encoding="utf-8"))
        saved_cfg, current_cfg = dict(saved["llm_config"]), asdict(self.config)
        saved_cfg.pop("device", None)
        current_cfg.pop("device", None)
        if tuple(saved["label_names"]) != self.label_names or saved_cfg != current_cfg:
            raise ValueError("Decoder label/config mismatch")
        self.projectors.load_state_dict(torch.load(path / "projectors.pt", map_location="cpu", weights_only=True))
        if self.config.lora_rank:
            from peft import load_peft_weights, set_peft_model_state_dict
            weights = load_peft_weights(str(path / "lora"), device="cpu")
            set_peft_model_state_dict(self.language_model, weights)
