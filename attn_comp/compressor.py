from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .attention import PrunedLlamaAttention


DEFAULT_SYSTEM_PROMPT = (
    "I will provide you with some documents and a question. "
    "Please read the documents carefully and answer the question based on the information provided.\n"
    "The following are the given documents.\n\n"
)
DEFAULT_CHECKPOINT_NAME = "llama-attention-layer13-SFT_epoch-7.pth"


@dataclass
class CompressionResult:
    question: str
    kept_indices: list[int]
    ranked_indices: list[int]
    kept_chunks: list[Any]
    compressed_context: str
    doc_scores: list[float]
    instruction_score: float
    p: float
    epsilon: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AttnCompCompressor:
    def __init__(
        self,
        model_name_or_path: str,
        checkpoint_path: str | Path | None = None,
        *,
        layer: int = 13,
        window_size: int = 32,
        num_heads: int = 16,
        device_map: str | dict[str, Any] = "auto",
        dtype: torch.dtype | str | None = None,
        attn_implementation: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.layer = layer
        self.window_size = window_size
        self.num_heads = num_heads
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)

        base_config = AutoConfig.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        truncated_config = base_config.to_dict()
        truncated_config["num_hidden_layers"] = layer
        truncated_config.pop("torch_dtype", None)
        truncated_config = base_config.__class__.from_dict(truncated_config)

        self.dtype = self._resolve_dtype(base_config, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs: dict[str, Any] = {
            "config": truncated_config,
            "device_map": device_map,
            "torch_dtype": self.dtype,
            "local_files_only": local_files_only,
        }
        chosen_attn_impl = attn_implementation or self._default_attn_implementation()
        if chosen_attn_impl is not None:
            model_kwargs["attn_implementation"] = chosen_attn_impl

        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.model.eval()

        self.attention = PrunedLlamaAttention(base_config, num_heads=num_heads)
        self.attention.load_state_dict(self._load_state_dict(self.checkpoint_path))
        self.attention.to(device=self.model_device, dtype=self.dtype)
        self.attention.eval()

    @property
    def model_device(self) -> torch.device:
        return getattr(self.model, "device", next(self.model.parameters()).device)

    def compress(
        self,
        question: str,
        chunks: Sequence[str | Mapping[str, Any]],
        *,
        p: float = 0.9,
        epsilon: float = 1e-2,
    ) -> CompressionResult:
        chunks = list(chunks)
        if not chunks:
            return CompressionResult(
                question=question,
                kept_indices=[],
                ranked_indices=[],
                kept_chunks=[],
                compressed_context="",
                doc_scores=[],
                instruction_score=1.0,
                p=p,
                epsilon=epsilon,
            )

        input_ids, doc_ranges, normalized_chunks = self._build_inputs(question, chunks)
        batch = self.tokenizer.pad({"input_ids": [input_ids]}, padding=True, return_tensors="pt")

        with torch.no_grad():
            hidden_states = self.model(
                input_ids=batch["input_ids"].to(self.model_device),
                attention_mask=batch["attention_mask"].to(self.model_device),
                use_cache=False,
            ).last_hidden_state

        attention_mask = batch["attention_mask"].clone().to(hidden_states.device)
        pad_count = int((attention_mask[0] == 0).sum().item())
        doc_end = doc_ranges[-1][1] + pad_count
        attention_mask[0, doc_end:] = 0

        position_ids = torch.arange(hidden_states.shape[1], device=hidden_states.device).unsqueeze(0)
        batch_attn_weights = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            window_size=self.window_size,
        )

        doc_scores, ranked_indices, instruction_score = self._rank_documents(
            batch_attn_weights=batch_attn_weights,
            attention_mask=attention_mask,
            doc_ranges=doc_ranges,
        )

        kept_indices = self._select_documents(doc_scores, ranked_indices, p=p, epsilon=epsilon)
        kept_chunks = [chunks[index] for index in kept_indices]
        compressed_context = "\n\n".join(normalized_chunks[index] for index in kept_indices)

        return CompressionResult(
            question=question,
            kept_indices=kept_indices,
            ranked_indices=ranked_indices,
            kept_chunks=kept_chunks,
            compressed_context=compressed_context,
            doc_scores=doc_scores.tolist(),
            instruction_score=instruction_score,
            p=p,
            epsilon=epsilon,
        )

    def _build_inputs(
        self,
        question: str,
        chunks: Sequence[str | Mapping[str, Any]],
    ) -> tuple[torch.Tensor, list[tuple[int, int]], list[str]]:
        instruction_ids = self.tokenizer.encode(
            DEFAULT_SYSTEM_PROMPT,
            return_tensors="pt",
            add_special_tokens=True,
        )
        prompt_length = instruction_ids.size(1)
        doc_ids: list[torch.Tensor] = []
        doc_ranges: list[tuple[int, int]] = []
        normalized_chunks: list[str] = []

        for index, chunk in enumerate(chunks):
            chunk_text = self._normalize_chunk(chunk)
            normalized_chunks.append(chunk_text)
            content = f"Doc {index + 1}: {chunk_text}\n\n"
            current_doc_ids = self.tokenizer.encode(content, return_tensors="pt", add_special_tokens=False)
            doc_ids.append(current_doc_ids)
            doc_ranges.append((prompt_length, prompt_length + current_doc_ids.size(1)))
            prompt_length += current_doc_ids.size(1)

        question_text = question.strip()
        if not question_text.endswith("?"):
            question_text += "?"
        question_ids = self.tokenizer.encode(
            f"Question: {question_text}",
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_ids = torch.cat([instruction_ids, *doc_ids, question_ids], dim=1)
        return input_ids[0], doc_ranges, normalized_chunks

    def _rank_documents(
        self,
        *,
        batch_attn_weights: torch.Tensor,
        attention_mask: torch.Tensor,
        doc_ranges: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, list[int], float]:
        attn_weights = batch_attn_weights[0].mean(dim=0)
        mask = attention_mask[0]
        query_mask = 1 - mask[-attn_weights.size(0) :]
        query_rows = attn_weights[query_mask.bool()]
        if query_rows.numel() == 0:
            raise RuntimeError("No query tokens were selected for compression scoring.")

        filtered_attn = query_rows.mean(dim=0)
        filtered_attn = filtered_attn[mask.bool()]
        doc_scores = torch.zeros(len(doc_ranges), device=filtered_attn.device, dtype=filtered_attn.dtype)
        for index, (start, end) in enumerate(doc_ranges):
            doc_scores[index] = filtered_attn[start:end].sum()

        ranked_indices = torch.argsort(doc_scores, descending=True).tolist()
        instruction_score = float(1.0 - doc_scores.sum().item())
        return doc_scores, ranked_indices, instruction_score

    @staticmethod
    def _select_documents(
        doc_scores: torch.Tensor,
        ranked_indices: list[int],
        *,
        p: float,
        epsilon: float,
    ) -> list[int]:
        running_score = 1.0 - doc_scores.sum()
        kept_indices: list[int] = []
        if float(running_score.item()) >= p:
            return kept_indices

        for index in ranked_indices:
            score = float(doc_scores[index].item())
            if score < epsilon:
                break
            kept_indices.append(index)
            running_score = running_score + doc_scores[index]
            if float(running_score.item()) >= p:
                break
        return kept_indices

    @staticmethod
    def _normalize_chunk(chunk: str | Mapping[str, Any]) -> str:
        if isinstance(chunk, str):
            return chunk.strip()

        if "contents" in chunk:
            contents = str(chunk["contents"]).strip()
            if "\n" in contents:
                title, body = contents.split("\n", 1)
                title = title.strip()
                body = body.strip()
                if title and body:
                    return f"{title}: {body}"
            return contents

        if "title" in chunk and ("text" in chunk or "content" in chunk):
            body = chunk.get("text", chunk.get("content", ""))
            return f"{chunk['title']}: {body}".strip()

        if "text" in chunk:
            return str(chunk["text"]).strip()

        if "content" in chunk:
            return str(chunk["content"]).strip()

        raise TypeError("Chunks must be strings or mappings with contents/text/content fields.")

    @staticmethod
    def _resolve_dtype(config: Any, dtype: torch.dtype | str | None) -> torch.dtype:
        if dtype is not None:
            if isinstance(dtype, str):
                return getattr(torch, dtype.replace("torch.", ""))
            return dtype

        config_dtype = getattr(config, "torch_dtype", None)
        if isinstance(config_dtype, str):
            return getattr(torch, config_dtype.replace("torch.", ""))
        if config_dtype is not None:
            return config_dtype
        if torch.cuda.is_available():
            return torch.bfloat16
        return torch.float32

    @staticmethod
    def _default_attn_implementation() -> str | None:
        if not torch.cuda.is_available():
            return "sdpa"
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            return "sdpa"
        return "flash_attention_2"

    @staticmethod
    def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
        try:
            return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(checkpoint_path, map_location="cpu")

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: str | Path | None) -> Path:
        if checkpoint_path is not None:
            return Path(checkpoint_path).expanduser().resolve()
        return Path(__file__).resolve().parent / "checkpoints" / DEFAULT_CHECKPOINT_NAME
