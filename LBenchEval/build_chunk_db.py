#!/usr/bin/env python3
"""Build a per-question chunk database for LongBench v1/v2 QA subsets."""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

try:
    import tiktoken
except ImportError:  # pragma: no cover - environment dependent
    tiktoken = None

try:
    import spacy
except ImportError:  # pragma: no cover - environment dependent
    spacy = None


DEFAULT_CHUNK_SIZE = 400
OVERLAP_RATIO = 0.10
DEFAULT_EMBED_MODEL = "microsoft/harrier-oss-v1-270m"
DEFAULT_EMBED_BATCH_SIZE = 64
MAX_SENTENCE_BLOCK_CHARS = 20_000
MIN_PUNCT_SPLIT_POINT = 0.50
MAX_SPACY_SENTENCE_CHARS = 200_000


@dataclass
class NormalizedRecord:
    index: int
    schema: str
    pool_id: str
    source_id: str
    query: str
    context: str
    metadata: Dict[str, Any]


@dataclass
class TextUnit:
    text: str
    num_tokens: int


class TokenCounter:
    """Token counting and decode abstraction with tiktoken fallback."""

    def __init__(self, fallback_tokenizer) -> None:
        self.backend = "hf"
        self.tokenizer = fallback_tokenizer
        self.encoding = None

        if tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
                self.backend = "tiktoken:cl100k_base"
            except Exception:
                self.encoding = None

    def encode(self, text: str) -> List[int]:
        if self.encoding is not None:
            return self.encoding.encode(text, disallowed_special=())
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )
        return encoded["input_ids"]

    def decode(self, token_ids: Sequence[int]) -> str:
        if self.encoding is not None:
            return self.encoding.decode(list(token_ids))
        return self.tokenizer.decode(
            list(token_ids),
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def count(self, text: str) -> int:
        return len(self.encode(text))


class SentenceSplitter:
    """Sentence splitter backed by spaCy en_core_web_sm."""

    def __init__(self) -> None:
        self.backend = "spacy:en_core_web_sm"
        self.max_spacy_chars = MAX_SPACY_SENTENCE_CHARS

        if spacy is None:
            raise RuntimeError(
                "spaCy is required for chunking. Activate the correct environment first."
            )

        try:
            self.nlp = spacy.load(
                "en_core_web_sm",
                exclude=["tagger", "parser", "ner", "lemmatizer", "tok2vec"],
            )
            if "senter" not in self.nlp.pipe_names:
                self.nlp.enable_pipe("senter")
            self.nlp.max_length = max(self.nlp.max_length, self.max_spacy_chars + 1_000)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load spaCy model 'en_core_web_sm'. Activate env '312' or run "
                "'python -m spacy download en_core_web_sm'."
            ) from exc

    def split(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []

        if len(text) > self.max_spacy_chars:
            raise ValueError(
                f"Text block length {len(text)} still exceeds spaCy safety limit "
                f"{self.max_spacy_chars}. This should have been pre-split earlier."
            )

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences or [text]


class Embedder:
    """Passage embedder using a Hugging Face embedding model."""

    def __init__(self, model_name: str, batch_size: int, device: Optional[str]) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = self._resolve_device(device)
        self.pooling = "last_token" if self._uses_last_token_pooling(model_name) else "cls"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = self._load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: Optional[str]) -> str:
        if device and device != "auto":
            return device
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _load_model(model_name: str):
        try:
            return AutoModel.from_pretrained(model_name, dtype="auto")
        except TypeError:
            try:
                return AutoModel.from_pretrained(model_name, torch_dtype="auto")
            except TypeError:
                return AutoModel.from_pretrained(model_name)

    @staticmethod
    def _uses_last_token_pooling(model_name: str) -> bool:
        return "harrier" in model_name.lower()

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.zeros((0, hidden_size), dtype=np.float32)

        outputs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=min(
                    getattr(self.tokenizer, "model_max_length", 512),
                    512,
                ),
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.inference_mode():
                model_out = self.model(**encoded)
                if self.pooling == "last_token":
                    pooled = self._pool_last_token(
                        model_out.last_hidden_state,
                        encoded["attention_mask"],
                    )
                else:
                    pooled = model_out.last_hidden_state[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

            outputs.append(pooled.detach().to(torch.float32).cpu().numpy())

        return np.concatenate(outputs, axis=0)

    @staticmethod
    def _pool_last_token(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = bool(torch.all(attention_mask[:, -1] == 1).item())
        if left_padding:
            return last_hidden_states[:, -1]

        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-question chunk database for LongBench QA subsets.")
    parser.add_argument("--input", "-i", required=True, help="Input qa_only.json file from LongBench v1 or v2.")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for the chunk database.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help=f"Chunk size in tokens. Default: {DEFAULT_CHUNK_SIZE}.")
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL, help=f"Embedding model name. Default: {DEFAULT_EMBED_MODEL}.")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE, help=f"Embedding batch size. Default: {DEFAULT_EMBED_BATCH_SIZE}.")
    parser.add_argument("--device", default="cuda", help="Embedding device. Use 'auto', 'cpu', or a CUDA device like 'cuda:0'.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests or partial builds.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it already exists.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    name = name.strip("._")
    return name or "record"


def read_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        while True:
            ch = f.read(1)
            if not ch:
                return []
            if not ch.isspace():
                first_char = ch
                break

    if first_char == "[":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected a top-level JSON array in {input_path}")
        return data

    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid JSON on line {line_no} of {input_path}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {input_path}")
            records.append(obj)
    return records


def normalize_record(record: Dict[str, Any], index: int) -> NormalizedRecord:
    if "question" in record and "context" in record and "choice_A" in record:
        source_id = str(record.get("_id", index))
        pool_id = f"{index:06d}_{safe_name(source_id)}"
        metadata = {
            "_id": source_id,
            "domain": record.get("domain"),
            "sub_domain": record.get("sub_domain"),
            "difficulty": record.get("difficulty"),
            "length": record.get("length"),
            "question": record.get("question"),
            "choice_A": record.get("choice_A"),
            "choice_B": record.get("choice_B"),
            "choice_C": record.get("choice_C"),
            "choice_D": record.get("choice_D"),
            "answer": record.get("answer"),
        }
        return NormalizedRecord(
            index=index,
            schema="longbench_v2",
            pool_id=pool_id,
            source_id=source_id,
            query=str(record["question"]),
            context=str(record["context"]),
            metadata=metadata,
        )

    if "input" in record and "context" in record and "answers" in record:
        source_id = str(record.get("_id", index))
        pool_id = f"{index:06d}_{safe_name(source_id)}"
        metadata = {
            "_id": source_id,
            "dataset": record.get("dataset"),
            "language": record.get("language"),
            "length": record.get("length"),
            "input": record.get("input"),
            "answers": record.get("answers"),
            "all_classes": record.get("all_classes"),
        }
        return NormalizedRecord(
            index=index,
            schema="longbench_v1",
            pool_id=pool_id,
            source_id=source_id,
            query=str(record["input"]),
            context=str(record["context"]),
            metadata=metadata,
        )

    keys = ", ".join(sorted(record.keys()))
    raise ValueError(f"Unsupported record schema at index {index}. Keys: {keys}")


def ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if any(output_dir.iterdir()):
            if not overwrite:
                raise FileExistsError(
                    f"Output directory {output_dir} already exists and is not empty. "
                    "Use --overwrite to replace it."
                )
            shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def iter_coarse_blocks(text: str) -> Iterable[str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return

    blocks = [block.strip() for block in re.split(r"\n{2,}", normalized) if block.strip()]
    if not blocks:
        blocks = [normalized]

    for block in blocks:
        if len(block) <= MAX_SENTENCE_BLOCK_CHARS:
            yield block
            continue

        line_parts = [line.strip() for line in block.split("\n") if line.strip()]
        if len(line_parts) > 1:
            current = ""
            for raw_line in line_parts:
                for line in split_block_by_chars(raw_line, MAX_SENTENCE_BLOCK_CHARS):
                    proposed = f"{current}\n{line}".strip() if current else line
                    if current and len(proposed) > MAX_SENTENCE_BLOCK_CHARS:
                        yield current
                        current = line
                    else:
                        current = proposed
            if current:
                yield current
            continue

        yield from split_block_by_chars(block, MAX_SENTENCE_BLOCK_CHARS)


def split_block_by_chars(text: str, max_chars: int) -> Iterable[str]:
    text = text.strip()
    if not text:
        return

    if len(text) <= max_chars:
        yield text
        return

    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            window = text[start:end]
            split_points = [
                window.rfind(marker)
                for marker in [". ", "? ", "! ", "; ", "\n", ".\n", ":\n", ", "]
            ]
            split_at = max(split_points)
            if split_at >= int(max_chars * MIN_PUNCT_SPLIT_POINT):
                end = start + split_at + 1

        piece = text[start:end].strip()
        if piece:
            yield piece
        start = end


def split_oversized_text(
    text: str,
    token_counter: TokenCounter,
    max_tokens: int,
    overlap_tokens: int,
) -> List[str]:
    token_ids = token_counter.encode(text)
    if len(token_ids) <= max_tokens:
        return [text.strip()]

    pieces: List[str] = []
    step = max(1, max_tokens - overlap_tokens)
    start = 0

    while start < len(token_ids):
        end = min(start + max_tokens, len(token_ids))
        piece = token_counter.decode(token_ids[start:end]).strip()
        if piece:
            pieces.append(piece)
        if end >= len(token_ids):
            break
        start += step

    return pieces


def build_text_units(
    text: str,
    splitter: SentenceSplitter,
    token_counter: TokenCounter,
    chunk_size: int,
    overlap_tokens: int,
) -> List[TextUnit]:
    units: List[TextUnit] = []

    for block in iter_coarse_blocks(text):
        sentences = splitter.split(block)
        if not sentences:
            sentences = [block.strip()]

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            token_count = token_counter.count(sentence)
            if token_count <= chunk_size:
                units.append(TextUnit(text=sentence, num_tokens=token_count))
                continue

            for piece in split_oversized_text(
                sentence,
                token_counter,
                max_tokens=chunk_size,
                overlap_tokens=overlap_tokens,
            ):
                piece_tokens = token_counter.count(piece)
                units.append(TextUnit(text=piece, num_tokens=piece_tokens))

    if units:
        return units

    fallback = text.strip()
    if not fallback:
        return []

    return [
        TextUnit(text=piece, num_tokens=token_counter.count(piece))
        for piece in split_oversized_text(
            fallback,
            token_counter,
            max_tokens=chunk_size,
            overlap_tokens=overlap_tokens,
        )
    ]


def join_unit_texts(units: Sequence[TextUnit]) -> str:
    return "\n".join(unit.text for unit in units).strip()


def build_chunks_from_units(
    units: Sequence[TextUnit],
    chunk_size: int,
    overlap_tokens: int,
    pool_id: str,
) -> List[Dict[str, Any]]:
    if not units:
        return []

    unit_token_offsets: List[tuple[int, int]] = []
    cursor = 0
    for unit in units:
        start = cursor
        cursor += unit.num_tokens
        unit_token_offsets.append((start, cursor))

    chunks: List[Dict[str, Any]] = []
    start_idx = 0

    while start_idx < len(units):
        end_idx = start_idx
        total_tokens = 0

        while end_idx < len(units):
            next_tokens = units[end_idx].num_tokens
            if total_tokens > 0 and total_tokens + next_tokens > chunk_size:
                break
            total_tokens += next_tokens
            end_idx += 1

        if end_idx == start_idx:
            end_idx += 1
            total_tokens = units[start_idx].num_tokens

        chunk_units = units[start_idx:end_idx]
        token_start = unit_token_offsets[start_idx][0]
        token_end = unit_token_offsets[end_idx - 1][1]
        chunks.append(
            {
                "chunk_id": f"{pool_id}_chunk_{len(chunks):04d}",
                "pool_id": pool_id,
                "position": len(chunks),
                "unit_start": start_idx,
                "unit_end_exclusive": end_idx,
                "token_start": token_start,
                "token_end": token_end,
                "num_tokens": token_end - token_start,
                "text": join_unit_texts(chunk_units),
            }
        )

        if end_idx >= len(units):
            break

        next_start = end_idx
        overlapped = 0
        while next_start > start_idx:
            candidate = units[next_start - 1].num_tokens
            if overlapped >= overlap_tokens:
                break
            next_start -= 1
            overlapped += candidate
            if overlapped >= overlap_tokens:
                break

        if next_start <= start_idx:
            next_start = start_idx + 1
        start_idx = next_start

    return chunks


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def build_manifest(
    *,
    input_path: Path,
    output_dir: Path,
    chunk_size: int,
    overlap_tokens: int,
    embedder: Embedder,
    splitter: SentenceSplitter,
    token_counter: TokenCounter,
    total_records: int,
    processed_records: int,
    total_chunks: int,
    total_context_tokens: int,
) -> Dict[str, Any]:
    return {
        "builder": "LBenchEval/build_chunk_db.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_dir": str(output_dir),
        "chunking": {
            "chunk_size_tokens": chunk_size,
            "overlap_ratio": OVERLAP_RATIO,
            "overlap_tokens": overlap_tokens,
            "sentence_splitter": splitter.backend,
            "token_counter_backend": token_counter.backend,
        },
        "embedding": {
            "model": embedder.model_name,
            "batch_size": embedder.batch_size,
            "device": embedder.device,
            "pooling": embedder.pooling,
            "normalized": True,
            "dtype": "float32",
        },
        "stats": {
            "input_records": total_records,
            "processed_records": processed_records,
            "total_chunks": total_chunks,
            "total_context_tokens": total_context_tokens,
            "avg_chunks_per_record": (
                float(total_chunks) / processed_records if processed_records else 0.0
            ),
        },
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    overlap_tokens = max(1, int(math.floor(args.chunk_size * OVERLAP_RATIO)))

    ensure_output_dir(output_dir, args.overwrite)
    pools_dir = output_dir / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading embedder: {args.embed_model}")
    embedder = Embedder(
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
        device=args.device,
    )
    token_counter = TokenCounter(embedder.tokenizer)
    splitter = SentenceSplitter()

    print(f"Reading records from: {input_path}")
    raw_records = read_records(input_path)
    total_records = len(raw_records)
    if args.limit is not None:
        raw_records = raw_records[:args.limit]

    normalized_records = [
        normalize_record(record, index)
        for index, record in enumerate(raw_records)
    ]

    records_manifest_rows: List[Dict[str, Any]] = []
    total_chunks = 0
    total_context_tokens = 0

    for record in tqdm(normalized_records, desc="Building chunk pools"):
        pool_dir = pools_dir / record.pool_id
        pool_dir.mkdir(parents=True, exist_ok=True)

        context_token_count = token_counter.count(record.context)
        total_context_tokens += context_token_count

        units = build_text_units(
            record.context,
            splitter=splitter,
            token_counter=token_counter,
            chunk_size=args.chunk_size,
            overlap_tokens=overlap_tokens,
        )
        chunks = build_chunks_from_units(
            units,
            chunk_size=args.chunk_size,
            overlap_tokens=overlap_tokens,
            pool_id=record.pool_id,
        )
        embeddings = embedder.encode([chunk["text"] for chunk in chunks])

        if embeddings.shape[0] != len(chunks):
            raise RuntimeError(
                f"Embedding row count mismatch for pool {record.pool_id}: "
                f"{embeddings.shape[0]} vs {len(chunks)} chunks"
            )

        np.save(pool_dir / "embeddings.npy", embeddings)
        write_jsonl(pool_dir / "chunks.jsonl", chunks)

        pool_meta = {
            "pool_id": record.pool_id,
            "source_id": record.source_id,
            "schema": record.schema,
            "query": record.query,
            "chunk_count": len(chunks),
            "context_num_tokens": context_token_count,
            "chunk_size_tokens": args.chunk_size,
            "overlap_tokens": overlap_tokens,
            "paths": {
                "chunks": "chunks.jsonl",
                "embeddings": "embeddings.npy",
            },
            "record_metadata": record.metadata,
        }
        write_json(pool_dir / "meta.json", pool_meta)

        records_manifest_rows.append(
            {
                "pool_id": record.pool_id,
                "source_id": record.source_id,
                "schema": record.schema,
                "query": record.query,
                "chunk_count": len(chunks),
                "context_num_tokens": context_token_count,
                "pool_dir": str(Path("pools") / record.pool_id),
                "record_metadata": record.metadata,
            }
        )
        total_chunks += len(chunks)

    write_jsonl(output_dir / "records.jsonl", records_manifest_rows)

    manifest = build_manifest(
        input_path=input_path,
        output_dir=output_dir,
        chunk_size=args.chunk_size,
        overlap_tokens=overlap_tokens,
        embedder=embedder,
        splitter=splitter,
        token_counter=token_counter,
        total_records=total_records,
        processed_records=len(normalized_records),
        total_chunks=total_chunks,
        total_context_tokens=total_context_tokens,
    )
    write_json(output_dir / "manifest.json", manifest)

    print("Chunk DB build complete.")
    print(f"Processed records: {len(normalized_records)} / {total_records}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
