#!/usr/bin/env python3
"""Retrieve top-k chunks within each LongBench record's own chunk pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_TOP_K = 10
DEFAULT_WORKERS = min(8, max(1, os.cpu_count() or 1))
DEFAULT_QUERY_INSTRUCTION = "Given a question, retrieve relevant context chunks that help answer it."
MIN_TEXT_OVERLAP_CHARS = 24
MAX_TEXT_OVERLAP_SCAN = 4096


@dataclass
class RetrievalRecord:
    row_index: int
    pool_id: str
    source_id: str
    schema: str
    query: str
    retrieval_query: str
    record_metadata: Dict[str, Any]
    pool_dir: Path


class Embedder:
    """Text embedder matching the chunk-db embedding path."""

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
                max_length=min(getattr(self.tokenizer, "model_max_length", 512), 32768),
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
    parser = argparse.ArgumentParser(description="Retrieve top-k chunks within each LongBench chunk pool.")
    parser.add_argument("--input-db", "-i", required=True, help="Input chunk database directory.")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for retrieval results.")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help=f"Number of chunks to retrieve per pool. Default: {DEFAULT_TOP_K}.")
    parser.add_argument("--embed-model", default=None, help="Query embedding model. Defaults to the chunk-db embedding model.")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE, help=f"Batch size for query embedding. Default: {DEFAULT_EMBED_BATCH_SIZE}.")
    parser.add_argument("--device", default="cuda", help="Embedding device. Use 'auto', 'cpu', or a CUDA device like 'cuda:0'.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of retrieval workers. Default: {DEFAULT_WORKERS}.")
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION, help="Instruction prefix used for Harrier query embeddings.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests or partial runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output directory if it already exists.")
    return parser.parse_args()


def safe_name(value: str) -> str:
    import re
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    name = name.strip("._")
    return name or "artifact"


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


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def uses_harrier_instruction(model_name: str) -> bool:
    return "harrier" in model_name.lower()


def build_retrieval_query(record: Dict[str, Any]) -> str:
    schema = record["schema"]
    base_query = record["query"].strip()

    if schema != "longbench_v2":
        return base_query

    meta = record.get("record_metadata", {})
    choices = []
    for label in ["A", "B", "C", "D"]:
        value = meta.get(f"choice_{label}")
        if value:
            choices.append(f"({label}) {value}")

    if not choices:
        return base_query

    return f"{base_query}\nChoices:\n" + "\n".join(choices)


def format_query_for_model(query: str, model_name: str, instruction: str) -> str:
    if uses_harrier_instruction(model_name):
        return f"Instruct: {instruction}\nQuery: {query}"
    return query


def build_query_signature(model_name: str, formatted_queries: Sequence[str]) -> str:
    hasher = hashlib.sha1()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(b"\0")
    for query in formatted_queries:
        hasher.update(query.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def load_or_create_query_embeddings(
    db_dir: Path,
    records: Sequence[RetrievalRecord],
    embedder: Embedder,
) -> tuple[np.ndarray, bool, Path]:
    cache_dir = db_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    formatted_queries = [record.retrieval_query for record in records]
    signature = build_query_signature(embedder.model_name, formatted_queries)
    cache_stub = f"query_embeddings__{safe_name(embedder.model_name)}__{signature[:12]}"
    cache_npy = cache_dir / f"{cache_stub}.npy"
    cache_meta = cache_dir / f"{cache_stub}.json"

    if cache_npy.exists() and cache_meta.exists():
        meta = read_json(cache_meta)
        if (
            meta.get("model_name") == embedder.model_name
            and meta.get("record_count") == len(records)
            and meta.get("signature") == signature
        ):
            return np.load(cache_npy), True, cache_npy

    query_embeddings = embedder.encode(formatted_queries)
    np.save(cache_npy, query_embeddings)
    with cache_meta.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": embedder.model_name,
                "record_count": len(records),
                "signature": signature,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    return query_embeddings, False, cache_npy


def load_selected_chunks(chunks_path: Path, wanted_indices: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    wanted = {int(idx) for idx in wanted_indices}
    selected: Dict[int, Dict[str, Any]] = {}

    with chunks_path.open("r", encoding="utf-8") as f:
        for row_idx, line in enumerate(f):
            if row_idx not in wanted:
                continue
            selected[row_idx] = json.loads(line)
            if len(selected) == len(wanted):
                break

    return selected


def merge_texts_with_overlap(texts: Sequence[str]) -> str:
    if not texts:
        return ""

    merged = texts[0].strip()
    for text in texts[1:]:
        next_text = text.strip()
        if not next_text:
            continue
        merged = merge_two_texts(merged, next_text)
    return merged


def merge_two_texts(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left

    max_overlap = min(len(left), len(right), MAX_TEXT_OVERLAP_SCAN)
    for overlap in range(max_overlap, MIN_TEXT_OVERLAP_CHARS - 1, -1):
        if left[-overlap:] == right[:overlap]:
            return left + right[overlap:]

    if left.endswith(right):
        return left
    if right.startswith(left):
        return right

    return f"{left}\n\n{right}"


def select_top_k(scores: np.ndarray, top_k: int) -> np.ndarray:
    if scores.size == 0:
        return np.empty((0,), dtype=np.int64)

    k = min(top_k, scores.shape[0])
    if k == scores.shape[0]:
        return np.argsort(scores)[::-1]

    candidate_idx = np.argpartition(scores, -k)[-k:]
    return candidate_idx[np.argsort(scores[candidate_idx])[::-1]]


def retrieve_one(
    record: RetrievalRecord,
    query_vector: np.ndarray,
    top_k: int,
) -> Dict[str, Any]:
    pool_dir = record.pool_dir
    embeddings = np.load(pool_dir / "embeddings.npy", mmap_mode="r")
    if embeddings.shape[0] == 0:
        return {
            "row_index": record.row_index,
            "pool_id": record.pool_id,
            "source_id": record.source_id,
            "schema": record.schema,
            "query": record.query,
            "retrieval_query": record.retrieval_query,
            "record_metadata": record.record_metadata,
            "top_k": top_k,
            "retrieved": [],
            "ordered_chunk_ids": [],
            "merged_context": "",
            "retrieved_token_count": 0,
        }

    scores = embeddings @ query_vector
    top_indices = select_top_k(np.asarray(scores), top_k)
    chunk_rows = load_selected_chunks(pool_dir / "chunks.jsonl", top_indices)

    retrieved = []
    for rank, chunk_idx in enumerate(top_indices, start=1):
        chunk = dict(chunk_rows[int(chunk_idx)])
        chunk["rank"] = rank
        chunk["score"] = float(scores[int(chunk_idx)])
        chunk["embedding_row"] = int(chunk_idx)
        retrieved.append(chunk)

    ordered = sorted(retrieved, key=lambda item: item["position"])
    ordered_chunk_ids = [item["chunk_id"] for item in ordered]
    merged_context = merge_texts_with_overlap([item["text"] for item in ordered])
    retrieved_token_count = int(sum(item["num_tokens"] for item in ordered))

    return {
        "row_index": record.row_index,
        "pool_id": record.pool_id,
        "source_id": record.source_id,
        "schema": record.schema,
        "query": record.query,
        "retrieval_query": record.retrieval_query,
        "record_metadata": record.record_metadata,
        "top_k": int(min(top_k, embeddings.shape[0])),
        "retrieved": retrieved,
        "ordered_chunk_ids": ordered_chunk_ids,
        "merged_context": merged_context,
        "retrieved_token_count": retrieved_token_count,
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    args = parse_args()
    db_dir = Path(args.input_db)
    output_dir = Path(args.output_dir)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    ensure_output_dir(output_dir, args.overwrite)

    manifest = read_json(db_dir / "manifest.json")
    record_rows = read_jsonl(db_dir / "records.jsonl")
    if args.limit is not None:
        record_rows = record_rows[:args.limit]

    model_name = args.embed_model or manifest["embedding"]["model"]
    print(f"Loading embedder: {model_name}")
    embedder = Embedder(model_name=model_name, batch_size=args.embed_batch_size, device=args.device)

    retrieval_records: List[RetrievalRecord] = []
    for idx, row in enumerate(record_rows):
        retrieval_query = build_retrieval_query(row)
        retrieval_query = format_query_for_model(retrieval_query, model_name, args.query_instruction)
        retrieval_records.append(
            RetrievalRecord(
                row_index=idx,
                pool_id=row["pool_id"],
                source_id=row["source_id"],
                schema=row["schema"],
                query=row["query"],
                retrieval_query=retrieval_query,
                record_metadata=row.get("record_metadata", {}),
                pool_dir=db_dir / row["pool_dir"],
            )
        )

    print(f"Preparing {len(retrieval_records)} retrieval queries")
    query_embeddings, used_cache, cache_path = load_or_create_query_embeddings(
        db_dir=db_dir,
        records=retrieval_records,
        embedder=embedder,
    )

    workers = max(1, min(args.workers, len(retrieval_records) or 1))
    print(f"Retrieving top-{args.top_k} chunks with {workers} worker(s)")

    results: List[Optional[Dict[str, Any]]] = [None] * len(retrieval_records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(retrieve_one, record, query_embeddings[idx], args.top_k)
            for idx, record in enumerate(retrieval_records)
        ]
        for future in tqdm(futures, desc="Retrieving", total=len(futures)):
            result = future.result()
            results[result["row_index"]] = result

    ordered_results = [result for result in results if result is not None]
    write_jsonl(output_dir / "retrieved.jsonl", ordered_results)

    avg_top_k = (
        float(sum(result["top_k"] for result in ordered_results)) / len(ordered_results)
        if ordered_results else 0.0
    )
    avg_retrieved_tokens = (
        float(sum(result["retrieved_token_count"] for result in ordered_results)) / len(ordered_results)
        if ordered_results else 0.0
    )

    out_manifest = {
        "builder": "LBenchEval/retrieve_chunks.py",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_db_dir": str(db_dir),
        "output_dir": str(output_dir),
        "retrieval": {
            "top_k": args.top_k,
            "embed_model": model_name,
            "embed_batch_size": args.embed_batch_size,
            "device": embedder.device,
            "workers": workers,
            "query_instruction": args.query_instruction if uses_harrier_instruction(model_name) else None,
            "used_query_cache": used_cache,
            "query_cache_path": str(cache_path),
        },
        "stats": {
            "records": len(ordered_results),
            "avg_top_k": avg_top_k,
            "avg_retrieved_tokens": avg_retrieved_tokens,
        },
    }
    write_json(output_dir / "manifest.json", out_manifest)

    print("Retrieval complete.")
    print(f"Records: {len(ordered_results)}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
