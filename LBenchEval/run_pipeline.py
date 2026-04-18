#!/usr/bin/env python3
"""Run LongBench chunk retrieval plus optional compression into reader-ready prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compressors import SearchResult, SUPPORTED_METHODS, get_compressor  # noqa: E402
from retrieve_chunks import (  # noqa: E402
    DEFAULT_EMBED_BATCH_SIZE,
    DEFAULT_QUERY_INSTRUCTION,
    DEFAULT_TOP_K,
    DEFAULT_WORKERS,
    Embedder,
    RetrievalRecord,
    build_retrieval_query,
    format_query_for_model,
    load_or_create_query_embeddings,
    read_json,
    read_jsonl,
    retrieve_one,
    uses_harrier_instruction,
)

try:
    import tiktoken
except ImportError:  # pragma: no cover - environment dependent
    tiktoken = None


METHOD_NONE = "none"
METHOD_CHOICES = (METHOD_NONE,) + SUPPORTED_METHODS


@dataclass
class SourceRecord:
    row_index: int
    schema: str
    source_id: str
    query: str
    metadata: Dict[str, Any]
    ground_truth: Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run retrieval plus optional compression for LongBench QA subsets.")
    parser.add_argument("--input", "-i", required=True, help="Input qa_only.json file used to build the chunk DB.")
    parser.add_argument("--chunk-db", required=True, help="Chunk DB directory produced by build_chunk_db.py.")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file for compressed reader prompts.")
    parser.add_argument("--method", "-m", required=True, choices=METHOD_CHOICES, help="Compression method, or 'none' for raw retrieved chunks.")
    parser.add_argument("--top-k", "-k", type=int, default=DEFAULT_TOP_K, help=f"Number of chunks to retrieve per item. Default: {DEFAULT_TOP_K}.")
    parser.add_argument("--embed-model", default=None, help="Query embedding model. Defaults to the chunk-db embedding model.")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE, help=f"Batch size for query embedding. Default: {DEFAULT_EMBED_BATCH_SIZE}.")
    parser.add_argument("--device", default="cuda", help="Embedding device. Use 'auto', 'cpu', or a CUDA device like 'cuda:0'.")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help=f"Number of retrieval workers. Default: {DEFAULT_WORKERS}.")
    parser.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION, help="Instruction prefix used for Harrier query embeddings.")
    parser.add_argument("--limit", type=int, default=None, help="Optional record limit for smoke tests or partial runs.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output file if it already exists.")
    return parser.parse_args()


def read_input_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        first = ""
        while True:
            ch = f.read(1)
            if not ch:
                return []
            if not ch.isspace():
                first = ch
                break

    if first == "[":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array in {input_path}")
        return data

    rows: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} in {input_path}")
            rows.append(obj)
    return rows


def normalize_source_record(raw: Dict[str, Any], row_index: int) -> SourceRecord:
    if "question" in raw and "choice_A" in raw and "context" in raw:
        source_id = str(raw.get("_id", row_index))
        metadata = {
            "_id": source_id,
            "domain": raw.get("domain"),
            "sub_domain": raw.get("sub_domain"),
            "difficulty": raw.get("difficulty"),
            "length": raw.get("length"),
            "question": raw.get("question"),
            "choice_A": raw.get("choice_A"),
            "choice_B": raw.get("choice_B"),
            "choice_C": raw.get("choice_C"),
            "choice_D": raw.get("choice_D"),
            "answer": raw.get("answer"),
        }
        return SourceRecord(
            row_index=row_index,
            schema="longbench_v2",
            source_id=source_id,
            query=str(raw["question"]),
            metadata=metadata,
            ground_truth=raw.get("answer"),
        )

    if "input" in raw and "answers" in raw and "context" in raw:
        source_id = str(raw.get("_id", row_index))
        metadata = {
            "_id": source_id,
            "dataset": raw.get("dataset"),
            "language": raw.get("language"),
            "length": raw.get("length"),
            "input": raw.get("input"),
            "answers": raw.get("answers"),
            "all_classes": raw.get("all_classes"),
        }
        return SourceRecord(
            row_index=row_index,
            schema="longbench_v1",
            source_id=source_id,
            query=str(raw["input"]),
            metadata=metadata,
            ground_truth=raw.get("answers"),
        )

    keys = ", ".join(sorted(raw.keys()))
    raise ValueError(f"Unsupported input record schema at row {row_index}. Keys: {keys}")


def verify_alignment(source_records: Sequence[SourceRecord], db_records: Sequence[Dict[str, Any]]) -> None:
    if len(source_records) != len(db_records):
        raise ValueError(
            f"Input and chunk DB record counts differ: {len(source_records)} vs {len(db_records)}"
        )

    for idx, (source, db_row) in enumerate(zip(source_records, db_records)):
        db_source_id = str(db_row.get("source_id"))
        db_schema = db_row.get("schema")
        if source.source_id != db_source_id or source.schema != db_schema:
            raise ValueError(
                "Input and chunk DB are not aligned at row "
                f"{idx}: input ({source.schema}, {source.source_id}) vs "
                f"chunk DB ({db_schema}, {db_source_id})"
            )


def output_path_guard(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output file {output_path} already exists. Use --overwrite.")
    output_path.parent.mkdir(parents=True, exist_ok=True)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    if tiktoken is None:
        return len(text.split())
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text, disallowed_special=()))


def concat_texts(texts: Iterable[str]) -> str:
    return "\n\n".join(text.strip() for text in texts if text and text.strip())


def make_search_results(retrieved_chunks: Sequence[Dict[str, Any]]) -> List[SearchResult]:
    results: List[SearchResult] = []
    for chunk in retrieved_chunks:
        position = int(chunk.get("position", len(results)))
        results.append(
            SearchResult(
                evi_id=position,
                docid=position,
                title="",
                text=chunk.get("text", ""),
                score=float(chunk.get("score", 0.0)),
            )
        )
    return results


def format_reader_prompt(record: SourceRecord, context: str) -> str:
    if record.schema == "longbench_v2":
        meta = record.metadata
        return (
            "You are given context and a multiple-choice question. Think carefully, then choose the best answer.\n\n"
            "<context>\n"
            f"{context}\n"
            "</context>\n\n"
            "Question:\n"
            f"{meta.get('question') or record.query}\n\n"
            "Choices:\n"
            f"(A) {meta.get('choice_A')}\n"
            f"(B) {meta.get('choice_B')}\n"
            f"(C) {meta.get('choice_C')}\n"
            f"(D) {meta.get('choice_D')}\n\n"
            "Return the final answer in exactly this format:\n"
            "The correct answer is (A)"
        )

    return (
        "You are given context and a question. Think carefully, then answer using only the information supported by the context.\n\n"
        "<context>\n"
        f"{context}\n"
        "</context>\n\n"
        "Question:\n"
        f"{record.query}\n\n"
        "Return a concise final answer."
    )


def build_retrieval_records(
    db_dir: Path,
    source_records: Sequence[SourceRecord],
    db_records: Sequence[Dict[str, Any]],
    model_name: str,
    query_instruction: str,
) -> List[RetrievalRecord]:
    retrieval_records: List[RetrievalRecord] = []

    for idx, (source, db_row) in enumerate(zip(source_records, db_records)):
        row_for_query = {
            "schema": source.schema,
            "query": source.query,
            "record_metadata": source.metadata,
        }
        retrieval_query = build_retrieval_query(row_for_query)
        retrieval_query = format_query_for_model(retrieval_query, model_name, query_instruction)
        retrieval_records.append(
            RetrievalRecord(
                row_index=idx,
                pool_id=db_row["pool_id"],
                source_id=source.source_id,
                schema=source.schema,
                query=source.query,
                retrieval_query=retrieval_query,
                record_metadata=source.metadata,
                pool_dir=db_dir / db_row["pool_dir"],
            )
        )

    return retrieval_records


def run_retrieval(
    db_dir: Path,
    retrieval_records: Sequence[RetrievalRecord],
    embedder: Embedder,
    top_k: int,
    workers: int,
) -> tuple[List[Dict[str, Any]], bool, Path]:
    query_embeddings, used_cache, cache_path = load_or_create_query_embeddings(
        db_dir=db_dir,
        records=retrieval_records,
        embedder=embedder,
    )

    worker_count = max(1, min(workers, len(retrieval_records) or 1))
    results: List[Optional[Dict[str, Any]]] = [None] * len(retrieval_records)
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(retrieve_one, record, query_embeddings[idx], top_k)
            for idx, record in enumerate(retrieval_records)
        ]
        for future in tqdm(futures, desc="Retrieving", total=len(futures)):
            result = future.result()
            results[result["row_index"]] = result

    return [result for result in results if result is not None], used_cache, cache_path


def compress_record(
    method: str,
    compressor,
    source: SourceRecord,
    retrieval_result: Dict[str, Any],
) -> Dict[str, Any]:
    retrieved_chunks = retrieval_result.get("retrieved", [])
    raw_retrieved_context = concat_texts(chunk.get("text", "") for chunk in retrieved_chunks)
    original_tokens = count_tokens(raw_retrieved_context)

    if method == METHOD_NONE:
        final_context = raw_retrieved_context
    else:
        search_results = make_search_results(retrieved_chunks)
        compressed_docs = compressor.compress(source.query, search_results)
        final_context = concat_texts(doc.text for doc in compressed_docs)

    compressed_tokens = count_tokens(final_context)
    prompt_api = format_reader_prompt(source, final_context)

    return {
        "source_id": source.source_id,
        "schema": source.schema,
        "question": source.query,
        "record_metadata": source.metadata,
        "ground_truth": source.ground_truth,
        "retrieved": retrieved_chunks,
        "retrieved_context": raw_retrieved_context,
        "compressed_context": final_context,
        "prompt_api": prompt_api,
        "original_tokens": original_tokens,
        "compressed_tokens": compressed_tokens,
        "retrieved_chunk_count": len(retrieved_chunks),
    }


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    db_dir = Path(args.chunk_db)
    output_path = Path(args.output)

    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    output_path_guard(output_path, args.overwrite)

    db_manifest = read_json(db_dir / "manifest.json")
    db_records = read_jsonl(db_dir / "records.jsonl")
    source_records = [
        normalize_source_record(raw, idx)
        for idx, raw in enumerate(read_input_records(input_path))
    ]

    if args.limit is not None:
        source_records = source_records[:args.limit]
        db_records = db_records[:args.limit]

    verify_alignment(source_records, db_records)

    model_name = args.embed_model or db_manifest["embedding"]["model"]
    print(f"Loading embedder: {model_name}")
    embedder = Embedder(model_name=model_name, batch_size=args.embed_batch_size, device=args.device)

    retrieval_records = build_retrieval_records(
        db_dir=db_dir,
        source_records=source_records,
        db_records=db_records,
        model_name=model_name,
        query_instruction=args.query_instruction,
    )

    print(f"Retrieving top-{args.top_k} chunks for {len(retrieval_records)} records")
    retrieval_results, used_query_cache, query_cache_path = run_retrieval(
        db_dir=db_dir,
        retrieval_records=retrieval_records,
        embedder=embedder,
        top_k=args.top_k,
        workers=args.workers,
    )

    compressor = None
    if args.method != METHOD_NONE:
        compressor = get_compressor(args.method)
        print(f"Compressor '{compressor.__class__.__name__}' loaded.")

    compressed_data: List[Dict[str, Any]] = []
    total_original_tokens = 0
    total_compressed_tokens = 0

    for source, retrieval_result in tqdm(
        zip(source_records, retrieval_results),
        desc="Compressing",
        total=len(source_records),
    ):
        entry = compress_record(
            method=args.method,
            compressor=compressor,
            source=source,
            retrieval_result=retrieval_result,
        )
        total_original_tokens += entry["original_tokens"]
        total_compressed_tokens += entry["compressed_tokens"]
        compressed_data.append(entry)

    compression_ratio = (
        total_compressed_tokens / total_original_tokens
        if total_original_tokens > 0
        else 0.0
    )

    output_data = {
        "metadata": {
            "builder": "LBenchEval/run_pipeline.py",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "input_file": str(input_path),
            "chunk_db": str(db_dir),
            "method": args.method,
            "top_k": args.top_k,
            "total_questions": len(compressed_data),
            "compression_ratio": compression_ratio,
            "total_original_tokens": total_original_tokens,
            "total_compressed_tokens": total_compressed_tokens,
            "retriever": {
                "embed_model": model_name,
                "embed_batch_size": args.embed_batch_size,
                "device": embedder.device,
                "pooling": embedder.pooling,
                "workers": max(1, min(args.workers, len(retrieval_records) or 1)),
                "query_instruction": (
                    args.query_instruction if uses_harrier_instruction(model_name) else None
                ),
                "used_query_cache": used_query_cache,
                "query_cache_path": str(query_cache_path),
            },
        },
        "data": compressed_data,
    }

    write_json(output_path, output_data)
    print("Pipeline complete.")
    print(f"Questions: {len(compressed_data)}")
    print(f"Compression ratio: {compression_ratio:.4f}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
