#!/usr/bin/env python3
"""Build a Contriever-retrieved QASPER benchmark file.

The output intentionally mirrors the JSONL-with-.json convention used by the
existing retrieved/contriever-msmarco_* benchmark files in this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tarfile
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


QASPER_VERSION = "0.3"
TRAIN_DEV_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz"
TEST_URL = "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz"
ARCHIVE_FILES = {
    "train": "qasper-train-v0.3.json",
    "dev": "qasper-dev-v0.3.json",
    "test": "qasper-test-v0.3.json",
}

DEFAULT_MODEL = "facebook/contriever-msmarco"
DEFAULT_CHUNK_WORDS = 100
DEFAULT_OVERLAP_RATIO = 0.15
DEFAULT_TOP_K = 100
DEFAULT_EMBED_BATCH_SIZE = 128
DEFAULT_QUERY_BATCH_SIZE = 128
CHUNKING_METHOD = "sentence_aware_v1"


@dataclass(frozen=True)
class Chunk:
    id: str
    paper_id: str
    title: str
    text: str
    source_split: str
    word_start: int
    word_end: int


@dataclass(frozen=True)
class SentenceUnit:
    text: str
    section_name: str
    word_start: int
    word_end: int
    num_words: int


@dataclass
class PreparedQuestion:
    record: dict[str, Any]
    retrieval_query: str
    answer_texts: list[str]
    evidence_texts: list[str]


class ContrieverEmbedder:
    """Mean-pooled Contriever encoder using dot-product scores."""

    def __init__(
        self,
        model_name: str,
        device: str,
        batch_size: int,
        max_length: int,
    ) -> None:
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def encode(self, texts: Sequence[str], desc: str) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for start in tqdm(range(0, len(texts), self.batch_size), desc=desc):
            batch = list(texts[start : start + self.batch_size])
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}

            with torch.inference_mode():
                model_out = self.model(**encoded)
                pooled = mean_pool(model_out.last_hidden_state, encoded["attention_mask"])

            outputs.append(pooled.detach().to(torch.float32).cpu().numpy())

        if not outputs:
            return np.zeros((0, int(self.model.config.hidden_size)), dtype=np.float32)
        return np.concatenate(outputs, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create retrieved QASPER JSONL files.")
    parser.add_argument("--output-dir", default="retrieved/contriever-msmarco_QASPER")
    parser.add_argument("--cache-dir", default="qasper/cache")
    parser.add_argument("--raw-dir", default="qasper/raw")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"])
    parser.add_argument("--embed-model", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE)
    parser.add_argument("--query-batch-size", type=int, default=DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument(
        "--chunk-words",
        type=int,
        default=DEFAULT_CHUNK_WORDS,
        help="Target chunk size in words. Sentence-aware chunks may be shorter or longer.",
    )
    parser.add_argument("--overlap-ratio", type=float, default=DEFAULT_OVERLAP_RATIO)
    parser.add_argument(
        "--query-mode",
        choices=["question", "title_question", "search_query_question"],
        default="title_question",
        help="Text sent to the retriever. The saved question field is always the original question.",
    )
    parser.add_argument(
        "--unanswerable-policy",
        choices=["all", "any"],
        default="all",
        help=(
            "'all' drops only questions where all annotations are unanswerable; "
            "'any' drops questions with any unanswerable annotation."
        ),
    )
    parser.add_argument("--write-500", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device(device)


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask[..., None].to(last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1e-6)
    return summed / denom


def download_qasper(raw_dir: Path) -> dict[str, Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archives = {
        "train_dev": raw_dir / "qasper-train-dev-v0.3.tgz",
        "test": raw_dir / "qasper-test-and-evaluator-v0.3.tgz",
    }
    urls = {"train_dev": TRAIN_DEV_URL, "test": TEST_URL}

    for key, path in archives.items():
        if path.exists() and path.stat().st_size > 0:
            continue
        print(f"Downloading {urls[key]} -> {path}")
        urllib.request.urlretrieve(urls[key], path)

    return archives


def load_raw_splits(raw_dir: Path) -> dict[str, dict[str, Any]]:
    archives = download_qasper(raw_dir)
    split_archives = {
        "train": archives["train_dev"],
        "dev": archives["train_dev"],
        "test": archives["test"],
    }

    datasets: dict[str, dict[str, Any]] = {}
    for split, archive_path in split_archives.items():
        with tarfile.open(archive_path, "r:gz") as tar:
            member = tar.getmember(ARCHIVE_FILES[split])
            with tar.extractfile(member) as f:
                if f is None:
                    raise FileNotFoundError(f"{ARCHIVE_FILES[split]} not found in {archive_path}")
                datasets[split] = json.load(f)
    return datasets


def stable_unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = normalize_space(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def normalize_space(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def split_words(text: str) -> list[str]:
    return normalize_space(text).split()


def split_sentences(text: str) -> list[str]:
    text = normalize_space(text)
    if not text:
        return []

    try:
        import spacy

        nlp = split_sentences._nlp
    except AttributeError:
        try:
            nlp = spacy.blank("en")
            nlp.add_pipe("sentencizer")
            split_sentences._nlp = nlp
        except Exception:
            split_sentences._nlp = None
            nlp = None
    except Exception:
        nlp = None

    if nlp is not None:
        doc = nlp(text)
        sentences = [normalize_space(sent.text) for sent in doc.sents if normalize_space(sent.text)]
        if sentences:
            return sentences

    # Fallback if spaCy is unavailable: split only after strong sentence-ending
    # punctuation followed by a likely new sentence.
    sentences = re.split(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])", text)
    return [sentence for sentence in sentences if sentence]


def safe_title(value: str) -> str:
    return normalize_space(value) or "Untitled paper"


def paper_blocks(paper: dict[str, Any]) -> list[tuple[str, str]]:
    title = safe_title(paper.get("title", ""))
    blocks: list[tuple[str, str]] = [("Title", title)]

    abstract = normalize_space(paper.get("abstract", ""))
    if abstract:
        blocks.append(("Abstract", f"Abstract {abstract}"))

    for section in paper.get("full_text", []) or []:
        section_name = normalize_space(section.get("section_name", "")) or "Body"
        for paragraph in section.get("paragraphs", []) or []:
            paragraph = normalize_space(paragraph)
            if paragraph:
                blocks.append((section_name, f"{section_name} {paragraph}"))
    return blocks


def paper_sentence_units(paper: dict[str, Any]) -> list[SentenceUnit]:
    units: list[SentenceUnit] = []
    cursor = 0
    for section_name, text in paper_blocks(paper):
        for sentence in split_sentences(text):
            sentence_words = split_words(sentence)
            if not sentence_words:
                continue
            units.append(
                SentenceUnit(
                    text=" ".join(sentence_words),
                    section_name=section_name,
                    word_start=cursor,
                    word_end=cursor + len(sentence_words),
                    num_words=len(sentence_words),
                )
            )
            cursor += len(sentence_words)
    return units


def should_add_sentence(current_words: int, sentence_words: int, target_words: int) -> bool:
    if current_words == 0:
        return True
    if current_words + sentence_words <= target_words:
        return True
    if current_words < int(target_words * 0.75):
        return True
    current_gap = abs(target_words - current_words)
    expanded_gap = abs(target_words - (current_words + sentence_words))
    return expanded_gap <= current_gap


def next_chunk_start(
    start_idx: int,
    end_idx: int,
    units: Sequence[SentenceUnit],
    overlap_words: int,
) -> int:
    overlap_start = end_idx
    overlap_count = 0
    while overlap_start > start_idx and overlap_count < overlap_words:
        overlap_start -= 1
        overlap_count += units[overlap_start].num_words

    if overlap_start <= start_idx:
        return min(start_idx + 1, end_idx)
    return overlap_start


def chunk_paper(
    paper_id: str,
    paper: dict[str, Any],
    split: str,
    chunk_words: int,
    overlap_words: int,
) -> list[Chunk]:
    stride = chunk_words - overlap_words
    if stride < 1:
        raise ValueError("chunk_words must be greater than overlap_words.")

    title = safe_title(paper.get("title", ""))
    units = paper_sentence_units(paper)
    if not units:
        return []

    chunks: list[Chunk] = []
    chunk_idx = 0
    start_idx = 0
    while start_idx < len(units):
        end_idx = start_idx
        current_words = 0
        while end_idx < len(units) and should_add_sentence(
            current_words,
            units[end_idx].num_words,
            chunk_words,
        ):
            current_words += units[end_idx].num_words
            end_idx += 1

        if end_idx == start_idx:
            end_idx += 1

        chunk_units = units[start_idx:end_idx]
        dominant_section = Counter(unit.section_name for unit in chunk_units).most_common(1)[0][0]
        chunk_title = f"{title} | {dominant_section}"
        chunks.append(
            Chunk(
                id=f"qasper:{paper_id}:{chunk_idx:05d}",
                paper_id=paper_id,
                title=chunk_title,
                text=" ".join(unit.text for unit in chunk_units),
                source_split=split,
                word_start=chunk_units[0].word_start,
                word_end=chunk_units[-1].word_end,
            )
        )
        chunk_idx += 1

        if end_idx >= len(units):
            break
        start_idx = next_chunk_start(start_idx, end_idx, units, overlap_words)

    return chunks


def answer_to_strings(answer: dict[str, Any]) -> list[str]:
    payload = answer.get("answer", {})
    if payload.get("unanswerable") is True:
        return []

    strings: list[str] = []
    yes_no = payload.get("yes_no")
    if yes_no is True:
        strings.append("yes")
    elif yes_no is False:
        strings.append("no")

    strings.extend(payload.get("extractive_spans") or [])
    free_form = payload.get("free_form_answer")
    if free_form:
        strings.append(free_form)
    return stable_unique(strings)


def answer_to_evidence(answer: dict[str, Any]) -> list[str]:
    payload = answer.get("answer", {})
    if payload.get("unanswerable") is True:
        return []
    return stable_unique((payload.get("highlighted_evidence") or []) + (payload.get("evidence") or []))


def prepare_questions(
    split: str,
    papers: dict[str, Any],
    query_mode: str,
    unanswerable_policy: str,
) -> tuple[list[PreparedQuestion], dict[str, int]]:
    prepared: list[PreparedQuestion] = []
    stats = {
        "raw_questions": 0,
        "kept_questions": 0,
        "dropped_questions": 0,
        "dropped_all_unanswerable": 0,
        "dropped_any_unanswerable": 0,
        "dropped_no_answer_text": 0,
        "stripped_unanswerable_annotations": 0,
    }

    for paper_id, paper in papers.items():
        title = safe_title(paper.get("title", ""))
        for qa in paper.get("qas", []) or []:
            stats["raw_questions"] += 1
            annotations = qa.get("answers", []) or []
            unanswerable_flags = [
                annotation.get("answer", {}).get("unanswerable") is True
                for annotation in annotations
            ]

            if annotations and all(unanswerable_flags):
                stats["dropped_questions"] += 1
                stats["dropped_all_unanswerable"] += 1
                continue

            if unanswerable_policy == "any" and any(unanswerable_flags):
                stats["dropped_questions"] += 1
                stats["dropped_any_unanswerable"] += 1
                continue

            kept_annotations = [
                annotation
                for annotation in annotations
                if annotation.get("answer", {}).get("unanswerable") is not True
            ]
            stats["stripped_unanswerable_annotations"] += len(annotations) - len(kept_annotations)

            answer_texts = stable_unique(
                text for annotation in kept_annotations for text in answer_to_strings(annotation)
            )
            evidence_texts = stable_unique(
                text for annotation in kept_annotations for text in answer_to_evidence(annotation)
            )
            if not answer_texts:
                stats["dropped_questions"] += 1
                stats["dropped_no_answer_text"] += 1
                continue

            question = normalize_space(qa.get("question", ""))
            retrieval_query = build_retrieval_query(question, title, qa, query_mode)
            answer_annotations = [
                {
                    "annotation_id": annotation.get("annotation_id"),
                    "worker_id": annotation.get("worker_id"),
                    "answer": annotation.get("answer"),
                }
                for annotation in kept_annotations
            ]

            record = {
                "_id": qa.get("question_id"),
                "id": paper_id,
                "paper_id": paper_id,
                "question_id": qa.get("question_id"),
                "title": title,
                "question": question,
                "answers": answer_texts,
                "answer": answer_texts[0],
                "answer_annotations": answer_annotations,
                "split": split,
                "retrieval_query": retrieval_query,
            }
            prepared.append(
                PreparedQuestion(
                    record=record,
                    retrieval_query=retrieval_query,
                    answer_texts=answer_texts,
                    evidence_texts=evidence_texts,
                )
            )

    stats["kept_questions"] = len(prepared)
    return prepared, stats


def build_retrieval_query(question: str, title: str, qa: dict[str, Any], mode: str) -> str:
    if mode == "question":
        return question
    if mode == "search_query_question":
        search_query = normalize_space(qa.get("search_query", ""))
        if search_query:
            return f"{search_query}\n{question}"
        return question
    if mode == "title_question":
        return f"{title}\n{question}"
    raise ValueError(f"Unsupported query mode: {mode}")


def corpus_fingerprint(chunks: Sequence[Chunk], args: argparse.Namespace) -> str:
    hasher = hashlib.sha1()
    payload = {
        "qasper_version": QASPER_VERSION,
        "embed_model": args.embed_model,
        "chunking_method": CHUNKING_METHOD,
        "chunk_words": args.chunk_words,
        "overlap_ratio": args.overlap_ratio,
        "top_k": args.top_k,
        "chunk_count": len(chunks),
    }
    hasher.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    for chunk in chunks:
        hasher.update(chunk.id.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(chunk.text.encode("utf-8"))
        hasher.update(b"\0")
    return hasher.hexdigest()


def cache_paths(cache_dir: Path, model_name: str, fingerprint: str) -> tuple[Path, Path, Path]:
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")
    base = cache_dir / f"{safe_model}_{fingerprint[:12]}"
    return base.with_suffix(".chunks.jsonl"), base.with_suffix(".embeddings.npy"), base.with_suffix(".meta.json")


def load_cached_index(chunks_path: Path, embeddings_path: Path, meta_path: Path, fingerprint: str) -> tuple[list[Chunk], np.ndarray] | None:
    if not (chunks_path.exists() and embeddings_path.exists() and meta_path.exists()):
        return None
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("fingerprint") != fingerprint:
        return None

    chunks: list[Chunk] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks, np.load(embeddings_path)


def save_index(
    chunks_path: Path,
    embeddings_path: Path,
    meta_path: Path,
    chunks: Sequence[Chunk],
    embeddings: np.ndarray,
    fingerprint: str,
    args: argparse.Namespace,
) -> None:
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")
    np.save(embeddings_path, embeddings)
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprint": fingerprint,
                "qasper_version": QASPER_VERSION,
                "embed_model": args.embed_model,
                "chunking_method": CHUNKING_METHOD,
                "chunk_words": args.chunk_words,
                "overlap_ratio": args.overlap_ratio,
                "overlap_words": int(round(args.chunk_words * args.overlap_ratio)),
                "top_k": args.top_k,
                "chunk_count": len(chunks),
                "chunk_word_stats": chunk_word_stats(chunks),
                "embedding_shape": list(embeddings.shape),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )


def build_corpus(raw_splits: dict[str, dict[str, Any]], args: argparse.Namespace) -> list[Chunk]:
    overlap_words = int(round(args.chunk_words * args.overlap_ratio))
    chunks: list[Chunk] = []
    seen_papers: set[str] = set()

    for split in ["train", "dev", "test"]:
        for paper_id, paper in raw_splits[split].items():
            if paper_id in seen_papers:
                continue
            seen_papers.add(paper_id)
            chunks.extend(chunk_paper(paper_id, paper, split, args.chunk_words, overlap_words))

    if len(chunks) < args.top_k:
        raise RuntimeError(f"Only {len(chunks)} chunks were built, less than top_k={args.top_k}.")
    return chunks


def chunk_word_stats(chunks: Sequence[Chunk]) -> dict[str, float | int]:
    lengths = np.array([len(chunk.text.split()) for chunk in chunks], dtype=np.float32)
    if lengths.size == 0:
        return {
            "min": 0,
            "p05": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0,
        }
    return {
        "min": int(lengths.min()),
        "p05": round(float(np.percentile(lengths, 5)), 2),
        "mean": round(float(lengths.mean()), 2),
        "median": round(float(np.median(lengths)), 2),
        "p95": round(float(np.percentile(lengths, 95)), 2),
        "max": int(lengths.max()),
    }


def load_or_build_index(
    raw_splits: dict[str, dict[str, Any]],
    embedder: ContrieverEmbedder,
    args: argparse.Namespace,
) -> tuple[list[Chunk], np.ndarray, dict[str, Any]]:
    chunks = build_corpus(raw_splits, args)
    fingerprint = corpus_fingerprint(chunks, args)
    chunks_path, embeddings_path, meta_path = cache_paths(Path(args.cache_dir), args.embed_model, fingerprint)

    cached = None if args.rebuild_index else load_cached_index(chunks_path, embeddings_path, meta_path, fingerprint)
    if cached is not None:
        cached_chunks, embeddings = cached
        print(f"Loaded cached chunk index from {embeddings_path}")
        return cached_chunks, embeddings, {
            "fingerprint": fingerprint,
            "chunks_path": str(chunks_path),
            "embeddings_path": str(embeddings_path),
            "cache_hit": True,
        }

    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.encode(texts, desc="Embedding QASPER chunks")
    save_index(chunks_path, embeddings_path, meta_path, chunks, embeddings, fingerprint, args)
    return chunks, embeddings, {
        "fingerprint": fingerprint,
        "chunks_path": str(chunks_path),
        "embeddings_path": str(embeddings_path),
        "cache_hit": False,
    }


def normalized_for_substring(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def has_gold_support(text: str, answers: Sequence[str], evidence_texts: Sequence[str]) -> bool:
    normalized_text = normalized_for_substring(text)
    for answer in answers:
        normalized_answer = normalized_for_substring(answer)
        if normalized_answer in {"", "yes", "no"}:
            continue
        if len(normalized_answer) >= 3 and normalized_answer in normalized_text:
            return True

    text_tokens = set(word_tokens(text))
    if not text_tokens:
        return False
    for evidence in evidence_texts:
        evidence_tokens = set(word_tokens(evidence))
        if len(evidence_tokens) < 8:
            continue
        overlap = len(text_tokens & evidence_tokens) / min(len(text_tokens), len(evidence_tokens))
        if overlap >= 0.50:
            return True
    return False


def retrieve_split(
    split: str,
    prepared: Sequence[PreparedQuestion],
    chunks: Sequence[Chunk],
    passage_embeddings: np.ndarray,
    embedder: ContrieverEmbedder,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not prepared:
        return []

    passage_tensor = torch.from_numpy(passage_embeddings).to(embedder.device)
    queries = [item.retrieval_query for item in prepared]
    records: list[dict[str, Any]] = []

    for start in tqdm(range(0, len(prepared), args.query_batch_size), desc=f"Retrieving {split}"):
        batch_items = list(prepared[start : start + args.query_batch_size])
        batch_queries = queries[start : start + args.query_batch_size]
        query_embeddings = embedder.encode(batch_queries, desc=f"Embedding {split} queries")
        query_tensor = torch.from_numpy(query_embeddings).to(embedder.device)
        with torch.inference_mode():
            scores = query_tensor @ passage_tensor.T
            top_scores, top_indices = torch.topk(scores, k=args.top_k, dim=1)
        top_scores_np = top_scores.detach().cpu().numpy()
        top_indices_np = top_indices.detach().cpu().numpy()

        for item, row_scores, row_indices in zip(batch_items, top_scores_np, top_indices_np):
            ctxs = []
            for score, chunk_idx in zip(row_scores, row_indices):
                chunk = chunks[int(chunk_idx)]
                ctxs.append(
                    {
                        "id": chunk.id,
                        "title": chunk.title,
                        "text": chunk.text,
                        "score": f"{float(score):.7f}",
                        "hasanswer": has_gold_support(chunk.text, item.answer_texts, item.evidence_texts),
                    }
                )
            record = dict(item.record)
            record["ctxs"] = ctxs
            records.append(record)

    return records


def write_jsonl(path: Path, records: Sequence[dict[str, Any]], overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_metadata(path: Path, metadata: dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_splits = load_raw_splits(Path(args.raw_dir))
    embedder = ContrieverEmbedder(
        model_name=args.embed_model,
        device=args.device,
        batch_size=args.embed_batch_size,
        max_length=args.max_length,
    )
    chunks, passage_embeddings, index_meta = load_or_build_index(raw_splits, embedder, args)

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "qasper_version": QASPER_VERSION,
        "source_urls": {"train_dev": TRAIN_DEV_URL, "test": TEST_URL},
        "output_dir": str(output_dir),
        "embed_model": args.embed_model,
        "retriever": "contriever-msmarco dot product over mean-pooled embeddings",
        "chunking_method": CHUNKING_METHOD,
        "target_chunk_words": args.chunk_words,
        "overlap_ratio": args.overlap_ratio,
        "overlap_words": int(round(args.chunk_words * args.overlap_ratio)),
        "overlap_strategy": "whole trailing sentences totaling at least overlap_words when possible",
        "top_k": args.top_k,
        "query_mode": args.query_mode,
        "unanswerable_policy": args.unanswerable_policy,
        "corpus": {
            "paper_splits": ["train", "dev", "test"],
            "paper_count": sum(len(raw_splits[split]) for split in ["train", "dev", "test"]),
            "chunk_count": len(chunks),
            "chunk_word_stats": chunk_word_stats(chunks),
            "index": index_meta,
        },
        "splits": {},
    }

    for split in args.splits:
        prepared, prep_stats = prepare_questions(
            split=split,
            papers=raw_splits[split],
            query_mode=args.query_mode,
            unanswerable_policy=args.unanswerable_policy,
        )
        records = retrieve_split(split, prepared, chunks, passage_embeddings, embedder, args)
        output_path = output_dir / f"{split}.json"
        write_jsonl(output_path, records, args.overwrite)

        split_meta = dict(prep_stats)
        split_meta["output_path"] = str(output_path)
        split_meta["records_written"] = len(records)

        if args.write_500 and len(records) >= 500:
            sample_path = output_dir / f"{split}_500.json"
            write_jsonl(sample_path, records[:500], args.overwrite)
            split_meta["sample_500_output_path"] = str(sample_path)
            split_meta["sample_500_records_written"] = 500

        metadata["splits"][split] = split_meta

    write_metadata(output_dir / "metadata.json", metadata, args.overwrite)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
