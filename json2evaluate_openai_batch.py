#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils import evaluate_batch, print_evaluation_results

DEFAULT_ENDPOINT = "/v1/responses"
DEFAULT_MODEL = "gpt-5-mini"
DEFAULT_MAX_OUTPUT_TOKENS = 1024
DEFAULT_POLL_INTERVAL = 30
DEFAULT_COMPLETION_WINDOW = "24h"


_YES_NO_RE = re.compile(
    r"^\s*(?:[\"\u201c\u201d'\[])?\s*(yes|no)\b",
    flags=re.IGNORECASE,
)


def canonicalize_text(text: str) -> str:
    """Normalize common Unicode punctuation differences that hurt EM/F1.

    This is intentionally *only* used in this OpenAI batch evaluation script,
    so we don't change the evaluation behavior of other pipelines that rely on
    utils.normalize_answer().
    """

    if not isinstance(text, str) or not text:
        return "" if text is None else str(text)

    # Normalize compatibility forms (e.g., full-width punctuation).
    text = unicodedata.normalize("NFKC", text)

    # Map smart quotes / apostrophes / dashes to ASCII equivalents.
    translation_table = {
        ord("\u2018"): "'",  # left single quote
        ord("\u2019"): "'",  # right single quote
        ord("\u201b"): "'",  # single high-reversed-9 quote
        ord("\u2032"): "'",  # prime
        ord("\u201c"): '"',  # left double quote
        ord("\u201d"): '"',  # right double quote
        ord("\u201f"): '"',  # double high-reversed-9 quote
        ord("\u2033"): '"',  # double prime
        ord("\u2013"): "-",  # en dash
        ord("\u2014"): "-",  # em dash
        ord("\u2212"): "-",  # minus
        ord("\u00a0"): " ",  # no-break space
    }
    text = text.translate(translation_table)

    # Strip wrapping quotes that some models add.
    stripped = text.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {'"', "'"}:
        stripped = stripped[1:-1].strip()

    # Normalize whitespace.
    stripped = re.sub(r"\s+", " ", stripped)
    return stripped


def _is_yes_no_string(value: str) -> bool:
    v = canonicalize_text(value).strip().lower()
    return v in {"yes", "no"}


def is_yes_no_ground_truth(gt: Any) -> bool:
    if isinstance(gt, str):
        return _is_yes_no_string(gt)
    if isinstance(gt, list):
        strings = [x for x in gt if isinstance(x, str)]
        if not strings:
            return False
        normalized = {canonicalize_text(x).strip().lower() for x in strings}
        normalized.discard("")
        return bool(normalized) and normalized.issubset({"yes", "no"})
    return False


def extract_yes_no_answer(prediction: str) -> Optional[str]:
    """Extract a yes/no answer from a model response.

    Tries a strict leading extraction first ("Yes. ..."), then a conservative
    near-start search to catch patterns like "... ? No — ...".
    """

    text = canonicalize_text(prediction)
    if not text:
        return None

    m = _YES_NO_RE.match(text)
    if m:
        return m.group(1).lower()

    # Conservative: search for yes/no very early (often after repeating the question).
    # This is only applied when the ground-truth is yes/no.
    prefix = text[:96]
    for m2 in re.finditer(r"\b(yes|no)\b", prefix, flags=re.IGNORECASE):
        return m2.group(1).lower()

    return None


def canonicalize_ground_truth(gt: Any) -> Any:
    if isinstance(gt, str):
        return canonicalize_text(gt)
    if isinstance(gt, list):
        out: List[Any] = []
        for x in gt:
            out.append(canonicalize_text(x) if isinstance(x, str) else x)
        return out
    return gt


def load_api_key(path: str) -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    key_path = Path(path)
    if key_path.is_file():
        key = key_path.read_text().strip()
        if key:
            return key
    raise FileNotFoundError(
        "OpenAI API key not found. Set OPENAI_API_KEY or put it in cache/openai_key.txt."
    )


def build_request_body(
    prompt: str,
    endpoint: str,
    model: str,
    max_output_tokens: int,
) -> Dict[str, Any]:
    if endpoint == "/v1/chat/completions":
        return {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_output_tokens,
        }

    if endpoint == "/v1/responses":
        body: Dict[str, Any] = {
            "model": model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if "gpt-5" in model.lower():
            body["reasoning"] = {"effort": "minimal"}
            body["text"] = {"verbosity": "low"}
        return body

    raise ValueError(f"Unsupported endpoint: {endpoint}")


def build_batch_requests(
    prompts: List[str],
    endpoint: str,
    model: str,
    max_output_tokens: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompts):
        rows.append(
            {
                "custom_id": str(idx),
                "method": "POST",
                "url": endpoint,
                "body": build_request_body(
                    prompt=prompt,
                    endpoint=endpoint,
                    model=model,
                    max_output_tokens=max_output_tokens,
                ),
            }
        )
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            json.dump(row, f, ensure_ascii=True)
            f.write("\n")


def upload_input_file(api_key: str, path: Path) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    with path.open("rb") as f:
        files = {"file": (path.name, f, "application/jsonl")}
        data = {"purpose": "batch"}
        response = requests.post(
            "https://api.openai.com/v1/files",
            headers=headers,
            files=files,
            data=data,
            timeout=(30, 600),
        )
    response.raise_for_status()
    payload = response.json()
    file_id = payload.get("id")
    if not file_id:
        raise RuntimeError(f"Unexpected file upload response: {payload}")
    return file_id


def _response_error_payload(response: requests.Response) -> Any:
    try:
        return response.json()
    except Exception:
        return {"raw": response.text}


def _raise_for_status_with_payload(response: requests.Response, context: str) -> None:
    if response.status_code < 400:
        return
    payload = _response_error_payload(response)
    raise RuntimeError(f"{context} ({response.status_code}): {payload}")


def get_file(api_key: str, file_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        f"https://api.openai.com/v1/files/{file_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def wait_for_file_processed(
    api_key: str,
    file_id: str,
    poll_interval: float = 2.0,
    timeout_s: float = 300.0,
) -> Dict[str, Any]:
    start = time.time()
    while True:
        file_obj = get_file(api_key, file_id)
        status = file_obj.get("status")
        if status == "processed":
            return file_obj
        if status in {"error", "failed"}:
            raise RuntimeError(f"File {file_id} failed to process: {file_obj}")
        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for file {file_id} to be processed (last status: {status})"
            )
        time.sleep(poll_interval)


def build_metadata(source: str, model: str, offset: int, limit: Optional[int]) -> Dict[str, str]:
    meta: Dict[str, str] = {"source": source, "model": model, "offset": str(offset)}
    if limit is not None:
        meta["limit"] = str(limit)
    return meta


def create_batch(
    api_key: str,
    input_file_id: str,
    endpoint: str,
    completion_window: str,
    metadata: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "input_file_id": input_file_id,
        "endpoint": endpoint,
        "completion_window": completion_window,
    }
    if metadata:
        payload["metadata"] = metadata
    response = requests.post(
        "https://api.openai.com/v1/batches",
        headers=headers,
        json=payload,
        timeout=60,
    )
    _raise_for_status_with_payload(response, "Failed to create batch")
    return response.json()


def get_batch(api_key: str, batch_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        f"https://api.openai.com/v1/batches/{batch_id}",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def wait_for_batch(api_key: str, batch_id: str, poll_interval: int) -> Dict[str, Any]:
    while True:
        batch = get_batch(api_key, batch_id)
        status = batch.get("status", "unknown")
        request_counts = batch.get("request_counts") or {}
        total = request_counts.get("total") or batch.get("total_requests") or 0
        completed = request_counts.get("completed") or batch.get("completed_requests") or 0
        failed = request_counts.get("failed") or batch.get("failed_requests") or 0
        if failed:
            print(f"Batch status: {status} ({completed}/{total}, failed={failed})")
        else:
            print(f"Batch status: {status} ({completed}/{total})")
        if status in {"completed", "failed", "cancelled", "expired"}:
            return batch
        time.sleep(poll_interval)


def download_file(api_key: str, file_id: str, output_path: Path) -> None:
    headers = {"Authorization": f"Bearer {api_key}"}
    response = requests.get(
        f"https://api.openai.com/v1/files/{file_id}/content",
        headers=headers,
        stream=True,
        timeout=(30, 600),
    )
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def preflight_request(
    api_key: str,
    endpoint: str,
    prompt: str,
    model: str,
    max_output_tokens: int,
) -> None:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = build_request_body(
        prompt=prompt,
        endpoint=endpoint,
        model=model,
        max_output_tokens=max_output_tokens,
    )
    response = requests.post(
        f"https://api.openai.com{endpoint}",
        headers=headers,
        json=body,
        timeout=60,
    )
    if response.status_code >= 400:
        payload = _response_error_payload(response)
        message = None
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                message = err.get("message")
        if isinstance(message, str) and "model output limit was reached" in message:
            raise RuntimeError(
                "Preflight failed because the model hit the output token limit. "
                f"Increase `--max-output-tokens` (currently {max_output_tokens}) and retry. "
                f"Raw error: {payload}"
            )
        raise RuntimeError(
            f"Preflight request failed ({response.status_code}) for {endpoint} with model '{model}': {payload}"
        )


def extract_text_from_body(body: Dict[str, Any], endpoint: str) -> str:
    if endpoint == "/v1/chat/completions":
        choices = body.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        return content if isinstance(content, str) else ""

    if endpoint == "/v1/responses":
        output_text = body.get("output_text")
        if isinstance(output_text, str):
            return output_text
        output_items = body.get("output") or []
        for item in output_items:
            content_items = item.get("content") or []
            for part in content_items:
                if part.get("type") in {"output_text", "text"}:
                    text = part.get("text")
                    if isinstance(text, str):
                        return text
        return ""

    return ""


def parse_batch_output(
    path: Path,
    expected_count: int,
    endpoint: str,
    error_path: Optional[Path] = None,
) -> Tuple[List[str], List[bool], Dict[str, int]]:
    predictions = [""] * expected_count
    valid = [False] * expected_count
    seen = [False] * expected_count
    stats: Dict[str, int] = {
        "non_200": 0,
        "empty_valid": 0,
        "incomplete": 0,
        "incomplete_max_output_tokens": 0,
        "incomplete_content_filter": 0,
        "missing_records": 0,
    }

    def process_record(record: Dict[str, Any]) -> None:
        custom_id = record.get("custom_id")
        try:
            idx = int(custom_id)
        except (TypeError, ValueError):
            return
        if idx >= expected_count or idx < 0:
            return

        seen[idx] = True
        error = record.get("error")
        response = record.get("response") or {}
        status_code = response.get("status_code")
        if error or (isinstance(status_code, int) and status_code != 200):
            stats["non_200"] += 1
            predictions[idx] = ""
            return

        body = response.get("body") or {}
        valid[idx] = True

        if endpoint == "/v1/responses":
            body_status = body.get("status")
            if body_status == "incomplete":
                stats["incomplete"] += 1
                incomplete_details = body.get("incomplete_details") or {}
                reason = incomplete_details.get("reason")
                if reason == "max_output_tokens":
                    stats["incomplete_max_output_tokens"] += 1
                elif reason == "content_filter":
                    stats["incomplete_content_filter"] += 1

        text = extract_text_from_body(body, endpoint)
        predictions[idx] = text.strip() if isinstance(text, str) else ""

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            process_record(json.loads(line))

    if error_path and error_path.is_file():
        with error_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                process_record(json.loads(line))

    stats["empty_valid"] = sum(
        1 for i in range(expected_count) if valid[i] and predictions[i] == ""
    )
    stats["missing_records"] = sum(1 for i in range(expected_count) if not seen[i])
    return predictions, valid, stats


def evaluate_predictions(
    predictions: List[str],
    ground_truths: List[Any],
    valid: List[bool],
    stats: Dict[str, int],
) -> Dict[str, float]:
    # Only exclude requests that failed at the HTTP level (non-200). Empty strings still count.
    # Additionally, apply OpenAI-only post-processing to mitigate:
    # - "Yes. <explanation>" / "No. <explanation>" for yes/no questions
    # - Unicode punctuation differences (smart quotes / long dashes)
    predictions_eval: List[str] = []
    ground_truths_eval: List[Any] = []
    yes_no_extracted = 0

    for pred, gt, ok in zip(predictions, ground_truths, valid):
        if not ok:
            continue

        pred_c = canonicalize_text(pred)
        gt_c = canonicalize_ground_truth(gt)

        if is_yes_no_ground_truth(gt_c):
            yn = extract_yes_no_answer(pred_c)
            if yn is not None:
                pred_c = yn
                yes_no_extracted += 1

        predictions_eval.append(pred_c)
        ground_truths_eval.append(gt_c)

    stats["yes_no_extracted"] = yes_no_extracted

    if not predictions_eval:
        return {
            "count": 0,
            "exact_match": 0,
            "f1": 0,
            "empty_valid": stats.get("empty_valid", 0),
            "invalid_responses": stats.get("non_200", 0),
            "missing_records": stats.get("missing_records", 0),
            "incomplete": stats.get("incomplete", 0),
            "incomplete_max_output_tokens": stats.get("incomplete_max_output_tokens", 0),
            "incomplete_content_filter": stats.get("incomplete_content_filter", 0),
            "exact_match_percentage": 0,
            "f1_percentage": 0,
        }

    batch_results = evaluate_batch(predictions_eval, ground_truths_eval)
    total_count = batch_results["count"]
    total_em = batch_results["exact_match"]
    total_f1 = batch_results["f1"]
    return {
        "count": total_count,
        "exact_match": total_em,
        "f1": total_f1,
        "empty_valid": stats.get("empty_valid", 0),
        "invalid_responses": stats.get("non_200", 0),
        "missing_records": stats.get("missing_records", 0),
        "incomplete": stats.get("incomplete", 0),
        "incomplete_max_output_tokens": stats.get("incomplete_max_output_tokens", 0),
        "incomplete_content_filter": stats.get("incomplete_content_filter", 0),
        "exact_match_percentage": 100.0 * total_em / total_count if total_count else 0,
        "f1_percentage": 100.0 * total_f1 / total_count if total_count else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate compressed contexts using OpenAI Batch API."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input JSON file with compressed data.",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=DEFAULT_MODEL,
        help="OpenAI model name.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        choices=["/v1/chat/completions", "/v1/responses"],
        help="OpenAI endpoint used for batch requests.",
    )
    parser.add_argument(
        "--api-key-file",
        type=str,
        default="cache/openai_key.txt",
        help="Path to file containing the OpenAI API key.",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Existing batch ID to resume/download results.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Submit the batch and exit without waiting for completion.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=DEFAULT_POLL_INTERVAL,
        help="Seconds between batch status checks.",
    )
    parser.add_argument(
        "--completion-window",
        type=str,
        default=DEFAULT_COMPLETION_WINDOW,
        help="Batch completion window (e.g. 24h).",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help="Max output tokens for each response.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start index within the dataset (0-based).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N items (after --offset).",
    )
    parser.add_argument(
        "--no-preflight",
        action="store_true",
        help="Skip the single-request preflight check before creating a batch.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="cache",
        help="Directory to store batch input/output files.",
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        default=None,
        help="Optional path to write the batch input JSONL file.",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional path to write the batch output JSONL file.",
    )

    parser.add_argument(
        "--existing-output-jsonl",
        type=str,
        default=None,
        help=(
            "Evaluate using an already-downloaded batch output JSONL file (no OpenAI API calls). "
            "Useful to recompute metrics without spending credits."
        ),
    )
    parser.add_argument(
        "--existing-error-jsonl",
        type=str,
        default=None,
        help="Optional existing error JSONL file to merge while parsing (offline mode).",
    )

    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    metadata = data["metadata"]
    compressed_data = data["data"]

    prompts_all = [item["prompt_api"] for item in compressed_data]
    ground_truths_all = [item["ground_truth"] for item in compressed_data]

    start = max(0, int(args.offset))
    end = len(prompts_all)
    if args.limit is not None:
        end = min(end, start + max(0, int(args.limit)))
    prompts = prompts_all[start:end]
    ground_truths = ground_truths_all[start:end]
    if start or args.limit is not None:
        print(f"Evaluating subset: {start}:{end} ({len(prompts)} items)")

    print(f"Loaded compressed data from: {args.input}")
    print(f"Method: {metadata['method']}")
    print(f"K: {metadata['k']}")
    print(f"Total questions: {metadata['total_questions']}")
    print(f"Compression ratio: {metadata['compression_ratio']:.3f}")

    output_dir = Path(args.output_dir)
    endpoint = args.endpoint or DEFAULT_ENDPOINT
    print(f"Using endpoint: {endpoint}")

    # Offline mode: user supplies output JSONL (and optionally error JSONL) directly.
    existing_output_path = Path(args.existing_output_jsonl) if args.existing_output_jsonl else None
    existing_error_path = Path(args.existing_error_jsonl) if args.existing_error_jsonl else None
    if existing_output_path is not None:
        if not existing_output_path.is_file():
            raise FileNotFoundError(f"--existing-output-jsonl not found: {existing_output_path}")

        predictions, valid, stats = parse_batch_output(
            existing_output_path,
            len(prompts),
            endpoint,
            error_path=existing_error_path,
        )
        results = evaluate_predictions(predictions, ground_truths, valid, stats)

        title = f"EVALUATION RESULTS for '{metadata['method'].upper()}' (k={metadata['k']})"
        print_evaluation_results(results, title)

        print("\nRESPONSE STATISTICS:")
        print(f"  Valid (HTTP 200): {results['count']}")
        print(f"  Empty responses (counted): {results['empty_valid']}")
        if results.get("yes_no_extracted"):
            print(f"  Yes/No extracted for eval: {results['yes_no_extracted']}")
        if results.get("invalid_responses"):
            print(f"  Invalid responses (non-200): {results['invalid_responses']}")
        if results.get("missing_records"):
            print(
                "  Missing records (not in output/error files): "
                f"{results['missing_records']}"
            )
        if endpoint == "/v1/responses":
            print(f"  Incomplete responses: {results.get('incomplete', 0)}")
            if results.get("incomplete_max_output_tokens"):
                print(
                    f"    - due to max_output_tokens: {results['incomplete_max_output_tokens']}"
                )
            if results.get("incomplete_content_filter"):
                print(
                    f"    - due to content_filter: {results['incomplete_content_filter']}"
                )

        print("\nCOMPRESSION STATISTICS:")
        print(f"  Compression Ratio: {metadata['compression_ratio']:.3f}")
        print(f"  Original Tokens: {metadata['total_original_tokens']:,}")
        print(f"  Compressed Tokens: {metadata['total_compressed_tokens']:,}")
        print("=" * 80)
        return

    # Online mode below: requires an API key.
    api_key = load_api_key(args.api_key_file)

    batch_id = args.batch_id
    if not batch_id:
        if not args.no_preflight and prompts:
            print(f"Preflight: testing {endpoint} with model '{args.model}' ...")
            preflight_request(
                api_key=api_key,
                endpoint=endpoint,
                prompt=prompts[0],
                model=args.model,
                max_output_tokens=args.max_output_tokens,
            )
            print("Preflight: OK")

        input_jsonl_path = (
            Path(args.input_jsonl)
            if args.input_jsonl
            else output_dir / f"openai_batch_input_{int(time.time())}.jsonl"
        )
        write_jsonl(
            input_jsonl_path,
            build_batch_requests(
                prompts=prompts,
                endpoint=endpoint,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
            ),
        )
        print(f"Prepared batch input file: {input_jsonl_path}")

        input_file_id = upload_input_file(api_key, input_jsonl_path)
        print(f"Uploaded input file: {input_file_id}")
        print("Waiting for input file to be processed...")
        wait_for_file_processed(api_key, input_file_id)
        print("Input file processed.")

        batch = create_batch(
            api_key=api_key,
            input_file_id=input_file_id,
            endpoint=endpoint,
            completion_window=args.completion_window,
            metadata=build_metadata(
                source=Path(args.input).name,
                model=args.model,
                offset=start,
                limit=args.limit,
            ),
        )
        batch_id = batch.get("id")
        if not batch_id:
            raise RuntimeError(f"Unexpected batch response: {batch}")
        print(f"Submitted batch: {batch_id}")

    if args.batch_id:
        batch_info = get_batch(api_key, batch_id)
        batch_endpoint = batch_info.get("endpoint")
        if batch_endpoint and batch_endpoint != endpoint:
            if args.endpoint is not None:
                raise RuntimeError(
                    f"Batch {batch_id} uses endpoint '{batch_endpoint}', but you requested '{endpoint}'."
                )
            print(f"Overriding endpoint from batch: {batch_endpoint}")
            endpoint = batch_endpoint

        meta = batch_info.get("metadata") or {}
        expected_source = meta.get("source")
        if expected_source and expected_source != Path(args.input).name:
            raise RuntimeError(
                f"Batch {batch_id} was created for '{expected_source}', but you provided '{Path(args.input).name}'."
            )
        request_counts = batch_info.get("request_counts") or {}
        expected_total = request_counts.get("total") or batch_info.get("total_requests")
        if expected_total and int(expected_total) != len(prompts):
            raise RuntimeError(
                f"Batch {batch_id} expects {expected_total} requests, but this run prepared {len(prompts)}."
            )

    if args.no_wait:
        batch = get_batch(api_key, batch_id)
        print(f"Batch status: {batch.get('status', 'unknown')}")
        return

    batch = wait_for_batch(api_key, batch_id, args.poll_interval)
    status = batch.get("status")

    error_file_id = batch.get("error_file_id")
    error_path: Optional[Path] = None
    if error_file_id:
        error_path = output_dir / f"openai_batch_error_{batch_id}.jsonl"
        download_file(api_key, error_file_id, error_path)
        print(f"Downloaded batch error file: {error_path}")

    if status != "completed":
        if error_path:
            raise RuntimeError(
                f"Batch {batch_id} did not complete successfully: {status}. See: {error_path}"
            )
        raise RuntimeError(f"Batch {batch_id} did not complete successfully: {status}")

    output_file_id = batch.get("output_file_id")
    if not output_file_id:
        if error_path:
            raise RuntimeError(
                f"Batch completed without an output file. See error file: {error_path}"
            )
        raise RuntimeError(
            "Batch completed without an output file and no error file ID was provided. "
            f"Batch ID: {batch_id}"
        )

    output_jsonl_path = (
        Path(args.output_jsonl)
        if args.output_jsonl
        else output_dir / f"openai_batch_output_{batch_id}.jsonl"
    )
    download_file(api_key, output_file_id, output_jsonl_path)
    print(f"Downloaded batch output: {output_jsonl_path}")

    predictions, valid, stats = parse_batch_output(
        output_jsonl_path, len(prompts), endpoint, error_path=error_path
    )
    results = evaluate_predictions(predictions, ground_truths, valid, stats)

    title = f"EVALUATION RESULTS for '{metadata['method'].upper()}' (k={metadata['k']})"
    print_evaluation_results(results, title)

    print("\nRESPONSE STATISTICS:")
    print(f"  Valid (HTTP 200): {results['count']}")
    print(f"  Empty responses (counted): {results['empty_valid']}")
    if results.get("yes_no_extracted"):
        print(f"  Yes/No extracted for eval: {results['yes_no_extracted']}")
    if results.get("invalid_responses"):
        print(f"  Invalid responses (non-200): {results['invalid_responses']}")
    if results.get("missing_records"):
        print(
            "  Missing records (not in output/error files): "
            f"{results['missing_records']}"
        )
    if endpoint == "/v1/responses":
        print(f"  Incomplete responses: {results.get('incomplete', 0)}")
        if results.get("incomplete_max_output_tokens"):
            print(
                f"    - due to max_output_tokens: {results['incomplete_max_output_tokens']}"
            )
        if results.get("incomplete_content_filter"):
            print(
                f"    - due to content_filter: {results['incomplete_content_filter']}"
            )

    print("\nCOMPRESSION STATISTICS:")
    print(f"  Compression Ratio: {metadata['compression_ratio']:.3f}")
    print(f"  Original Tokens: {metadata['total_original_tokens']:,}")
    print(f"  Compressed Tokens: {metadata['total_compressed_tokens']:,}")
    print("=" * 80)


if __name__ == "__main__":
    main()
