#!/usr/bin/env python3
import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Sequence


DEFAULT_BATCH_SIZES = (4, 8)
DEFAULT_READER_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def describe_evaluator(args: argparse.Namespace) -> str:
    if args.api:
        if args.api_base_url:
            return f"OpenAI-compatible API model `{args.api}` @ `{args.api_base_url}`"
        return f"Default API routing model `{args.api}`"
    return f"Local model `{args.reader_model_name}`"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run json2evaluate over every JSON file directly inside an input "
            "directory, using equal values for batch_size and "
            "reader_batch_size on each run, then write one report per "
            "experiment under reports/."
        )
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing experiment JSON files directly under it.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("reports"),
        help="Directory where per-experiment reports will be written.",
    )
    parser.add_argument(
        "--batch-sizes",
        "-b",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help=(
            "Batch sizes to evaluate. For each run, the script sets "
            "batch_size = reader_batch_size = that value."
        ),
    )
    parser.add_argument(
        "--reader_model_name",
        "-rm",
        type=str,
        default=DEFAULT_READER_MODEL,
        help="Hugging Face model name for the reader.",
    )
    parser.add_argument(
        "--auto_dtype",
        "-ad",
        action="store_true",
        help='Use torch_dtype="auto" for reader model loading.',
    )
    parser.add_argument(
        "--api",
        "-api",
        type=str,
        default=None,
        help="API model name. If set, uses API instead of a local reader.",
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL, such as http://host:8004/v1 for a vLLM server.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Optional API key for --api-base-url. If omitted, uses OPENAI_API_KEY or VLLM_API_KEY when available.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.api is None and not args.reader_model_name:
        raise ValueError("Either --reader_model_name or --api must be provided.")

    if not args.input_dir.is_dir():
        raise ValueError(f"Input directory does not exist or is not a directory: {args.input_dir}")

    if not args.batch_sizes:
        raise ValueError("Provide at least one batch size with --batch-sizes.")

    if args.api is None and (args.api_base_url is not None or args.api_key is not None):
        raise ValueError("--api-base-url and --api-key require --api.")

    invalid_batch_sizes = [size for size in args.batch_sizes if size <= 0]
    if invalid_batch_sizes:
        raise ValueError(f"Batch sizes must be positive integers: {invalid_batch_sizes}")


def list_json_files(input_dir: Path) -> List[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".json"
    )


def load_experiment(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "metadata" not in data or "data" not in data:
        raise ValueError(f"{path} must contain top-level 'metadata' and 'data' keys.")

    if not isinstance(data["data"], list):
        raise ValueError(f"{path} has invalid 'data'; expected a list.")

    return data


def compute_average_metrics(results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    metric_keys = [
        "count",
        "skipped",
        "failed_requests",
        "empty_valid_answers",
        "exact_match",
        "f1",
        "exact_match_percentage",
        "f1_percentage",
    ]
    return {key: mean(float(result.get(key, 0)) for result in results) for key in metric_keys}


def format_metric_block(results: Dict[str, Any]) -> List[str]:
    lines = [
        f"Questions evaluated: {int(round(results['count']))}",
        f"Exact Match: {results['exact_match']:.2f} ({results['exact_match_percentage']:.2f}%)",
        f"F1: {results['f1']:.2f} ({results['f1_percentage']:.2f}%)",
    ]
    if "failed_requests" in results:
        lines.insert(1, f"Request failures (excluded): {results['failed_requests']:.2f}")
    else:
        lines.insert(1, f"Excluded from scoring: {results['skipped']:.2f}")
    if "empty_valid_answers" in results:
        lines.insert(2, f"Empty valid answers (counted): {results['empty_valid_answers']:.2f}")
    return lines


def build_report(
    experiment_path: Path,
    metadata: Dict[str, Any],
    batch_sizes: Sequence[int],
    per_batch_results: Sequence[Dict[str, Any]],
    average_metrics: Dict[str, float],
    args: argparse.Namespace,
) -> str:
    lines: List[str] = []
    lines.append(f"Experiment report: {experiment_path.name}")
    lines.append("=" * 80)
    lines.append(f"Generated at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Input file: {experiment_path}")
    lines.append(f"Method: {metadata.get('method', 'unknown')}")
    lines.append(f"K: {metadata.get('k', 'unknown')}")
    lines.append(f"Total questions (metadata): {metadata.get('total_questions', 'unknown')}")

    compression_ratio = metadata.get("compression_ratio")
    if isinstance(compression_ratio, (int, float)):
        lines.append(f"Compression ratio: {compression_ratio:.6f}")
    else:
        lines.append(f"Compression ratio: {compression_ratio}")

    lines.append(f"Original tokens: {metadata.get('total_original_tokens', 'unknown')}")
    lines.append(f"Compressed tokens: {metadata.get('total_compressed_tokens', 'unknown')}")
    lines.append("Evaluator: " + describe_evaluator(args))
    lines.append(
        "Batch size pairs tested: "
        + ", ".join(
            f"(batch_size={size}, reader_batch_size={size})"
            for size in batch_sizes
        )
    )
    lines.append("")
    lines.append("Per-batch results")
    lines.append("-" * 80)

    for batch_size, results in zip(batch_sizes, per_batch_results):
        lines.append(f"batch_size={batch_size}, reader_batch_size={batch_size}")
        lines.extend(format_metric_block(results))
        lines.append("")

    lines.append("Average across all batch sizes")
    lines.append("-" * 80)
    lines.extend(format_metric_block(average_metrics))
    lines.append("")
    return "\n".join(lines)


def write_error_report(
    report_path: Path,
    experiment_path: Path,
    args: argparse.Namespace,
    exc: BaseException,
) -> None:
    lines = [
        f"Experiment report: {experiment_path.name}",
        "=" * 80,
        f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"Input file: {experiment_path}",
        "Status: FAILED",
        "Evaluator: " + describe_evaluator(args),
        "",
        f"Error type: {type(exc).__name__}",
        f"Error message: {exc}",
        "",
        "Traceback",
        "-" * 80,
        traceback.format_exc().rstrip(),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()

    try:
        validate_args(args)
    except ValueError as exc:
        print(f"Argument error: {exc}", file=sys.stderr)
        return 2

    json_files = list_json_files(args.input_dir)
    if not json_files:
        print(
            f"No JSON files found directly under {args.input_dir}. Nothing to evaluate.",
            file=sys.stderr,
        )
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)

    from json2evaluate import EvaluationPipeline

    initial_batch_size = args.batch_sizes[0]
    pipeline = EvaluationPipeline(
        reader_model_name=args.reader_model_name,
        reader_batch_size=initial_batch_size,
        use_auto_dtype=args.auto_dtype,
        api_model=args.api,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
    )

    failures = 0

    for experiment_path in json_files:
        report_path = args.output_dir / f"{experiment_path.stem}.report.txt"
        print(f"\nEvaluating {experiment_path.name} -> {report_path}")

        try:
            experiment = load_experiment(experiment_path)
            metadata = experiment["metadata"]
            compressed_data = experiment["data"]

            per_batch_results: List[Dict[str, Any]] = []
            for batch_size in args.batch_sizes:
                print(f"  Running batch_size={batch_size}, reader_batch_size={batch_size}")
                pipeline.reader_batch_size = batch_size
                results = pipeline.evaluate_compressed_data(
                    compressed_data,
                    batch_size=batch_size,
                )
                per_batch_results.append(results)

            average_metrics = compute_average_metrics(per_batch_results)
            report_text = build_report(
                experiment_path=experiment_path,
                metadata=metadata,
                batch_sizes=args.batch_sizes,
                per_batch_results=per_batch_results,
                average_metrics=average_metrics,
                args=args,
            )
            report_path.write_text(report_text, encoding="utf-8")
            print(f"  Wrote report: {report_path}")
        except Exception as exc:
            failures += 1
            write_error_report(report_path, experiment_path, args, exc)
            print(f"  Failed: {exc}", file=sys.stderr)
            print(f"  Wrote failure report: {report_path}", file=sys.stderr)

    if failures:
        print(f"\nCompleted with {failures} failure(s). Check reports in {args.output_dir}.")
        return 1

    print(f"\nCompleted successfully. Reports written to {args.output_dir}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
