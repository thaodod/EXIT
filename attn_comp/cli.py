import argparse
import json

from .compressor import AttnCompCompressor


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone AttnComp inference CLI")
    parser.add_argument("--model-path", required=True, help="Path or Hugging Face id for Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--question", required=True, help="Question/query used to score the chunks")
    parser.add_argument("--chunks-file", required=True, help="JSON file containing a list of chunks")
    parser.add_argument("--checkpoint", default=None, help="Optional override for the bundled attention checkpoint")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the full result as JSON")
    args = parser.parse_args()

    with open(args.chunks_file, "r", encoding="utf-8") as file:
        chunks = json.load(file)

    compressor = AttnCompCompressor(
        model_name_or_path=args.model_path,
        checkpoint_path=args.checkpoint,
        layer=args.layer,
        window_size=args.window_size,
        local_files_only=args.local_files_only,
    )
    result = compressor.compress(args.question, chunks, p=args.p, epsilon=args.epsilon)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        return

    print("Kept indices:", result.kept_indices)
    print("Compressed context:\n")
    print(result.compressed_context)


if __name__ == "__main__":
    main()
