import argparse

from .compressor import AttnCompCompressor


def build_demo_chunks() -> list[str]:
    return [
        "Cooking Tips: Salt boiling water and stir pasta early to keep it from sticking.",
        "Travel Diary: Reykjavik is known for cafes, geothermal spas, and volcanic landscapes.",
        "Smartphone Review: New phones focus on AI cameras, battery life, and repairability.",
        "Ancient Egypt: The Nile supported agriculture, transport, and religion for centuries.",
        (
            "Retrieval-Augmented Generation improves factual accuracy, brings in up-to-date knowledge, "
            "reduces hallucination, and helps domain adaptation by using external documents."
        ),
        "Football Match: The game was decided by a late counterattack goal in stoppage time.",
        "Black Holes: The event horizon is the boundary beyond which light cannot escape.",
        "Finance Report: Subscription revenue rose, but infrastructure costs also increased.",
        "Dictionary Entry: Serendipity means a beneficial event discovered by chance.",
        (
            "Failure Modes in LLMs: External evidence can reduce hallucination, "
            "but irrelevant context still harms accuracy."
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone AttnComp demo")
    parser.add_argument("--model-path", required=True, help="Path or Hugging Face id for Meta-Llama-3.1-8B-Instruct")
    parser.add_argument("--checkpoint", default=None, help="Optional override for the bundled attention checkpoint")
    parser.add_argument("--layer", type=int, default=13)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--epsilon", type=float, default=1e-2)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    compressor = AttnCompCompressor(
        model_name_or_path=args.model_path,
        checkpoint_path=args.checkpoint,
        layer=args.layer,
        window_size=args.window_size,
        local_files_only=args.local_files_only,
    )

    question = "What are the advantages of Retrieval-Augmented Generation (RAG)?"
    result = compressor.compress(question, build_demo_chunks(), p=args.p, epsilon=args.epsilon)

    print("Question:", question)
    print("Kept indices:", result.kept_indices)
    print("Compressed context:\n")
    print(result.compressed_context)
    print("\nDocument scores:")
    for index in result.ranked_indices:
        print(f"  doc {index}: {result.doc_scores[index]:.6f}")
    print("Instruction score:", f"{result.instruction_score:.6f}")


if __name__ == "__main__":
    main()
