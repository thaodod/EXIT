"""Shared factory for baseline compressor construction."""

from __future__ import annotations

import os

from .base import BaseCompressor


SUPPORTED_METHODS = (
    "compact",
    "exit",
    "refiner",
    "recomp_abstractive",
    "recomp_extractive",
    "longllmlingua",
    "provence",
    "attn_comp",
)


def _get_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_compressor(method: str) -> BaseCompressor:
    """Build a compressor instance for a supported method name."""
    print(f"Initializing compressor for method: {method}")

    if method == "compact":
        from .baselines.compact.compressor import CompActCompressor

        return CompActCompressor(model_dir="cwyoon99/CompAct-7b", device="cuda")
    if method == "exit":
        from .baselines.exit.compressor import EXITCompressor

        return EXITCompressor(checkpoint="doubleyyh/exit-gemma-2b", device="cuda")
    if method == "refiner":
        from .baselines.refiner.compressor import RefinerCompressor

        return RefinerCompressor()
    if method == "recomp_abstractive":
        from .baselines.recomp_abst.compressor import RecompAbstractiveCompressor

        return RecompAbstractiveCompressor()
    if method == "recomp_extractive":
        from .baselines.recomp_extr.compressor import RecompExtractiveCompressor

        return RecompExtractiveCompressor()
    if method == "longllmlingua":
        from .baselines.longllmlingua.compressor import LongLLMLinguaCompressor

        return LongLLMLinguaCompressor()
    if method == "provence":
        from .baselines.provence.compressor import ProvenceCompressor

        return ProvenceCompressor(device="cuda", threshold=0.1)
    if method == "attn_comp":
        from .baselines.attn_comp.compressor import AttnCompCompressor

        return AttnCompCompressor(
            model_name_or_path=os.getenv(
                "ATTN_COMP_MODEL_NAME_OR_PATH",
                "meta-llama/Meta-Llama-3.1-8B-Instruct",
            ),
            checkpoint_path=os.getenv("ATTN_COMP_CHECKPOINT_PATH") or None,
            p=float(os.getenv("ATTN_COMP_P", "0.9")),
            epsilon=float(os.getenv("ATTN_COMP_EPSILON", "1e-2")),
            local_files_only=_get_bool_env("ATTN_COMP_LOCAL_FILES_ONLY", False),
        )

    raise ValueError(f"Unknown compression method: {method}")
