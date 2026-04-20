"""Adapter baseline for the standalone AttnComp package."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from attn_comp import AttnCompCompressor as StandaloneAttnCompCompressor

from ...base import BaseCompressor, SearchResult


class AttnCompCompressor(BaseCompressor):
    """Wrap the standalone AttnComp package behind the repo's baseline interface."""

    def __init__(
        self,
        model_name_or_path: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        checkpoint_path: str | Path | None = None,
        *,
        p: float = 0.74,
        epsilon: float = 1e-2,
        layer: int = 13,
        window_size: int = 32,
        num_heads: int = 16,
        device_map: str | dict[str, Any] = "auto",
        dtype: Any = None,
        attn_implementation: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        self.p = p
        self.epsilon = epsilon
        self.last_result = None
        self.backend = StandaloneAttnCompCompressor(
            model_name_or_path=model_name_or_path,
            checkpoint_path=checkpoint_path,
            layer=layer,
            window_size=window_size,
            num_heads=num_heads,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            local_files_only=local_files_only,
        )

    def compress(self, query: str, documents: list[SearchResult]) -> list[SearchResult]:
        """Compress documents by selecting the most query-relevant chunks."""
        if not documents:
            return []

        chunks = [{"title": doc.title, "text": doc.text} for doc in documents]
        result = self.backend.compress(
            question=query,
            chunks=chunks,
            p=self.p,
            epsilon=self.epsilon,
        )
        self.last_result = result

        return [
            SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text=result.compressed_context,
                score=1.0 if result.kept_indices else 0.0,
            )
        ]
