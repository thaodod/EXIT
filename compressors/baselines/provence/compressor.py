"""Provence implementation for context pruning."""

import os
import sys
import io
import torch
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from typing import List
from transformers import AutoModel
from ...base import BaseCompressor, SearchResult


class ProvenceCompressor(BaseCompressor):
    """Provence: Neural context pruning with sentence-level selection."""
    
    def __init__(
        self,
        model_name: str = "naver/provence-reranker-debertav3-v1",
        device: str = 'cuda',
        threshold: float = 0.1,
        always_select_title: bool = True,
        batch_size: int = 32,
        reorder: bool = False,
        top_k: int = 5,
        enable_warnings: bool = False
    ):
        """Initialize Provence compressor.
        
        Args:
            model_name: Hugging Face model name
            device: Device to use ('cuda' or 'cpu')
            threshold: Threshold for context pruning (0.1 for conservative, 0.5 for aggressive)
            always_select_title: Keep the first sentence (title) if available
            batch_size: Batch size for processing
            reorder: Whether to reorder contexts by relevance
            top_k: Number of top-ranked passages to keep if reorder=True
            enable_warnings: Whether to print warnings about model usage
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.threshold = threshold
        self.always_select_title = always_select_title
        self.batch_size = batch_size
        self.reorder = reorder
        self.top_k = top_k
        self.enable_warnings = enable_warnings
        
        print(f"Loading Provence model from {model_name}...")
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Move model to device if it's not already handled by AutoModel
        if hasattr(self.model, 'to'):
            self.model.to(self.device)
        
        print("Provence model loaded successfully.")

    @contextmanager
    def _quiet_provence(self):
        """Temporarily silence Provence tqdm/stdout/stderr noise."""
        prev = os.environ.get("TQDM_DISABLE")
        os.environ["TQDM_DISABLE"] = "1"
        buf_out, buf_err = io.StringIO(), io.StringIO()
        try:
            with redirect_stdout(buf_out), redirect_stderr(buf_err):
                yield
        finally:
            if prev is None:
                os.environ.pop("TQDM_DISABLE", None)
            else:
                os.environ["TQDM_DISABLE"] = prev
    
    def compress(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Compress documents using Provence pruning.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            Compressed documents with pruned text
        """
        if not documents:
            return []
        
        compressed_docs: List[SearchResult] = []

        # Call Provence per-document to avoid shape mismatches in the HF helper
        for doc in documents:
            with self._quiet_provence():
                output = self.model.process(
                    question=query,
                    context=doc.text,
                    title="first_sentence",  # rely on Provence's default title handling
                    threshold=self.threshold,
                    always_select_title=self.always_select_title,
                    batch_size=self.batch_size,
                    reorder=self.reorder,
                    top_k=self.top_k,
                    enable_warnings=self.enable_warnings
                )

            if not isinstance(output, dict):
                # Unexpected format; skip this doc
                continue

            pruned_text = output.get('pruned_context', '')
            score = output.get('reranking_score')

            if pruned_text and pruned_text.strip():
                compressed_docs.append(
                    SearchResult(
                        evi_id=doc.evi_id,
                        docid=doc.docid,
                        title=doc.title,
                        text=pruned_text,
                        score=score
                    )
                )

        return compressed_docs
