"""LongLLMLingua implementation for token-level compression."""

from typing import List, Optional
from llmlingua import PromptCompressor
from ...base import BaseCompressor, SearchResult
import re

class LongLLMLinguaCompressor(BaseCompressor):
    """LongLLMLingua: Dynamic token-level compression."""
    """If got error past_key_values, then read https://github.com/microsoft/LLMLingua/issues/210
    We can fix by modifying the package file, nothing to change here"""
    
    def __init__(
        self,
        device_map: str = "auto",
        context_budget: str = "+100",
        compression_rate: float = 0.4,
        reorder_context: str = "sort"
    ):
        self.device_map = device_map
        self.context_budget = context_budget
        self.compression_rate = compression_rate
        self.reorder_context = reorder_context
        
        # Initialize LLMLingua
        self.compressor = PromptCompressor(
            device_map=device_map,
        )
        
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using LongLLMLingua.
        
        Args:
            query: Input question
            documents: List of documents to compress
            
        Returns:
            List containing single SearchResult with compressed text
        """
        # Prepare documents
        prompt_list = [
            f"{doc.title}\n{doc.text}"
            for doc in documents
        ]
        
        instruction_text = "Given the context information, answer the query in plain text. Do not provide any explanation."
        # Compress using LLMLingua
        result = self.compressor.compress_prompt(
            prompt_list,
            question=query,
            instruction=instruction_text,
            rate=self.compression_rate,
            condition_compare=True,
            condition_in_question="after",
            rank_method="longllmlingua",
            use_sentence_level_filter=False,
            context_budget=self.context_budget,
            dynamic_context_compression_ratio=0.4,
            reorder_context=self.reorder_context
        )
        
        # Remove the instruction and the original question from the compressed prompt
        compressed_prompt = result["compressed_prompt"]
        lines = compressed_prompt.splitlines()
        filtered = [
            ln for ln in lines
            if ln.strip() not in {instruction_text.strip(), query.strip()}
        ]
        # Deduplicate spaces in each line
        cleaned_lines = [re.sub(r'\s+', ' ', ln).strip() for ln in filtered]
        cleaned_prompt = "\n".join(cleaned_lines).strip()
        
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=cleaned_prompt,
            score=1.0
        )]