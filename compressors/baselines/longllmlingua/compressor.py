"""LongLLMLingua implementation for token-level compression."""

from typing import List, Optional
from llmlingua import PromptCompressor
from ...base import BaseCompressor, SearchResult
import re
import torch
import types

class LongLLMLinguaCompressor(BaseCompressor):
    """LongLLMLingua with cache compatibility for recent transformers versions."""
    
    def __init__(
        self,
        device_map: str = "auto",
        context_budget: str = "+100",
        compression_rate: float = 0.3,
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
        self._patch_get_ppl_for_cache_compat()

    def _patch_get_ppl_for_cache_compat(self) -> None:
        """Patch llmlingua get_ppl for transformers cache compatibility."""
        if getattr(self.compressor, "_cache_compat_patched", False):
            return

        try:
            from transformers.cache_utils import DynamicCache
        except Exception:
            # Older transformers versions don't need this patch.
            return

        def _compatible_get_ppl(
            this,
            text: str,
            granularity: str = "sentence",
            input_ids=None,
            attention_mask=None,
            past_key_values=None,
            return_kv=False,
            end=None,
            condition_mode: str = "none",
            condition_pos_id: int = 0,
        ):
            if input_ids is None:
                tokenized_text = this.tokenizer(text, return_tensors="pt")
                input_ids = tokenized_text["input_ids"].to(this.device)
                attention_mask = tokenized_text["attention_mask"].to(this.device)

            if past_key_values is not None:
                if hasattr(past_key_values, "get_seq_length"):
                    past_length = past_key_values.get_seq_length()
                else:
                    past_length = past_key_values[0][0].shape[2]
            else:
                past_length = 0

            if end is None:
                end = input_ids.shape[1]
            end = min(end, past_length + this.max_position_embeddings)

            model_past_key_values = past_key_values
            if past_key_values is not None and not hasattr(past_key_values, "get_seq_length"):
                try:
                    legacy_cache = tuple((layer[0], layer[1]) for layer in past_key_values)
                    model_past_key_values = DynamicCache.from_legacy_cache(legacy_cache)
                except Exception:
                    model_past_key_values = past_key_values

            with torch.no_grad():
                response = this.model(
                    input_ids[:, past_length:end],
                    attention_mask=attention_mask[:, :end],
                    past_key_values=model_past_key_values,
                    use_cache=True,
                )
                past_key_values = response.past_key_values

            if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
                legacy_cache = past_key_values.to_legacy_cache()
                past_key_values = [[k, v] for (k, v) in legacy_cache]

            shift_logits = response.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., past_length + 1 : end].contiguous()
            active = (attention_mask[:, past_length:end] == 1)[..., :-1].view(-1)
            active_logits = shift_logits.view(-1, shift_logits.size(-1))[active]
            active_labels = shift_labels.view(-1)[active]
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            loss = loss_fct(active_logits, active_labels)
            if condition_mode == "before":
                loss = loss[:condition_pos_id]
            elif condition_mode == "after":
                loss = loss[condition_pos_id:]
            res = loss.mean() if granularity == "sentence" else loss
            return (res, past_key_values) if return_kv else res

        self.compressor.get_ppl = types.MethodType(_compatible_get_ppl, self.compressor)
        self.compressor._cache_compat_patched = True
        
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
            dynamic_context_compression_ratio=0.3,
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