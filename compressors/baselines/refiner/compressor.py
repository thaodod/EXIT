"""Refiner implementation for abstractive compression."""

import torch
from typing import List
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import PeftModel
from ...base import BaseCompressor, SearchResult

class RefinerCompressor(BaseCompressor):
    """Refiner: Abstractive document compression."""
    
    def __init__(
        self,
        base_model: str = "meta-llama/Llama-2-7b-chat-hf",
        adapter: str = "al1231/Refiner-7B",
        device: str = "cuda",
        max_tokens: int = 4096,  # Changed to 4k for better performance
        cache_dir: str = "./cache"
    ):
        """Initialize Refiner compressor.
        
        Args:
            base_model: Base model path
            adapter: Path to trained adapter
            device: Device to use
            max_tokens: Maximum tokens per context (4k recommended)
            cache_dir: Cache directory for models
        """
        self.device = device
        self.max_tokens = max_tokens
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            cache_dir=cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
            use_cache=True,
        )
        
        # Load adapter
        self.model = PeftModel.from_pretrained(
            self.base_model,
            adapter,
            is_trainable=False
        )
        self.model.eval()
        self.model = torch.compile(self.model, mode="max-autotune", fullgraph=True)
        
        # Prompt template
        self.template = (
            "[INST]<<SYS>>"
            "[MONITOR]{context}"
            "<</SYS>>{question}[/INST] "
        )
    
    def _truncate_context(self, context: str) -> str:
        """Truncate context to maximum tokens."""
        tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False
        )
        if len(tokens) <= self.max_tokens:
            return context
            
        truncated_tokens = tokens[:self.max_tokens]
        return self.tokenizer.decode(
            truncated_tokens,
            skip_special_tokens=True
        )
    
    def _prepare_input(self, query: str, context: str) -> dict:
        """Prepare model input."""
        context = self._truncate_context(context)
        prompt = self.template.format(
            question=query,
            context=context
        )
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )
        return {
            k: v.to(self.device)
            for k, v in inputs.items()
        }
    
    def _generate(self, inputs: dict) -> List[str]:
        """Generate compressed output."""
        with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type=self.device):
            outputs = self.model.generate(
                **inputs,
                top_p=1,
                temperature=None,
                do_sample=False,
                max_new_tokens=1024,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated tokens
        pred_tokens = outputs.sequences[
            :,
            inputs['input_ids'].shape[1]:
        ]
        return self.tokenizer.batch_decode(
            pred_tokens,
            skip_special_tokens=True
        )
    
    def _create_context_chunks(self, query: str, documents: List[SearchResult]) -> List[str]:
        """Split documents into multiple context chunks that fit within token limit."""
        chunks = []
        current_chunk = ""
        
        # Calculate tokens needed for prompt template (excluding context)
        template_without_context = self.template.format(question=query, context="")
        template_tokens = len(self.tokenizer.encode(template_without_context, add_special_tokens=False))
        
        # Reserve tokens for generation and safety margin
        available_tokens = self.max_tokens - template_tokens - 100  # 100 token safety margin
        
        for doc in documents:
            if not doc.text.strip():
                continue
                
            doc_text = f"## {doc.title}\n{doc.text}\n"
            doc_tokens = len(self.tokenizer.encode(doc_text, add_special_tokens=False))
            
            # Check if adding this document would exceed token limit
            current_tokens = len(self.tokenizer.encode(current_chunk, add_special_tokens=False))
            
            if current_tokens + doc_tokens > available_tokens and current_chunk:
                # Save current chunk and start new one
                chunks.append(current_chunk.strip())
                current_chunk = doc_text
            else:
                # Add document to current chunk
                current_chunk += doc_text
        
        # Add the last chunk if it's not empty
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def _prepare_batch_inputs(self, query: str, context_chunks: List[str]) -> dict:
        """Prepare batch inputs for multiple contexts."""
        prompts = [
            self.template.format(question=query, context=chunk)
            for chunk in context_chunks
        ]
        
        # Tokenize all prompts
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_tokens
        )
        
        return {
            k: v.to(self.device)
            for k, v in inputs.items()
        }
    
    def _generate_batch(self, inputs: dict) -> List[str]:
        """Generate compressed output for batch of contexts."""
        with torch.no_grad(), torch.amp.autocast(dtype=torch.bfloat16, device_type=self.device):
            outputs = self.model.generate(
                **inputs,
                top_p=1,
                temperature=None,
                do_sample=False,
                max_new_tokens=1024,
                num_return_sequences=1,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Extract generated tokens for each sequence in batch
        batch_size = inputs['input_ids'].shape[0]
        pred_tokens = outputs.sequences[
            :,
            inputs['input_ids'].shape[1]:
        ]
        
        return self.tokenizer.batch_decode(
            pred_tokens,
            skip_special_tokens=True
        )
    
    def compress(
        self,
        query: str,
        documents: List[SearchResult]
    ) -> List[SearchResult]:
        """Compress documents using Refiner with batch processing."""
        # Split documents into multiple context chunks
        context_chunks = self._create_context_chunks(query, documents)
        
        if not context_chunks:
            return [SearchResult(
                evi_id=0,
                docid=0,
                title="",
                text="",
                score=1.0
            )]
        
        # Prepare batch inputs
        batch_inputs = self._prepare_batch_inputs(query, context_chunks)
        
        # Generate compressed text for all chunks
        compressed_outputs = self._generate_batch(batch_inputs)
        
        # Concatenate all compressed texts
        final_compressed_text = "\n".join(
            output.strip() for output in compressed_outputs
            if output.strip()
        )
        
        # Return single compressed result
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=final_compressed_text,
            score=1.0
        )]