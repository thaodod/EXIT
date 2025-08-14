"""CompAct implementation for document compression."""

import re
import torch
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from ...base import BaseCompressor, SearchResult

class CompActCompressor(BaseCompressor):
    """CompAct: Context compression using iterative summarization."""
    
    def __init__(
        self,
        model_dir: str = 'cwyoon99/CompAct-7b',
        device: str = 'cuda',
        cache_dir: str = "./cache",
        batch_size: int = 10
    ):
        """Initialize CompAct compressor.
        
        Args:
            model_dir: HuggingFace model path
            device: Device to run model on
            cache_dir: Cache directory for models
            batch_size: Number of documents to process at once
        """
        self.device = device
        self.batch_size = batch_size
        
        # Load model and tokenizer with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            cache_dir=cache_dir,
            attn_implementation="flash_attention_2",
            low_cpu_mem_usage=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir, 
            use_fast=True,
            padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token   
        self.model.eval()  # Set to eval mode for inference optimization
        self.model = torch.compile(self.model, mode="reduce-overhead")
        
    def _create_prompt(
        self,
        query: str,
        documents: str,
        prev_summary: str = "",
        prev_eval: str = "",
        iteration: int = 0
    ) -> str:
        """Create prompt for CompAct model."""
        if iteration == 0:
            instruction = (
                "1. Generate a summary of source documents to answer the question. "
                "Ensure the summary is under 200 words and does not include any pronouns. "
                "DO NOT make assumptions or attempt to answer the question; "
                "your job is to summarize only.\n\n"
                "2. Evaluate the summary based solely on the information of it, "
                "without any additional background context: if it lacks sufficient "
                "details to answer the question, print '[INCOMPLETE]'. If it provides "
                "all necessary details, print '[COMPLETE]'. You should provide the "
                "reason of evalution."
            )
            prompt = f"{instruction}\n\nQuestion: {query}\n\nSource documents: {documents}\n\nSummary:"
        else:
            instruction = (
                "1. Generate a summary of the previous summary and the source documents "
                "to answer the question based on the evaluation of the previous summary. "
                "The evaluation indicates the missing information needed to answer the "
                "question. Ensure the summary is under 200 words and does not include "
                "any pronouns. DO NOT make assumptions or attempt to answer the question; "
                "your job is to summarize only.\n\n"
                "2. Evaluate the summary based solely on the information of it, without "
                "any additional background context: if it lacks sufficient details to "
                "answer the question, print '[INCOMPLETE]'. If it provides all necessary "
                "details, print '[COMPLETE]'. You should provide the reason of evalution."
            )
            prompt = (
                f"{instruction}\n\nQuestion: {query}\n\n"
                f"Previous summary: {prev_summary}\n\n"
                f"Evaluation of previous summary: {prev_eval}\n\n"
                f"Source documents: {documents}\n\nSummary:"
            )
        
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    def _parse_output(self, text: str) -> Dict[str, str]:
        """Parse model output into summary and evaluation."""
        # The model outputs summary directly, then "Evaluation:" followed by evaluation
        # Look for "Evaluation:" to split the text
        eval_start = text.find("Evaluation:")
        
        if eval_start != -1:
            # Split at "Evaluation:"
            summary = text[:eval_start].strip()
            evaluation = text[eval_start:].strip()
        else:
            # Fallback: look for [COMPLETE] or [INCOMPLETE] markers
            complete_pos = text.find('[COMPLETE]')
            incomplete_pos = text.find('[INCOMPLETE]')
            
            if complete_pos != -1:
                # Find the last sentence before [COMPLETE]
                marker_context = text[:complete_pos + len('[COMPLETE]')]
                lines = marker_context.split('\n')
                
                # Find where evaluation starts (usually after a line break before [COMPLETE])
                summary_lines = []
                eval_lines = []
                found_eval = False
                
                for line in lines:
                    if '[COMPLETE]' in line or '[INCOMPLETE]' in line:
                        eval_lines.append(line)
                        found_eval = True
                    elif found_eval:
                        eval_lines.append(line)
                    else:
                        summary_lines.append(line)
                
                summary = '\n'.join(summary_lines).strip()
                evaluation = '\n'.join(eval_lines).strip()
                
            elif incomplete_pos != -1:
                # Similar logic for [INCOMPLETE]
                marker_context = text[:incomplete_pos + len('[INCOMPLETE]')]
                lines = marker_context.split('\n')
                
                summary_lines = []
                eval_lines = []
                found_eval = False
                
                for line in lines:
                    if '[COMPLETE]' in line or '[INCOMPLETE]' in line:
                        eval_lines.append(line)
                        found_eval = True
                    elif found_eval:
                        eval_lines.append(line)
                    else:
                        summary_lines.append(line)
                
                summary = '\n'.join(summary_lines).strip()
                evaluation = '\n'.join(eval_lines).strip()
            else:
                # No evaluation markers found, treat whole text as summary
                summary = text.strip()
                evaluation = "[COMPLETE]"
        
        # Clean up the text
        summary = re.sub(r'\s+', ' ', summary).strip()
        evaluation = re.sub(r'\s+', ' ', evaluation).strip()
        
        return {
            "summary": summary,
            "eval": evaluation
        }
    
    def compress(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        """Compress documents using iterative summarization."""
        prev_summaries = []
        prev_evals = []
        
        for i in range(0, len(documents), self.batch_size):
            batch_docs = documents[i:i + self.batch_size]
            batch_text = "\n".join(f"{doc.title}\n{doc.text}" for doc in batch_docs)
            
            # Get previous summary and eval if available
            prev_summary = prev_summaries[-1] if prev_summaries else ""
            prev_eval = prev_evals[-1].replace('[INCOMPLETE]', '').strip() if prev_evals else ""
            
            # Generate summary
            prompt = self._create_prompt(
                query, batch_text, prev_summary, prev_eval, i // self.batch_size
            )
            
            # Optimized inference with reduced memory usage
            with torch.no_grad():
                # Tokenize with optimizations
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096,  # Prevent extremely long inputs
                    padding=False
                ).to(self.device)
                
                # Optimized generation parameters
                outputs = self.model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=512,
                    top_p=1.0,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,  # Greedy decoding is faster
                    use_cache=True,   # Enable KV cache for faster generation
                    num_beams=1,      # Greedy search instead of beam search
                    repetition_penalty=1.0,  # Disable repetition penalty for speed
                    output_attentions=False,  # Don't compute attention weights
                    output_hidden_states=False,  # Don't compute hidden states
                    return_dict_in_generate=False  # Return only token ids
                )
                
                output_text = self.tokenizer.decode(
                    outputs[0][inputs.input_ids.size(1):],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False  # Faster decoding
                ).strip()
                
                # Parse output
                parsed = self._parse_output(output_text)
                prev_summaries.append(parsed["summary"])
                prev_evals.append(parsed["eval"])
                
                # Early stopping if complete (preserving CompAct's logic)
                if "[COMPLETE]" in parsed["eval"]:
                    break
                
                # Clear GPU cache periodically to prevent OOM
                if i > 0 and i % (self.batch_size * 2) == 0:
                    torch.cuda.empty_cache()
        
        # Return compressed result - use the final summary directly
        final_summary = prev_summaries[-1] if prev_summaries else ""
        
        return [SearchResult(
            evi_id=0,
            docid=0,
            title="",
            text=final_summary,
            score=1.0
        )]