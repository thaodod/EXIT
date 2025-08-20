#!/usr/bin/env python3
import json
import argparse
import os
import torch
from typing import List, Dict, Any
from tqdm import tqdm
import warnings

# Assuming transformers is installed
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from utils import (
    print_evaluation_results,
    generate_answers_api,
    evaluate_batch
)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MAX_OUT_LENGTH = 360
MAX_IN_LENGTH = 9216

class EvaluationPipeline:
    def __init__(self, reader_model_name: str = None, reader_batch_size: int = 8,
                 use_auto_dtype: bool = False, api_model: str = None):
        self.reader_batch_size = reader_batch_size
        self.use_api = api_model is not None
        self.api_model = api_model
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = self.num_gpus > 1

        # Load reader model only if not using API
        if not self.use_api:
            print(f"Loading reader model: {reader_model_name}")
            self.reader_tokenizer = AutoTokenizer.from_pretrained(
                reader_model_name, padding_side="left", trust_remote_code=True)
            if self.reader_tokenizer.pad_token is None:
                self.reader_tokenizer.pad_token = self.reader_tokenizer.eos_token
            
            torch_dtype = "auto" if use_auto_dtype else torch.bfloat16
            device_map = "auto" if self.use_multi_gpu else {"": 0}
            attn_impl = "sdpa" if "gemma" in reader_model_name.lower() else "flash_attention_2"
            
            self.reader_model = AutoModelForCausalLM.from_pretrained(
                reader_model_name,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                attn_implementation=attn_impl
            )
            self.reader_model.eval()
            print("Reader model loaded successfully.")
        else:
            print(f"Using Vertex AI API with model: {self.api_model}")
            self.reader_tokenizer = None
            self.reader_model = None

    def generate_answers(self, prompts: List[str]) -> List[str]:
        """Generate answers using either a local model or an API."""
        if not prompts:
            return []
        
        if self.use_api:
            return generate_answers_api(prompts, self.api_model, self.reader_batch_size, MAX_OUT_LENGTH)
        else:
            return self._generate_answers_local(prompts)

    def _generate_answers_local(self, prompts: List[str]) -> List[str]:
        """Generate answers using the local reader model."""
        all_responses = []
        for i in tqdm(range(0, len(prompts), self.reader_batch_size), desc="Generating Answers", leave=False):
            batch_prompts = prompts[i:i + self.reader_batch_size]
            inputs = self.reader_tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=MAX_IN_LENGTH, add_special_tokens=False
            ).to(self.device)
            
            generate_kwargs = {
                "max_new_tokens": MAX_OUT_LENGTH, "temperature": 0.0001,
                "pad_token_id": self.reader_tokenizer.pad_token_id,
            }
            
            with torch.no_grad():
                outputs = self.reader_model.generate(**inputs, **generate_kwargs)
            
            input_lengths = inputs.input_ids.shape[1]
            generated_tokens = outputs[:, input_lengths:]
            batch_responses = self.reader_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
            all_responses.extend([resp.strip() for resp in batch_responses])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return all_responses

    def evaluate_compressed_data(self, compressed_data: List[Dict], batch_size: int = 8) -> Dict[str, float]:
        """Evaluate compressed data by generating answers and computing metrics."""
        
        # Extract prompts and ground truths
        if self.use_api:
            prompts = [item['prompt_api'] for item in compressed_data]
        else:
            # For local models, we need to apply chat template to the API prompt content
            prompts = []
            for item in compressed_data:
                if self.reader_tokenizer and hasattr(self.reader_tokenizer, 'apply_chat_template'):
                    # Use the same content as API but apply chat template for local model
                    content = item['prompt_api']  # Use the same prompt content
                    messages = [{"role": "user", "content": content}]
                    prompt = self.reader_tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    prompts.append(prompt)
                else:
                    # Fallback: use the API prompt directly
                    prompts.append(item['prompt_api'])
        
        ground_truths = [item['ground_truth'] for item in compressed_data]
        
        total_em, total_f1, total_count = 0, 0, 0
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating batches"):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]
            
            # Generate answers
            predictions = self.generate_answers(batch_prompts)
            
            # Evaluate batch
            batch_results = evaluate_batch(predictions, batch_ground_truths)
            
            total_em += batch_results['exact_match']
            total_f1 += batch_results['f1']
            total_count += batch_results['count']
        
        if total_count > 0:
            return {
                'count': total_count,
                'exact_match': total_em,
                'f1': total_f1,
                'exact_match_percentage': 100.0 * total_em / total_count,
                'f1_percentage': 100.0 * total_f1 / total_count
            }
        else:
            return {
                'count': 0,
                'exact_match': 0,
                'f1': 0,
                'exact_match_percentage': 0,
                'f1_percentage': 0
            }

def main():
    parser = argparse.ArgumentParser(description='Evaluate Compressed Contexts')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSON file with compressed data.')
    parser.add_argument('--reader_model_name', '-rm', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Hugging Face model name for the reader.")
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Batch size for evaluation.')
    parser.add_argument('--reader_batch_size', '-rb', type=int, default=8, help='Batch size for the reader model.')
    parser.add_argument('--auto_dtype', '-ad', action='store_true', help='Use torch_dtype="auto" for reader model loading.')
    parser.add_argument('--api', '-api', type=str, default=None, help='Vertex AI model name. If set, uses API instead of local reader.')

    args = parser.parse_args()

    if args.api is None and args.reader_model_name is None:
        parser.error("Either --reader_model_name or --api must be provided.")

    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(
        reader_model_name=args.reader_model_name,
        reader_batch_size=args.reader_batch_size,
        use_auto_dtype=args.auto_dtype,
        api_model=args.api
    )

    # Load compressed data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    compressed_data = data['data']
    
    print(f"Loaded compressed data from: {args.input}")
    print(f"Method: {metadata['method']}")
    print(f"K: {metadata['k']}")
    print(f"Total questions: {metadata['total_questions']}")
    print(f"Compression ratio: {metadata['compression_ratio']:.3f}")
    
    # Evaluate
    results = pipeline.evaluate_compressed_data(compressed_data, args.batch_size)
    
    # Print results
    title = f"EVALUATION RESULTS for '{metadata['method'].upper()}' (k={metadata['k']})"
    print_evaluation_results(results, title)
    
    print(f"\nCOMPRESSION STATISTICS:")
    print(f"  Compression Ratio: {metadata['compression_ratio']:.3f}")
    print(f"  Original Tokens: {metadata['total_original_tokens']:,}")
    print(f"  Compressed Tokens: {metadata['total_compressed_tokens']:,}")
    print("="*80)

if __name__ == "__main__":
    main()