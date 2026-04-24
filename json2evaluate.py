#!/usr/bin/env python3
import json
import argparse
import os
import re
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings

# Assuming transformers is installed
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from utils import (
    print_evaluation_results,
    generate_answers_api,
    evaluate_batch,
    format_prompt,
)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MAX_OUT_LENGTH = 1024
MAX_IN_LENGTH = 9216


def parse_method_and_k_from_filename(input_path: str) -> Tuple[str, Any]:
    """Infer method and k from names like ours_k20_NQ.json."""
    filename = os.path.basename(input_path)
    match = re.match(r"^(.+)_k(\d+)_", filename)
    if not match:
        return "unknown", "unknown"
    return match.group(1), int(match.group(2))


def is_ours_filename(input_path: str) -> bool:
    """Return whether the file basename marks an external ours output."""
    return "ours" in os.path.basename(input_path).lower()


def normalize_external_ours_records(records: List[Dict[str, Any]], input_path: str) -> List[Dict[str, Any]]:
    """Adapt external 'ours' records to the evaluator's expected item shape."""
    normalized_data = []

    for index, item in enumerate(records):
        if not isinstance(item, dict):
            raise ValueError(f"{input_path} item {index} must be an object.")

        question = item.get("question")
        context = item.get("compressed_document", item.get("compressed_context"))
        if question is None or context is None:
            raise ValueError(
                f"{input_path} item {index} must contain 'question' and "
                "'compressed_document' or 'compressed_context'."
            )

        if "ground_truth" in item:
            ground_truth = item["ground_truth"]
        elif "answer" in item:
            ground_truth = item["answer"]
        elif "answers" in item:
            ground_truth = item["answers"]
        else:
            raise ValueError(
                f"{input_path} item {index} must contain 'answer', 'answers', "
                "or 'ground_truth'."
            )

        prompt_api = item.get("prompt_api")
        if prompt_api is None:
            prompt_api = format_prompt(question, context, None, use_api=True)

        normalized_item = {
            "question": question,
            "compressed_context": context,
            "prompt_api": prompt_api,
            "ground_truth": ground_truth,
        }
        for optional_key in ("original_tokens", "compressed_tokens"):
            if optional_key in item:
                normalized_item[optional_key] = item[optional_key]
        normalized_data.append(normalized_item)

    return normalized_data


def normalize_compressed_input(data: Any, input_path: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize supported compressed JSON formats for evaluation.

    Existing compressor outputs already contain metadata/data and are returned
    unchanged. External list-style outputs are only accepted for filenames that
    contain "ours", and are adapted from question/compressed_document/answer(s)
    into the evaluator's prompt format.
    """
    if isinstance(data, dict) and "metadata" in data and "data" in data:
        if not isinstance(data["data"], list):
            raise ValueError(f"{input_path} has invalid 'data'; expected a list.")
        metadata = data["metadata"]
        records = data["data"]

        if is_ours_filename(input_path):
            method, k = parse_method_and_k_from_filename(input_path)
            metadata = dict(metadata)
            metadata.setdefault("method", method)
            metadata.setdefault("k", k)

            if records and isinstance(records[0], dict):
                first_item = records[0]
                if "prompt_api" not in first_item or "ground_truth" not in first_item:
                    return metadata, normalize_external_ours_records(records, input_path)

        return metadata, records

    if isinstance(data, list):
        if not is_ours_filename(input_path):
            raise ValueError(
                f"{input_path} is a list-style compressed file, but only external "
                "files with 'ours' in the filename are supported by this adapter."
            )

        method, k = parse_method_and_k_from_filename(input_path)
        normalized_data = normalize_external_ours_records(data, input_path)

        metadata = {
            "input_file": input_path,
            "method": method,
            "k": k,
            "total_questions": len(normalized_data),
            "compression_ratio": "unknown",
            "total_original_tokens": "unknown",
            "total_compressed_tokens": "unknown",
        }
        return metadata, normalized_data

    raise ValueError(
        f"{input_path} must be either an object with top-level 'metadata' and "
        "'data' keys, or a list of external compressed records."
    )


class EvaluationPipeline:
    def __init__(self, reader_model_name: str = None, reader_batch_size: int = 8,
                 use_auto_dtype: bool = False, api_model: str = None,
                 api_base_url: str = None, api_key: str = None,
                 openrouter_thinking: bool = False):
        self.reader_batch_size = reader_batch_size
        self.use_api = api_model is not None
        self.api_model = api_model
        self.api_base_url = api_base_url
        self.api_key = api_key
        self.openrouter_thinking = openrouter_thinking
        
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
            if self.api_base_url:
                print(f"Using OpenAI-compatible API at {self.api_base_url} with model: {self.api_model}")
            else:
                print(f"Using configured API routing with model: {self.api_model}")
            self.reader_tokenizer = None
            self.reader_model = None

    def generate_answers(self, prompts: List[str], return_metadata: bool = False) -> List[Any]:
        """Generate answers using either a local model or an API."""
        if not prompts:
            return []
        
        if self.use_api:
            return generate_answers_api(
                prompts,
                self.api_model,
                self.reader_batch_size,
                MAX_OUT_LENGTH,
                api_base_url=self.api_base_url,
                api_key=self.api_key,
                openrouter_thinking=self.openrouter_thinking,
                return_metadata=return_metadata,
            )
        else:
            responses = self._generate_answers_local(prompts)
            if return_metadata:
                return [
                    {
                        "text": response,
                        "ok": True,
                        "finish_reason": "LOCAL",
                        "provider": "local",
                        "error": None,
                    }
                    for response in responses
                ]
            return responses

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
                "max_new_tokens": MAX_OUT_LENGTH,
                "temperature": 0.001,
                "top_p": 1.0,
                "do_sample": True,
                "pad_token_id": self.reader_tokenizer.pad_token_id,
                "eos_token_id": self.reader_tokenizer.eos_token_id
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
        failed_requests_total = 0
        empty_valid_answers_total = 0
        
        # Process in batches
        for i in tqdm(range(0, len(prompts), batch_size), desc="Evaluating batches"):
            batch_prompts = prompts[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]
            
            # Generate answers
            if self.use_api:
                prediction_results = self.generate_answers(batch_prompts, return_metadata=True)
                failed_requests_total += sum(
                    1 for result in prediction_results if not result.get("ok")
                )
                empty_valid_answers_total += sum(
                    1
                    for result in prediction_results
                    if result.get("ok") and not result.get("text", "").strip()
                )
                filtered_pairs = [
                    (result.get("text", ""), gt)
                    for result, gt in zip(prediction_results, batch_ground_truths)
                    if result.get("ok")
                ]
                if filtered_pairs:
                    predictions, batch_ground_truths = map(list, zip(*filtered_pairs))
                else:
                    predictions, batch_ground_truths = [], []
            else:
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
                'skipped': failed_requests_total,
                'failed_requests': failed_requests_total,
                'empty_valid_answers': empty_valid_answers_total,
                'exact_match_percentage': 100.0 * total_em / total_count,
                'f1_percentage': 100.0 * total_f1 / total_count
            }
        else:
            return {
                'count': 0,
                'exact_match': 0,
                'f1': 0,
                'skipped': failed_requests_total,
                'failed_requests': failed_requests_total,
                'empty_valid_answers': empty_valid_answers_total,
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
    parser.add_argument('--api', '-api', type=str, default=None, help='API model name. If set, uses API instead of local reader.')
    parser.add_argument(
        '--openrouter-thinking',
        action='store_true',
        help='Enable OpenRouter thinking/reasoning. Disabled by default.',
    )
    parser.add_argument(
        '--api-base-url',
        type=str,
        default=None,
        help='OpenAI-compatible API base URL, such as http://host:8004/v1 for a vLLM server.',
    )
    parser.add_argument(
        '--api-key',
        type=str,
        default=None,
        help='Optional API key for --api-base-url. If omitted, uses OPENAI_API_KEY or VLLM_API_KEY when available.',
    )

    args = parser.parse_args()

    if args.api is None and args.reader_model_name is None:
        parser.error("Either --reader_model_name or --api must be provided.")
    if args.api is None and (args.api_base_url is not None or args.api_key is not None):
        parser.error("--api-base-url and --api-key require --api.")

    # Initialize evaluation pipeline
    pipeline = EvaluationPipeline(
        reader_model_name=args.reader_model_name,
        reader_batch_size=args.reader_batch_size,
        use_auto_dtype=args.auto_dtype,
        api_model=args.api,
        api_base_url=args.api_base_url,
        api_key=args.api_key,
        openrouter_thinking=args.openrouter_thinking,
    )

    # Load compressed data
    with open(args.input, 'r') as f:
        data = json.load(f)
    
    metadata, compressed_data = normalize_compressed_input(data, args.input)
    
    print(f"Loaded compressed data from: {args.input}")
    print(f"Method: {metadata['method']}")
    print(f"K: {metadata['k']}")
    print(f"Total questions: {metadata['total_questions']}")
    compression_ratio = metadata["compression_ratio"]
    if isinstance(compression_ratio, (int, float)):
        compression_ratio_text = f"{compression_ratio:.3f}"
    else:
        compression_ratio_text = str(compression_ratio)
    print(f"Compression ratio: {compression_ratio_text}")
    
    # Evaluate
    results = pipeline.evaluate_compressed_data(compressed_data, args.batch_size)
    
    # Print results
    title = f"EVALUATION RESULTS for '{metadata['method'].upper()}' (k={metadata['k']})"
    print_evaluation_results(results, title)
    
    print(f"\nCOMPRESSION STATISTICS:")
    print(f"  Compression Ratio: {compression_ratio_text}")
    original_tokens = metadata["total_original_tokens"]
    compressed_tokens = metadata["total_compressed_tokens"]
    original_tokens_text = (
        f"{original_tokens:,}" if isinstance(original_tokens, int) else str(original_tokens)
    )
    compressed_tokens_text = (
        f"{compressed_tokens:,}" if isinstance(compressed_tokens, int) else str(compressed_tokens)
    )
    print(f"  Original Tokens: {original_tokens_text}")
    print(f"  Compressed Tokens: {compressed_tokens_text}")
    print("="*80)

if __name__ == "__main__":
    main()
