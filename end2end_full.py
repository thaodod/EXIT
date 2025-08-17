#!/usr/bin/env python3
import json
import argparse
import os
import time
import spacy
import torch
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor

# Assuming transformers is installed
from transformers import AutoTokenizer, AutoModelForCausalLM
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from compressors import (
    SearchResult,
    CompActCompressor,
    EXITCompressor,
    RefinerCompressor,
    RecompAbstractiveCompressor,
    RecompExtractiveCompressor,
    LongLLMLinguaCompressor
)

# --- Utility Functions (Adapted from your utils.py) ---
from utils import (
    print_evaluation_results,
    format_prompt,
    generate_answers_api,
    evaluate_batch,
    preprocess_contexts,
    count_tokens
)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
MAX_OUT_LENGTH = 360
MAX_IN_LENGTH = 9216

# Load spaCy model for sentence segmentation once
try:
    nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'tok2vec'])
    nlp.enable_pipe("senter")
except IOError:
    print("Spacy model not found. Please run 'python -m spacy download en_core_web_sm'")
    exit()


def get_compressor(method: str):
    """Factory function to initialize and return a compressor instance."""
    print(f"Initializing compressor for method: {method}")
    if method == "compact":
        return CompActCompressor(model_dir='cwyoon99/CompAct-7b', device='cuda')
    elif method == "exit":
        return EXITCompressor(checkpoint="doubleyyh/exit-gemma-2b", device='cuda')
    elif method == "refiner":
        return RefinerCompressor()
    elif method == "recomp_abstractive":
        return RecompAbstractiveCompressor()
    elif method == "recomp_extractive":
        return RecompExtractiveCompressor()
    elif method == "longllmlingua":
        return LongLLMLinguaCompressor()
    else:
        raise ValueError(f"Unknown compression method: {method}")

def preprocess_single_question(question_data_with_k: Tuple[Dict, int]) -> List[Dict]:
    """Preprocess contexts for a single question. Used for parallel processing."""
    question_data, k = question_data_with_k
    local_nlp = spacy.load("en_core_web_sm", exclude=['tagger', 'parser', 'ner', 'lemmatizer', 'tok2vec'])
    local_nlp.enable_pipe("senter")
    return preprocess_contexts(question_data, k, local_nlp)


class EndToEndBenchmarkPipeline:
    def __init__(self, method: str, reader_model_name: str = None,
                 k: int = 10, reader_batch_size: int = 8,
                 use_auto_dtype: bool = False, api_model: str = None):
        self.k = k
        self.reader_batch_size = reader_batch_size
        self.use_api = api_model is not None
        self.api_model = api_model
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.use_multi_gpu = self.num_gpus > 1

        # Initialize the selected compressor
        self.compressor = get_compressor(method)
        print(f"Compressor '{self.compressor.__class__.__name__}' loaded.")
        
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
            self.use_auto_dtype = use_auto_dtype
            print("Reader model loaded successfully.")
        else:
            print(f"Using Vertex AI API with model: {self.api_model}")
            self.reader_tokenizer = None
            self.reader_model = None
            self.use_auto_dtype = False

    def preprocess_all_questions(self, questions: List[Dict]) -> List[List[Dict]]:
        """Preprocess all questions in parallel using CPU processes."""
        print("Preprocessing all questions...")
        questions_with_k = [(q, self.k) for q in questions]
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            preprocessed_segments = list(tqdm(
                executor.map(preprocess_single_question, questions_with_k),
                total=len(questions),
                desc="Preprocessing contexts"
            ))
        return preprocessed_segments

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

    def process_batch(self, batch_questions: List[str], batch_segments: List[List[Dict]], batch_ground_truths: List[Any]) -> Tuple[List[str], List[Any], Dict[str, float]]:
        """Process a batch of questions end-to-end with preprocessed segments."""
        prompts = []
        total_compress_time = 0
        total_original_tokens = 0
        total_compressed_tokens = 0

        # Iterate through each question in the batch for compression
        for i, question in enumerate(batch_questions):
            segments = batch_segments[i]
            
            # Calculate original tokens from preprocessed segments
            original_text = "\n".join(seg['text'] for seg in segments)
            original_tokens = count_tokens(original_text)
            total_original_tokens += original_tokens
            
            # Convert segments (dicts) to SearchResult objects for the compressor
            search_results = [
                SearchResult(
                    evi_id=idx, docid=idx,  # Using index as placeholder
                    title=seg['title'], text=seg['text'], score=float(seg['score'])
                ) for idx, seg in enumerate(segments)
            ]
            
            # Time and perform compression for a single question
            compress_start = time.time()
            compressed_docs = self.compressor.compress(question, search_results)
            total_compress_time += (time.time() - compress_start)
            
            # Combine compressed text into a final document
            final_document = "\n".join(doc.text for doc in compressed_docs)
            
            # Calculate compressed tokens
            compressed_tokens = count_tokens(final_document)
            total_compressed_tokens += compressed_tokens
            
            # Create prompt for the reader model
            prompt = format_prompt(question, final_document, self.reader_tokenizer, self.use_api)
            prompts.append(prompt)
        
        # Generate answers for the entire batch of prompts
        generate_start = time.time()
        predictions = self.generate_answers(prompts)
        generate_time = time.time() - generate_start
        
        batch_size = len(batch_questions)
        timing_info = {
            'compress_time_per_question': total_compress_time / batch_size,
            'generate_time_per_question': generate_time / batch_size,
            'original_tokens': total_original_tokens,
            'compressed_tokens': total_compressed_tokens
        }
        
        return predictions, batch_ground_truths, timing_info


def main():
    parser = argparse.ArgumentParser(description='End-to-End Benchmark Pipeline for Document Compression')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSON file with questions and contexts.')
    parser.add_argument('--method', '-m', type=str, required=True, 
                        choices=["compact", "exit", "refiner", "recomp_abstractive", "recomp_extractive", "longllmlingua"],
                        help="Compression method to use for the evaluation.")
    parser.add_argument('--reader_model_name', '-rm', type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                        help="Hugging Face model name for the reader.")
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='Number of queries to process per batch.')
    parser.add_argument('--k', '-k', type=int, default=10, help='Top k contexts to preprocess and compress.')
    parser.add_argument('--reader_batch_size', '-rb', type=int, default=8, help='Batch size for the reader model.')
    parser.add_argument('--auto_dtype', '-ad', action='store_true', help='Use torch_dtype="auto" for reader model loading.')
    parser.add_argument('--api', '-api', type=str, default=None, help='Vertex AI model name. If set, uses API instead of a local reader.')

    args = parser.parse_args()

    if args.api is None and args.reader_model_name is None:
        parser.error("Either --reader_model_name or --api must be provided.")

    pipeline = EndToEndBenchmarkPipeline(
        method=args.method,
        reader_model_name=args.reader_model_name,
        k=args.k,
        reader_batch_size=args.reader_batch_size,
        use_auto_dtype=args.auto_dtype,
        api_model=args.api
    )

    with open(args.input, 'r') as f:
        questions_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions_data)} questions from {args.input}")
    
    # Preprocess all question contexts first
    all_preprocessed_segments = pipeline.preprocess_all_questions(questions_data)
    
    all_questions = [item['question'] for item in questions_data]
    all_ground_truths = [item.get('answer') or item.get('answers') for item in questions_data]

    total_em, total_f1, total_count = 0, 0, 0
    total_compress_time, total_generate_time = 0, 0
    total_original_tokens, total_compressed_tokens = 0, 0

    for i in tqdm(range(0, len(questions_data), args.batch_size), desc="Processing Batches"):
        batch_questions = all_questions[i:i + args.batch_size]
        batch_segments = all_preprocessed_segments[i:i + args.batch_size]
        batch_ground_truths = all_ground_truths[i:i + args.batch_size]
        
        predictions, ground_truths, timing_info = pipeline.process_batch(
            batch_questions, batch_segments, batch_ground_truths
        )
        
        batch_results = evaluate_batch(predictions, ground_truths)
        
        total_em += batch_results['exact_match']
        total_f1 += batch_results['f1']
        total_count += batch_results['count']
        
        batch_item_count = len(batch_questions)
        total_compress_time += timing_info['compress_time_per_question'] * batch_item_count
        total_generate_time += timing_info['generate_time_per_question'] * batch_item_count
        total_original_tokens += timing_info['original_tokens']
        total_compressed_tokens += timing_info['compressed_tokens']

    if total_count > 0:
        final_results = {
            'count': total_count,
            'exact_match_percentage': 100.0 * total_em / total_count,
            'f1_percentage': 100.0 * total_f1 / total_count
        }
        avg_compress_time = total_compress_time / total_count
        avg_generate_time = total_generate_time / total_count
        avg_total_time = avg_compress_time + avg_generate_time
        compression_ratio = total_compressed_tokens / total_original_tokens if total_original_tokens > 0 else 0

        print_evaluation_results(final_results, f"FINAL EVALUATION RESULTS for '{args.method.upper()}'")
        print("\nAVERAGE PROCESSING TIME PER QUESTION:")
        print(f"  Compression:      {avg_compress_time:.4f} seconds")
        print(f"  Answer Generation:  {avg_generate_time:.4f} seconds")
        print(f"  ---------------------------------")
        print(f"  Total Inference:    {avg_total_time:.4f} seconds")
        print(f"\nCOMPRESSION STATISTICS:")
        print(f"  Compression Ratio:  {compression_ratio:.3f}")
        print("="*80)
    else:
        print("No questions were processed.")

if __name__ == "__main__":
    main()