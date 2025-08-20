#!/usr/bin/env python3
import json
import argparse
import os
import spacy
import torch
from typing import List, Dict, Tuple
from tqdm import tqdm
import warnings
from concurrent.futures import ProcessPoolExecutor

from compressors import (
    SearchResult,
    CompActCompressor,
    EXITCompressor,
    RefinerCompressor,
    RecompAbstractiveCompressor,
    RecompExtractiveCompressor,
    LongLLMLinguaCompressor
)

from utils import (
    format_prompt,
    preprocess_contexts,
    count_tokens
)

warnings.filterwarnings("ignore")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

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

def extract_dataset_name(input_path: str) -> str:
    """Extract dataset name from input path."""
    # Example: retrieved/contriever-msmarco_NQ/dev.json -> NQ
    parts = input_path.split('/')
    for part in parts:
        if '_' in part:
            # Split by underscore and take the last part
            dataset_name = part.split('_')[-1]
            return dataset_name
    # Fallback: use the filename without extension
    filename = os.path.basename(input_path)
    return os.path.splitext(filename)[0]

def generate_output_path(input_path: str, method: str, k: int) -> str:
    """Generate output path based on input path, method, and k."""
    dataset_name = extract_dataset_name(input_path)
    output_filename = f"{method}_k{k}_{dataset_name}.json"
    
    # Place in the same directory as input or in 'retrieved' directory
    if 'retrieved' in input_path:
        output_path = os.path.join('retrieved', output_filename)
    else:
        output_path = output_filename
    
    return output_path

class CompressionPipeline:
    def __init__(self, method: str, k: int = 10):
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the selected compressor
        self.compressor = get_compressor(method)
        print(f"Compressor '{self.compressor.__class__.__name__}' loaded.")

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

    def compress_and_format(self, questions_data: List[Dict], preprocessed_segments: List[List[Dict]]) -> Tuple[List[Dict], Dict[str, float]]:
        """Compress contexts and format prompts for all questions."""
        compressed_data = []
        total_original_tokens = 0
        total_compressed_tokens = 0
        
        for i, question_data in enumerate(tqdm(questions_data, desc="Compressing contexts")):
            question = question_data['question']
            segments = preprocessed_segments[i]
            
            # Calculate original tokens from preprocessed segments
            original_text = "\n".join(seg['text'] for seg in segments)
            original_tokens = count_tokens(original_text)
            total_original_tokens += original_tokens
            
            # Convert segments to SearchResult objects for the compressor
            search_results = [
                SearchResult(
                    evi_id=idx, docid=idx,
                    title=seg['title'], text=seg['text'], score=float(seg['score'])
                ) for idx, seg in enumerate(segments)
            ]
            
            # Perform compression
            compressed_docs = self.compressor.compress(question, search_results)
            
            # Combine compressed text into a final document
            final_document = "\n".join(doc.text for doc in compressed_docs)
            
            # Calculate compressed tokens
            compressed_tokens = count_tokens(final_document)
            total_compressed_tokens += compressed_tokens
            
            # Create prompt (we'll store both API and local formats)
            prompt_api = format_prompt(question, final_document, None, use_api=True)
            prompt_local = format_prompt(question, final_document, None, use_api=False)
            
            # Store the compressed data
            compressed_entry = {
                'question': question,
                'compressed_context': final_document,
                'prompt_api': prompt_api,
                'prompt_local': prompt_local,
                'ground_truth': question_data.get('answer') or question_data.get('answers'),
                'original_tokens': original_tokens,
                'compressed_tokens': compressed_tokens
            }
            compressed_data.append(compressed_entry)
        
        compression_stats = {
            'total_original_tokens': total_original_tokens,
            'total_compressed_tokens': total_compressed_tokens,
            'compression_ratio': total_compressed_tokens / total_original_tokens if total_original_tokens > 0 else 0
        }
        
        return compressed_data, compression_stats

def main():
    parser = argparse.ArgumentParser(description='Context Compression Pipeline')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input JSON file with questions and contexts.')
    parser.add_argument('--method', '-m', type=str, required=True, 
                        choices=["compact", "exit", "refiner", "recomp_abstractive", "recomp_extractive", "longllmlingua"],
                        help="Compression method to use.")
    parser.add_argument('--k', '-k', type=int, default=10, help='Top k contexts to preprocess and compress.')
    parser.add_argument('--output', '-o', type=str, default=None, help='Output JSON file path. If not provided, auto-generated.')

    args = parser.parse_args()

    # Generate output path if not provided
    if args.output is None:
        args.output = generate_output_path(args.input, args.method, args.k)
    
    print(f"Output will be saved to: {args.output}")

    # Initialize pipeline
    pipeline = CompressionPipeline(method=args.method, k=args.k)

    # Load input data
    with open(args.input, 'r') as f:
        questions_data = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(questions_data)} questions from {args.input}")
    
    # Preprocess all question contexts
    all_preprocessed_segments = pipeline.preprocess_all_questions(questions_data)
    
    # Compress contexts and format prompts
    compressed_data, compression_stats = pipeline.compress_and_format(questions_data, all_preprocessed_segments)
    
    # Add metadata
    output_data = {
        'metadata': {
            'input_file': args.input,
            'method': args.method,
            'k': args.k,
            'total_questions': len(questions_data),
            'compression_ratio': compression_stats['compression_ratio'],
            'total_original_tokens': compression_stats['total_original_tokens'],
            'total_compressed_tokens': compression_stats['total_compressed_tokens']
        },
        'data': compressed_data
    }
    
    # Save compressed data
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCompression completed!")
    print(f"Total questions: {len(questions_data)}")
    print(f"Compression ratio: {compression_stats['compression_ratio']:.3f}")
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()