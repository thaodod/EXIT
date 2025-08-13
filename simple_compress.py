#!/usr/bin/env python3

import argparse
from compressors import (
    SearchResult, 
    CompActCompressor,
    EXITCompressor,
    RefinerCompressor,
    RecompAbstractiveCompressor,
    RecompExtractiveCompressor,
    LongLLMLinguaCompressor
)
from typing import List
from utils import create_sample_documents

def compress_with_compact(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using CompAct method."""
    print("🔄 Using CompAct Compressor...")
    try:
        compressor = CompActCompressor(
            model_dir='cwyoon99/CompAct-7b',
            device='cuda',
            batch_size=4
        )
        compressed = compressor.compress(query, documents)
        return compressed[0].text if compressed else "No compression result"
    except Exception as e:
        return f"CompAct compression failed: {str(e)}"

def compress_with_exit(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using EXIT method."""
    print("🔄 Using EXIT Compressor...")
    try:
        compressor = EXITCompressor(
            checkpoint="doubleyyh/exit-gemma-2b",
            device='cuda'
        )
        compressed = compressor.compress(query, documents)
        return compressed[0].text if compressed else "No compression result"
    except Exception as e:
        return f"EXIT compression failed: {str(e)}"

def compress_with_refiner(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using Refiner method."""
    print("🔄 Using Refiner Compressor...")
    try:
        compressor = RefinerCompressor()
        compressed = compressor.compress(query, documents)
        return compressed[0].text if compressed else "No compression result"
    except Exception as e:
        return f"Refiner compression failed: {str(e)}"

def compress_with_recomp_abstractive(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using Recomp Abstractive method."""
    print("🔄 Using Recomp Abstractive Compressor...")
    try:
        compressor = RecompAbstractiveCompressor()
        compressed = compressor.compress(query, documents)
        return compressed[0].text if compressed else "No compression result"
    except Exception as e:
        return f"Recomp Abstractive compression failed: {str(e)}"

def compress_with_recomp_extractive(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using Recomp Extractive method."""
    print("🔄 Using Recomp Extractive Compressor...")
    try:
        compressor = RecompExtractiveCompressor()
        compressed = compressor.compress(query, documents)
        return compressed[0].text if compressed else "No compression result"
    except Exception as e:
        return f"Recomp Extractive compression failed: {str(e)}"

def compress_with_longllmlingua(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using LongLLMLingua method."""
    print("🔄 Using LongLLMLingua Compressor...")
    try:
        compressor = LongLLMLinguaCompressor()
        compressed = compressor.compress(query, documents)
        return compressed[0].text if compressed else "No compression result"
    except Exception as e:
        return f"LongLLMLingua compression failed: {str(e)}"

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document compression example")
    parser.add_argument("--method", "-m", type=str, required=True,
                        choices=["compact", "exit", "refiner", "recomp_abstractive", 
                                "recomp_extractive", "longllmlingua"],
                        help="Compression method to use")
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    method = args.method
    
    # Use a fixed query (will be replaced with JSON input in the future)
    query = "How do solid-state drives improve computer performance?"
    
    # Create sample documents
    documents = create_sample_documents()
    
    print(f"\n❓ Query: {query}")
    print(f"\n📄 Number of input documents: {len(documents)}")
    
    # Show original document content
    print("\n📋 Original Documents:")
    for i, doc in enumerate(documents, 1):
        print(f"\n{i}. {doc.title}")
        print(f"   Score: {doc.score}")
        print(f"   Text: {doc.text[:100]}...")
    
    print("\n" + "=" * 50)
    print("COMPRESSION RESULTS")
    print("=" * 50)
    
    # Dictionary of compression methods
    compression_methods = {
        "compact": ("CompAct", compress_with_compact),
        "exit": ("EXIT", compress_with_exit),
        "refiner": ("Refiner", compress_with_refiner),
        "recomp_abstractive": ("Recomp (Abstractive)", compress_with_recomp_abstractive),
        "recomp_extractive": ("Recomp (Extractive)", compress_with_recomp_extractive),
        "longllmlingua": ("LongLLMLingua", compress_with_longllmlingua),
    }
    
    # Run only the selected compression method
    results = {}
    method_name, compress_func = compression_methods[method]
    print(f"\n🔹 {method_name} Compression:")
    print("-" * 40)
    try:
        compressed_text = compress_func(query, documents)
        results[method_name] = compressed_text
        print(f"✅ Compressed text ({len(compressed_text)} chars):")
        print(f"{compressed_text}")
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        results[method_name] = error_msg
        print(error_msg)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 COMPRESSION SUMMARY")
    print("=" * 50)
    
    original_length = sum(len(doc.text) for doc in documents)
    print(f"Original total length: {original_length} characters")
    
    for method, result in results.items():
        if not result.startswith("❌") and not result.endswith("failed"):
            compression_ratio = len(result) / original_length
            print(f"{method}: {len(result)} chars (ratio: {compression_ratio:.2f})")
        else:
            print(f"{method}: Failed")

if __name__ == "__main__":
    # Run the example with command line arguments
    main()
