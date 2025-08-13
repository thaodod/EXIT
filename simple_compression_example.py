#!/usr/bin/env python3

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

def create_sample_documents() -> List[SearchResult]:
    """Create sample documents for demonstration."""
    documents = [
        SearchResult(
            evi_id=1,
            docid=1,
            title="Solid State Drives Overview",
            text="""Solid-state drives (SSDs) are storage devices that use flash memory to store data persistently. 
            Unlike traditional hard disk drives (HDDs), SSDs have no moving mechanical parts, which makes them more 
            reliable and faster. SSDs use NAND flash memory cells to store data electronically.""",
            score=0.95
        ),
        SearchResult(
            evi_id=2,
            docid=2,
            title="SSD Performance Benefits",
            text="""SSDs provide significant performance improvements over traditional hard drives. They offer faster 
            boot times, quicker application loading, and improved overall system responsiveness. The access time for 
            SSDs is typically under 1 millisecond, compared to 5-10 milliseconds for HDDs.""",
            score=0.90
        ),
        SearchResult(
            evi_id=3,
            docid=3,
            title="SSD Technology Details",
            text="""Modern SSDs use various types of NAND flash memory, including SLC, MLC, TLC, and QLC. 
            They connect to computers via interfaces like SATA, PCIe, or M.2. SSDs consume less power than HDDs 
            and generate less heat, making them ideal for laptops and mobile devices.""",
            score=0.85
        ),
        SearchResult(
            evi_id=4,
            docid=4,
            title="Computing History",
            text="""The first computers were built in the 1940s and used vacuum tubes. Personal computers became 
            popular in the 1980s. The internet was invented in the late 20th century. Today, we have smartphones 
            and cloud computing. Weather prediction has also improved significantly.""",
            score=0.30
        ),
        SearchResult(
            evi_id=5,
            docid=5,
            title="SSD vs HDD Comparison",
            text="""When comparing SSDs to HDDs, SSDs win in speed, reliability, and power consumption. However, 
            HDDs typically offer more storage capacity per dollar. SSDs have become much more affordable in recent 
            years, making them the preferred choice for most users seeking better performance.""",
            score=0.88
        )
    ]
    return documents

def compress_with_compact(query: str, documents: List[SearchResult]) -> str:
    """Compress documents using CompAct method."""
    print("🔄 Using CompAct Compressor...")
    try:
        compressor = CompActCompressor(
            model_dir='cwyoon99/CompAct-7b',
            device='cuda',
            batch_size=3
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

def main():
    """Main function demonstrating document compression."""
    print("📚 EXIT Document Compression Example")
    print("=" * 50)
    
    # Define query and create sample documents
    query = "How do solid-state drives improve computer performance?"
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
    print("🗜️  COMPRESSION RESULTS")
    print("=" * 50)
    
    # Dictionary of compression methods
    compression_methods = {
        "CompAct": compress_with_compact,
        "EXIT": compress_with_exit,
        "Refiner": compress_with_refiner,
        "Recomp (Abstractive)": compress_with_recomp_abstractive,
        "Recomp (Extractive)": compress_with_recomp_extractive,
        "LongLLMLingua": compress_with_longllmlingua,
    }
    
    # Try each compression method
    results = {}
    for method_name, compress_func in compression_methods.items():
        print(f"\n🔹 {method_name} Compression:")
        print("-" * 40)
        try:
            compressed_text = compress_func(query, documents)
            results[method_name] = compressed_text
            print(f"✅ Compressed text ({len(compressed_text)} chars):")
            print(f"   {compressed_text}")
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
    # Run the full example
    main()
