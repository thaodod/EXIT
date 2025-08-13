#!/usr/bin/env python3
"""
Basic Document Compression Example

A simple script showing how to compress a list of documents into a single text
using the EXIT compression framework.
"""

from compressors import SearchResult, CompActCompressor
from typing import List

def main():
    """Simple example of document compression."""
    
    print("📚 Simple Document Compression Example")
    print("=" * 45)
    
    # 1. Define your query
    query = "How do SSDs improve computer performance?"
    print(f"🔍 Query: {query}")
    
    # 2. Create a list of documents (your context)
    documents = [
        SearchResult(
            evi_id=1,
            docid=1,
            title="SSD Performance",
            text="Solid-state drives are much faster than traditional hard drives. They have no moving parts, which allows for instant data access. Boot times are significantly reduced.",
            score=1.0
        ),
        SearchResult(
            evi_id=2,
            docid=2,
            title="SSD Technology",
            text="SSDs use flash memory to store data. They consume less power than HDDs and are more reliable. The lack of mechanical components makes them quieter too.",
            score=0.9
        ),
        SearchResult(
            evi_id=3,
            docid=3,
            title="Random Topic",
            text="Today I went to the store and bought some groceries. The weather was nice and sunny. I saw a dog in the park.",
            score=0.1
        ),
        SearchResult(
            evi_id=4,
            docid=4,
            title="SSD Benefits",
            text="Applications load faster with SSDs. File transfers are quicker. Overall system responsiveness improves dramatically with SSD storage.",
            score=0.95
        )
    ]
    
    print(f"\n📄 Input: {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. {doc.title} (score: {doc.score})")
    
    # 3. Initialize compressor
    print(f"\n🔧 Initializing CompAct compressor...")
    try:
        compressor = CompActCompressor(
            model_dir='cwyoon99/CompAct-7b',
            device='cuda',  # Change to 'cpu' if no GPU
            batch_size=2
        )
        
        # 4. Compress documents
        print(f"🗜️  Compressing documents...")
        compressed_results = compressor.compress(query, documents)
        
        # 5. Get the compressed text
        compressed_text = compressed_results[0].text
        
        # 6. Show results
        print(f"\n✅ Compression complete!")
        print(f"📊 Original total length: {sum(len(doc.text) for doc in documents)} characters")
        print(f"📉 Compressed length: {len(compressed_text)} characters")
        print(f"📈 Compression ratio: {len(compressed_text) / sum(len(doc.text) for doc in documents):.2f}")
        
        print(f"\n📝 Compressed text:")
        print("-" * 40)
        print(compressed_text)
        print("-" * 40)
        
    except Exception as e:
        print(f"❌ Error during compression: {e}")
        print("\n💡 Tips:")
        print("   - Make sure you have the required models downloaded")
        print("   - Try changing device='cuda' to device='cpu' if no GPU")
        print("   - Check if all dependencies are installed")

if __name__ == "__main__":
    main()
