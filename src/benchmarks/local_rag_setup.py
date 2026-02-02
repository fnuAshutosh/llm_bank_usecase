"""
Local Benchmark Setup Script
---------------------------
Purpose: Lightweight script to verify local environment and Vector DB state 
before running the heavy benchmark on Colab.

Usage:
    python src/benchmarks/local_rag_setup.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.vector_service import VectorService
import asyncio

async def check_readiness():
    print("🔍 Checking Local Benchmarking Readiness...\n")
    
    # 1. Check Credentials
    pinecone_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_key:
        print("❌ PINECONE_API_KEY is missing in environment!")
        return
    print("✅ Pinecone API Key found.")
    
    # 2. Check Vector DB Stats
    try:
        vs = VectorService()
        stats = vs.index.describe_index_stats()
        count = stats.get('total_vector_count', 0)
        print(f"✅ Connected to Pinecone Index: {vs.index_name}")
        print(f"📊 Current Vector Count: {count}")
        
        if count < 100:
            print("⚠️  Warning: Index has very few vectors. Benchmark might not be meaningful.")
            print("   Action: Run the Colab 'Training/Ingestion' notebook first to populate DB.")
        else:
            print("✅ Index populated and ready for benchmarking.")
            
    except Exception as e:
        print(f"❌ Failed to connect to Pinecone: {e}")

    print("\n👉 NEXT STEP:")
    print("Upload 'src/benchmarks/Colab_RAG_Benchmark.ipynb' to Google Colab to run the heavy evaluation.")

if __name__ == "__main__":
    asyncio.run(check_readiness())
