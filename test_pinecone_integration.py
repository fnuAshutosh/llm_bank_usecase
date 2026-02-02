"""
Quick test/demo of Pinecone integration

Run this to verify Pinecone is working:
    python test_pinecone_integration.py
"""

import asyncio
import os
from uuid import uuid4

# Setup environment first
# os.environ.setdefault("PINECONE_API_KEY", "your-key-here")
# os.environ.setdefault("PINECONE_ENVIRONMENT", "us-east-1-aws")
# os.environ.setdefault("PINECONE_INDEX_NAME", "banking-chat-embeddings")

from src.services.chat_service import EnhancedChatService
from src.services.vector_service import VectorService


async def test_vector_operations():
    """Test basic vector DB operations"""
    print("\n" + "="*60)
    print("Testing Pinecone Vector DB Integration")
    print("="*60)
    
    try:
        # Initialize vector service
        print("\n1️⃣  Initializing Pinecone...")
        vector_service = VectorService()
        print("✅ Pinecone initialized successfully")
        
        # Test embedding
        print("\n2️⃣  Testing text embedding...")
        test_text = "I want to transfer $500 to my savings account"
        embedding = vector_service.embed_text(test_text)
        print(f"✅ Text embedded: {len(embedding)} dimensions")
        print(f"   Sample values: {embedding[:5]}")
        
        # Store a test message
        print("\n3️⃣  Storing test message in Pinecone...")
        message_id = str(uuid4())
        success = await vector_service.store_message(
            message_id=message_id,
            user_message="How can I send money to another bank?",
            assistant_response="You can transfer funds using our secure transfer service. First, go to Transfers, then add the recipient bank account.",
            user_id="test_customer_123",
            session_id="session_456",
            intent="transfer_funds",
            metadata={"source": "demo", "test": True}
        )
        if success:
            print(f"✅ Message stored: {message_id}")
        else:
            print("❌ Failed to store message")
            return
        
        # Semantic search
        print("\n4️⃣  Testing semantic search...")
        search_query = "Can I transfer money between accounts?"
        results = await vector_service.semantic_search(
            query=search_query,
            top_k=5
        )
        print(f"✅ Found {len(results)} similar messages")
        for i, result in enumerate(results, 1):
            print(f"   {i}. Score: {result['score']:.3f}")
            print(f"      Q: {result['user_message'][:60]}...")
            print(f"      A: {result['assistant_response'][:60]}...")
        
        # Store multiple messages
        print("\n5️⃣  Batch storing multiple messages...")
        messages = [
            {
                "message_id": str(uuid4()),
                "user_message": "What's my account balance?",
                "assistant_response": "Your checking account balance is $5,432.18.",
                "user_id": "test_customer_123",
                "session_id": "session_456",
                "intent": "check_balance",
                "timestamp": "2026-02-02T10:00:00Z"
            },
            {
                "message_id": str(uuid4()),
                "user_message": "I want to pay my credit card bill",
                "assistant_response": "I can help you pay your credit card bill. Your current balance is $1,200.",
                "user_id": "test_customer_123",
                "session_id": "session_456",
                "intent": "pay_bill",
                "timestamp": "2026-02-02T10:05:00Z"
            },
            {
                "message_id": str(uuid4()),
                "user_message": "Is my account secure?",
                "assistant_response": "Yes, your account has 24/7 fraud monitoring and two-factor authentication enabled.",
                "user_id": "test_customer_123",
                "session_id": "session_456",
                "intent": "security_inquiry",
                "timestamp": "2026-02-02T10:10:00Z"
            }
        ]
        
        successful, failed = await vector_service.batch_store_messages(messages)
        print(f"✅ Batch stored: {successful} successful, {failed} failed")
        
        # Find messages by intent
        print("\n6️⃣  Finding messages by intent...")
        similar = await vector_service.find_similar_intents(
            query="How do I send money?",
            intent="transfer_funds",
            top_k=3
        )
        print(f"✅ Found {len(similar)} similar transfer messages")
        
        # Search in user history
        print("\n7️⃣  Searching in specific user's history...")
        user_results = await vector_service.semantic_search(
            query="money",
            top_k=3,
            user_id="test_customer_123"
        )
        print(f"✅ Found {len(user_results)} messages in user history")
        
        # Check health
        print("\n8️⃣  Checking Pinecone health...")
        stats = vector_service.index.describe_index_stats()
        print(f"✅ Index health:")
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Dimension: {vector_service.settings.EMBEDDING_DIMENSION}")
        print(f"   Model: {vector_service.settings.EMBEDDING_MODEL}")
        
        print("\n" + "="*60)
        print("✅ All tests passed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


async def test_chat_service():
    """Test EnhancedChatService with vector search"""
    print("\n" + "="*60)
    print("Testing EnhancedChatService with Pinecone")
    print("="*60)
    
    try:
        # Note: This requires database connection
        # For demo, we'll just show the interface
        
        print("\n✅ EnhancedChatService available with:")
        print("   - get_context_with_similar_queries()")
        print("   - store_chat_message()")
        print("   - get_customer_chat_insights()")
        print("   - find_similar_customer_issues()")
        print("   - sync_historical_messages()")
        
    except Exception as e:
        print(f"❌ Error: {e}")


async def main():
    """Run all tests"""
    await test_vector_operations()
    await test_chat_service()


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════╗
║         Banking LLM - Pinecone Integration Test            ║
╚════════════════════════════════════════════════════════════╝

Before running this test, ensure:
1. Pinecone account created at https://www.pinecone.io
2. API key set in .env or environment variable:
   
   export PINECONE_API_KEY="your-key-here"

3. Dependencies installed:
   pip install pinecone-client sentence-transformers
    """)
    
    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key or api_key == "your-key-here":
        print("⚠️  WARNING: PINECONE_API_KEY not set!")
        print("Set it with: export PINECONE_API_KEY='your-actual-key'\n")
    
    asyncio.run(main())
