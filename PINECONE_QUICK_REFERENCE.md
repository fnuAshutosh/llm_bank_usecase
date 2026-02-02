# 🚀 Pinecone Integration - Quick Reference

## Installation (2 minutes)

```bash
# 1. Get API key from https://www.pinecone.io (free account)
# 2. Set in .env
export PINECONE_API_KEY="pcsk_xxxxx"

# 3. Install deps
pip install pinecone-client==3.2.0 sentence-transformers==2.2.2

# 4. Test
python test_pinecone_integration.py
```

---

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v2/search/similar` | POST | Find semantically similar messages |
| `/api/v2/search/by-intent` | POST | Find by banking intent |
| `/api/v2/search/health` | GET | Check vector DB health |
| `/api/v2/search/test` | POST | Test with sample data |

---

## Code Usage

### Search
```python
from src.services.vector_service import get_vector_service

vector_service = await get_vector_service()

# Semantic search
results = await vector_service.semantic_search(
    query="Transfer money",
    top_k=5
)
```

### Store
```python
# Store message
await vector_service.store_message(
    message_id="msg_123",
    user_message="How to transfer?",
    assistant_response="You can...",
    user_id="customer_456",
    session_id="session_789",
    intent="transfer_funds"
)
```

### Batch
```python
# Store multiple messages
successful, failed = await vector_service.batch_store_messages(messages)
```

---

## cURL Examples

### Search
```bash
curl -X POST http://localhost:8000/api/v2/search/similar \
  -H "Content-Type: application/json" \
  -d '{"query": "How to transfer money?", "top_k": 5}'
```

### Search by Intent
```bash
curl -X POST http://localhost:8000/api/v2/search/by-intent \
  -H "Content-Type: application/json" \
  -d '{"query": "Send $100", "intent": "transfer_funds", "top_k": 3}'
```

### Health Check
```bash
curl http://localhost:8000/api/v2/search/health
```

---

## Environment Variables

```env
PINECONE_API_KEY=pcsk_xxxxx
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=banking-chat-embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

---

## Files Reference

| File | Purpose |
|------|---------|
| `src/services/vector_service.py` | Vector DB service |
| `src/services/chat_service.py` | Chat with vector search |
| `src/api/routes/search.py` | Search API endpoints |
| `test_pinecone_integration.py` | Test suite |
| `PINECONE_SETUP.md` | Full setup guide |

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Key not set" | `export PINECONE_API_KEY="your-key"` |
| "Connection refused" | Verify API key in Pinecone console |
| "Index not found" | Auto-created on first use, wait 1-2 min |
| "Slow search" | First query downloads model (~1s), next are fast |

---

## Costs

- **Free tier**: 100K vectors, $0/month
- **Pro**: 1M vectors, ~$30-50/month
- **Enterprise**: 100M+ vectors, custom

Your setup uses free tier now! ✅

---

## Performance

- Embedding: 10-50ms (first load: 1s)
- Search: 50-200ms
- Storage: 100-300ms
- Batch (1000 msgs): 5-10s

---

## What's Included

✅ Semantic search  
✅ Intent filtering  
✅ Batch operations  
✅ GDPR deletion  
✅ Health checks  
✅ API endpoints  
✅ Full test suite  

---

**Next**: Run `python test_pinecone_integration.py` to verify!
