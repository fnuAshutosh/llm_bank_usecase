# Pinecone Configuration

## Getting Started with Pinecone

### 1. Create Pinecone Account
1. Go to [pinecone.io](https://www.pinecone.io)
2. Sign up (free tier available)
3. Create API key from dashboard

### 2. Get Your API Key
- Go to Pinecone console
- Navigate to "API Keys"
- Copy your API key

### 3. Set Environment Variables

```bash
# Option 1: Add to .env file
export PINECONE_API_KEY="your-api-key-here"
export PINECONE_ENVIRONMENT="us-east-1-aws"  # or your region
export PINECONE_INDEX_NAME="banking-chat-embeddings"

# Option 2: Docker/Kubernetes
docker run -e PINECONE_API_KEY="your-key" ...
```

### 4. Configuration in .env

```env
# Pinecone Vector Database
PINECONE_API_KEY=pcsk_xxxxxxxxxxxxx
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=banking-chat-embeddings

# Embedding Model (HuggingFace)
EMBEDDING_MODEL=all-MiniLM-L6-v2
EMBEDDING_DIMENSION=384
```

## Free Tier Limits

- **Index**: 1 index
- **Storage**: 100,000 vectors
- **Queries**: 25,000/month
- **Updates**: 10,000/month

## Production Tier

- **Index**: Unlimited
- **Storage**: Unlimited (pay per vector)
- **Queries**: Unlimited
- **Updates**: Unlimited

## Cost Estimation

| Scale | Vectors | Monthly Cost |
|-------|---------|--------------|
| Dev | 10K | $0 (free tier) |
| Test | 100K | $0 (free tier) |
| Small | 1M | ~$30-50 |
| Medium | 10M | ~$300-500 |
| Large | 100M | ~$3,000-5,000 |

## API Endpoints

### Semantic Search
```bash
curl -X POST http://localhost:8000/api/v2/search/similar \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I transfer money?",
    "top_k": 5
  }'
```

### Find Similar Intents
```bash
curl -X POST http://localhost:8000/api/v2/search/by-intent \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Send $100 to John",
    "intent": "transfer_funds",
    "top_k": 3
  }'
```

### Health Check
```bash
curl http://localhost:8000/api/v2/search/health
```

### Test Vector DB
```bash
curl -X POST http://localhost:8000/api/v2/search/test
```

## Documentation

- [Pinecone Docs](https://docs.pinecone.io)
- [Sentence Transformers](https://www.sbert.net)
- [Semantic Search Guide](https://docs.pinecone.io/guides/learning/semantic-search)

## Troubleshooting

### "PINECONE_API_KEY not set"
- Ensure API key is in `.env` file
- Run: `echo $PINECONE_API_KEY`
- Restart API after setting env var

### "Connection refused"
- Check Pinecone service status
- Verify API key is valid
- Try test endpoint: `/api/v2/search/test`

### "Index not found"
- Service will auto-create index on first use
- Wait 1-2 minutes for creation
- Check Pinecone console for index status

### Slow searches
- Ensure embedding model is downloaded (~400MB)
- First query downloads model (takes 30-60s)
- Subsequent queries are fast (~50ms)

## Production Checklist

- [ ] API key stored in secure vault (not git)
- [ ] Index configured with appropriate replication
- [ ] Monitoring enabled for query latency
- [ ] Backup strategy documented
- [ ] Cost tracking enabled
- [ ] Rate limiting configured
- [ ] Auto-scaling tested
