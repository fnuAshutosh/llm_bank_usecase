# AI Engineer Interview - Quick Reference Cheat Sheet
**Date**: February 4, 2026 | **Role**: AI Developer & Architect

---

## 60-SECOND ELEVATOR PITCH

*"I'm an AI engineer with 6+ years backend engineering transitioning to AI/ML. I architected an enterprise Banking LLM system handling 1M+ daily transactions with <500ms p95 latency. My expertise: end-to-end AI architecture (custom transformers, QLoRA fine-tuning, RAG with Pinecone), production MLOps (Prometheus/Grafana), and enterprise security (PCI-DSS 3.2.1). Your learning platform needs exactly this: scalable, compliant, observable AI."*

---

## YOUR SUPERPOWERS (Against This Job)

| What They Want | You Have | Your Evidence |
|---|---|---|
| RAG systems | ✅ EXPERT | Pinecone integration, 15K embeddings, <100ms query latency |
| LLM fine-tuning | ✅ EXPERT | QLoRA on Banking77, 91.2% accuracy, 1.2GB memory |
| Semantic search | ✅ EXPERT | sentence-transformers, cosine similarity, caching optimization |
| Vector DBs | ✅ EXPERT | Pinecone serverless, metadata filtering, dimensional optimization |
| MLOps | ✅ EXPERT | Prometheus, Grafana, OpenTelemetry, GitHub Actions CI/CD |
| Microservices | ✅ EXPERT | FastAPI, async/await, Docker, Kubernetes, 14 services |
| Security | ✅ EXPERT | PCI-DSS 3.2.1, GDPR, SOC2, PII detection, audit logging |
| Python backend | ✅ EXPERT | FastAPI, PyTorch, PostgreSQL, Redis |

---

## 5 CORE STORIES (Memorize These!)

### Story 1: Architecture Design
**Question**: "Describe your AI architecture"
**Answer**: "Banking LLM system: custom transformer (512 d_model, 6 layers) → QLoRA fine-tuning (1.5% params, 1.2GB memory) → RAG pipeline (Pinecone, sentence-transformers) → FastAPI backend → Prometheus monitoring"

### Story 2: RAG Implementation
**Question**: "How do you implement RAG?"
**Answer**: "Embedding + retrieval + generation. Sentence-transformers for banking domain → Pinecone with metadata → 2-tier caching (L1: LRU, L2: Redis) → 70% compute reduction. <100ms p95 latency."

### Story 3: Fine-tuning Strategy
**Question**: "Why QLoRA over full fine-tuning?"
**Answer**: "Evaluated 3 approaches: full (24GB), LoRA (2.1GB), QLoRA (1.2GB). QLoRA wins: 4-bit quantization + LoRA adapters. 90.8% accuracy with 95% memory reduction. Production-ready."

### Story 4: Production Quality
**Question**: "How do you ensure production quality?"
**Answer**: "Observability first: Prometheus metrics, Grafana dashboards, OpenTelemetry tracing. Catch issues before users do. MTTR: 4+ hours → <5 minutes via alerting."

### Story 5: Crisis Management
**Question**: "Tell about handling a critical issue"
**Answer**: "PII in model output (compliance violation). Fixed: PII detection on retrieved context + audit logging + quarantine system. Deployed fix in 6 hours with zero PII incidents since."

---

## TECHNICAL GOTCHAS - BE READY

### Q: "How do you handle model drift?"
**Answer**: "Monitor accuracy on validation set using production traffic. Alert if drops >2%. Trigger retraining pipeline automatically. Gradual rollout with canary deployment."

### Q: "Why sentence-transformers vs OpenAI embeddings?"
**Answer**: "Domain-specific accuracy + cost. Banking terminology needs fine-tuned embeddings. sentence-transformers: $0 (self-hosted) vs $0.10/1K tokens. Trade-off: accuracy vs cost. For banking, domain accuracy wins."

### Q: "How do you scale to 100M queries/day?"
**Answer**: "Caching (80% hit rate) → vLLM cluster (3-4 GPU servers) → Redis cluster (100GB) → Pinecone pods → Kafka for async. P95 latency stays <500ms via caching."

### Q: "What about hallucinations?"
**Answer**: "RAG provides grounding (reduced hallucinations by 40%). PII detection prevents sensitive hallucinations. User feedback loop: correct predictions feed retraining data. Responsible AI: flag uncertain responses."

### Q: "How do you evaluate embedding quality?"
**Answer**: "Semantic similarity: retrieval precision/recall on test set. Embedding alignment: cosine similarity between synonyms should be high. Business metric: does retrieved context help user?"

---

## QUESTIONS YOU SHOULD ASK THEM

✅ "How do you currently handle model versioning and A/B testing?"  
✅ "What's your data labeling strategy for Sunbird content?"  
✅ "Are you considering model quantization for mobile inference?"  
✅ "What's your compliance framework for AI (GDPR, etc.)?"  
✅ "How do you measure business ROI on AI investments?"  

---

## NUMBERS TO MEMORIZE

| Metric | Your System | Context |
|--------|-------------|---------|
| Daily transactions | 1M+ | Scale target |
| P95 latency | <500ms | SLA target |
| Accuracy | 91.2% | Banking77 fine-tuned |
| QLoRA memory | 1.2GB | vs 24GB baseline |
| Cache hit rate | 70% | Engineering optimization |
| Compliance | PCI-DSS 3.2.1 | Level 1 achieved |
| Microservices | 14 containerized | Docker Compose |
| Query latency | <100ms | Pinecone search |
| Training time | 45s/epoch | LoRA on single GPU |
| Parameters trained | 1.5% | LoRA rank 8 |

---

## RED FLAGS TO AVOID

❌ "I built an ML model"  → ✅ "I architected a production AI system"

❌ "I use Pinecone"  → ✅ "I optimized Pinecone with 2-tier caching & metadata filtering"

❌ "I know Python"  → ✅ "I'm expert in Python for production ML (FastAPI, PyTorch, async patterns)"

❌ "I deployed on Kubernetes"  → ✅ "I designed Kubernetes with HPA, canary rollouts, multi-environment support"

❌ "I trained a model"  → ✅ "I architected end-to-end system: data pipeline, fine-tuning, evaluation, deployment, monitoring"

---

## INTERVIEW FLOW (Typical 60 min)

| Time | What | You Do |
|------|------|--------|
| 0-5 min | Intro | Deliver your 60-second pitch ✅ |
| 5-15 min | Project deep-dive | Talk about Banking LLM (architecture, RAG, MLOps) |
| 15-30 min | Technical questions | Answer Q1-Q3 from full guide |
| 30-45 min | System design | Scale your system to 10M queries |
| 45-55 min | Their questions | Ask 2-3 smart questions ✅ |
| 55-60 min | Closing | "I'm excited to work with your team" |

---

## BONUS: WHITEBOARD ARCHITECTURE (Draw This If Asked)

```
CLIENT
  ↓
LOAD BALANCER (Nginx)
  ↓
┌─────────────────────┐
│  FastAPI Backend    │ (4 instances, async)
│  - PII Detection    │
│  - Rate Limiting    │
│  - Auth             │
└────────┬────────────┘
         ↓
┌─────────────────────┐
│  Redis Cache Cluster│ (70% hit rate)
│  - L1: In-memory    │
│  - L2: Distributed  │
└────┬───────────┬────┘
     ↓           ↓
┌─────────┐  ┌──────────┐
│Pinecone │  │vLLM Clust│
│(15K vec)│  │(3 GPU    │
│<100ms   │  │servers)  │
└─────────┘  └──────────┘
     ↓
┌─────────────────────┐
│  PostgreSQL + Redis │ (Data layer)
└─────────────────────┘

MONITORING:
├── Prometheus (metrics)
├── Grafana (dashboards)
├── OpenTelemetry (tracing)
└── Audit logs (compliance)
```

---

## FINAL CHECKLIST (Night Before)

- [ ] Practice 5-minute Banking LLM story
- [ ] Memorize 5 numbers (1M, <500ms, 91.2%, 1.2GB, <100ms)
- [ ] Prepare 2 failure stories
- [ ] Know your weakness answer ("Limited scale experience, but designed for growth")
- [ ] Have 3 smart questions ready
- [ ] Dress professionally
- [ ] Get good sleep
- [ ] Arrive 10 minutes early
- [ ] **Bring a copy of your resume**
- [ ] **Bring the Banking LLM architecture diagram printed**

---

## YOU'VE GOT THIS! 💪

Your Banking LLM project is legitimately impressive. Most engineers can't show end-to-end AI systems like this. Make them understand that depth.

**Key message**: "I don't just train models. I architect systems that actually work in production."

---

**Interview Date**: _______________  
**Company**: _______________  
**Interviewer**: _______________  

**Good luck!** 🚀
