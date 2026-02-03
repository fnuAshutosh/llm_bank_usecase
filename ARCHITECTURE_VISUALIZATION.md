# Architecture Visualization & Design

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         REST API Endpoints                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   POST /api/v1/chat          GET /benchmark/summary                   │
│   {"message": "..."}         POST /model/switch/{version}              │
│                                                                         │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│               BankingLLMPipeline (Abstraction Layer)                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  - generate(prompt) → Any model                                        │
│  - Automatic benchmarking every N requests                             │
│  - JSONL logging for analysis                                          │
│  - Runtime model switching                                             │
│                                                                         │
└────────────────────────────┬──────────────────────────────────────────┘
                             │
                             ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                      ModelFactory (Router)                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Reads: LLM_MODEL_VERSION from .env                                   │
│  Returns: Appropriate adapter (singleton)                              │
│                                                                         │
└────┬────────────────┬────────────────┬────────────────────────────────┘
     │                │                │
     ↓                ↓                ↓
┌────────────────────┐  ┌─────────────────────┐  ┌──────────────────┐
│ CustomTransformer  │  │   QLoRA + vLLM      │  │   Gemini API     │
│    Adapter         │  │    Adapter          │  │   Adapter        │
├────────────────────┤  ├─────────────────────┤  ├──────────────────┤
│                    │  │                     │  │                  │
│ ModelAdapter impl  │  │ ModelAdapter impl   │  │ ModelAdapter impl│
│                    │  │                     │  │                  │
│ Your transformer   │  │ Llama 3.1-8B        │  │ Gemini 2.0 Flash │
│ 19M params        │  │ 8B params           │  │ Cloud-hosted     │
│                    │  │                     │  │                  │
│ CPU                │  │ GPU (T4)            │  │ Cloud (Google)   │
│ 195ms/token       │  │ 18ms/token          │  │ 75ms/token       │
│ 70-80% accuracy   │  │ 95% accuracy        │  │ 98% accuracy     │
│                    │  │                     │  │                  │
│ .env.custom       │  │ .env.qlora          │  │ .env.gemini      │
│                    │  │                     │  │                  │
└────────────────────┘  └─────────────────────┘  └──────────────────┘
```

---

## Configuration Flow

```
User: cp .env.qlora .env
            ↓
Pipeline starts
            ↓
Load config: LLM_MODEL_VERSION=qlora_vllm_finetuned_model_v1.0.0
            ↓
ModelFactory.get_active_model()
            ↓
Parse version string: "qlora_vllm_..." → "qlora"
            ↓
Create QLoRASambaNovaAdapter
            ↓
Health check: loads vLLM + Llama 3.1-8B
            ↓
Ready for requests
            ↓
Every 50 requests: Benchmark all 3 models
            ↓
Log to benchmark_results.jsonl
            ↓
User views results and decides next model
```

---

## Request Flow (High Level)

```
User sends: POST /api/v1/chat {"message": "What is my balance?"}
            ↓
pipeline.generate(prompt)
            ↓
self.model.generate(prompt)  ← delegates to active adapter
            ↓
Adapter returns:
{
  'text': 'Your balance is...',
  'latency_ms': 18.32,
  'tokens': 12,
  'confidence': 0.95,
  'model_version': 'qlora_vllm_finetuned_model_v1.0.0',
  'device': 'gpu'
}
            ↓
Pipeline.request_counter += 1
            ↓
if request_counter % 50 == 0:
  _run_benchmark()
            ↓
Response to user
```

---

## Benchmark Comparison Output

```
Request Count: 50

╔════════════════════╦═══════════╦═════════╦═════════╦════════════╗
║ Model              ║ Device    ║ Latency ║ P99     ║ Accuracy   ║
╠════════════════════╬═══════════╬═════════╬═════════╬════════════╣
║ custom             ║ CPU       ║ 195.2ms ║ 210ms   ║ 72%        ║
║ qlora              ║ GPU       ║  18.3ms ║  22ms   ║ 95%        ║
║ gemini             ║ Cloud     ║  75.2ms ║  92ms   ║ 98%        ║
╚════════════════════╩═══════════╩═════════╩═════════╩════════════╝

Recommendation: qlora for fastest, gemini for most accurate
```

---

## Class Hierarchy

```
ModelAdapter (Abstract Base Class)
├── Defines: generate(), get_metrics(), health_check()
│
├── CustomTransformerAdapter
│   └── Your transformer implementation
│       ├── Uses: src/llm_training/transformer.py
│       ├── Device: CPU
│       ├── Characteristics: Educational, shows fundamentals
│       └── Version: custom_model_built_0.0.0
│
├── QLoRASambaNovaAdapter
│   └── QLoRA fine-tuned model
│       ├── Uses: vLLM + Llama 3.1-8B
│       ├── Device: GPU
│       ├── Characteristics: Production, efficient, accurate
│       └── Version: qlora_vllm_finetuned_model_v1.0.0
│
└── GeminiAdapter
    └── Google Gemini API
        ├── Uses: google.generativeai
        ├── Device: Cloud
        ├── Characteristics: State-of-the-art, managed
        └── Version: gemini_2.0_flash_v1.0.0
```

---

## Design Pattern Usage

### 1. Adapter Pattern
**Problem**: Different models have different interfaces  
**Solution**: ModelAdapter base class + implementations  
**Result**: Pipeline doesn't care which model is active

```python
# Pipeline code (no model-specific logic!)
result = await self.model.generate(prompt)  # Works for ANY model
```

### 2. Factory Pattern
**Problem**: Need to create right model based on config  
**Solution**: ModelFactory handles creation logic  
**Result**: Clean separation of creation from usage

```python
# API doesn't create models
model = ModelFactory.get_active_model()  # Factory handles it
```

### 3. Singleton Pattern
**Problem**: Expensive to load model multiple times  
**Solution**: Cache instances in factory  
**Result**: Model loads once, reused for all requests

```python
# First request: creates and caches
model1 = ModelFactory.create('qlora')

# Second request: returns cached instance
model2 = ModelFactory.create('qlora')
# model1 is model2 → True (same object)
```

---

## Data Flow for Benchmarking

```
Request #1-49:
Normal processing, count requests
        ↓
Request #50:
trigger_benchmark = True
        ↓
For each model in ['custom', 'qlora', 'gemini']:
  - Create adapter instance
  - Test 3 prompts
  - Measure latency
  - Get metrics
        ↓
Aggregate results:
{
  'timestamp': '2026-02-03T15:30:45',
  'request_count': 50,
  'models': {
    'custom': {...},
    'qlora': {...},
    'gemini': {...}
  }
}
        ↓
Append to benchmark_results.jsonl
        ↓
Print summary
        ↓
Request #51-99:
Normal processing
        ↓
Request #100:
Benchmark again
        ↓
And so on...
```

---

## Environment Configuration Strategy

```
Development:
  .env → .env.custom
  → Custom model for fast iteration
  → No GPU needed
  → Quick feedback

Testing:
  .env → .env.qlora
  → Intermediate model
  → Tests production performance
  → Affordable GPU

Production:
  .env → .env.qlora (cost-effective) or .env.gemini (reliability)
  → QLoRA: High accuracy, controlled cost
  → Gemini: Highest accuracy, pay-per-token

A/B Testing:
  Switch between .env files
  → No code changes
  → Benchmark compares automatically
  → Data-driven decisions
```

---

## Error Handling & Fallback

```
Pipeline tries custom model
        ↓
If fails: Custom model error captured
        ↓
Try next: QLoRA model
        ↓
If fails: QLoRA error captured
        ↓
Try next: Gemini API
        ↓
If fails: Return error to user
        ↓
Log all errors for analysis
        ↓
Metrics track: error_rate, failure_modes
```

---

## Observability & Monitoring

```
Metrics collected per model:
├── total_requests
├── total_latency_ms
├── avg_latency_ms
├── p50_latency_ms
├── p99_latency_ms
├── total_tokens
├── error_count
├── confidence_score
└── device_utilization

Logged to:
├── In-memory (for API endpoints)
├── JSONL (for analysis)
└── Metrics endpoint (for monitoring)

Alerts:
├── If avg_latency > threshold
├── If error_rate > threshold
├── If model becomes unhealthy
└── If cache_hit_rate drops
```

---

## Deployment Scenarios

### Scenario 1: Quick Iteration
```
Developer: "Let me test a change to custom model"
Action: python -m uvicorn src.api.main:app --reload
Result: Uses custom model from .env.custom
Time: Seconds
```

### Scenario 2: Production Testing
```
DevOps: "Let's test QLoRA before full rollout"
Action: cp .env.qlora .env (or ConfigMap in K8s)
Result: Next pod restart uses QLoRA
Time: Minutes
```

### Scenario 3: Gradual Rollout
```
Product: "Roll out to 10% of users first"
Action: Use load balancer or canary deployment
Config: 10% of servers get .env.qlora, 90% keep .env.gemini
Result: Benchmark data from real production
Time: Hours/Days
```

### Scenario 4: Model Comparison
```
ML Engineer: "Which model performs best for banking queries?"
Action: Send requests → automatic benchmarks every 50 requests
Result: JSONL file with comparative metrics
Analysis: Model A vs B vs C latency/accuracy trade-off
Decision: Choose best fit for use case
```

---

## Code Organization

```
src/
├── llm/
│   ├── __init__.py
│   ├── pipeline.py              ← Main orchestrator
│   ├── model_factory.py         ← Factory pattern
│   │
│   └── models/
│       ├── __init__.py
│       ├── base_adapter.py      ← Abstract base
│       ├── custom_transformer_adapter.py
│       ├── qlora_adapter.py
│       └── gemini_adapter.py
│
├── api/
│   ├── main.py                  ← FastAPI app
│   └── chat_router.py           ← Chat endpoints
│
├── llm_training/
│   ├── transformer.py           ← Your custom model
│   ├── tokenizer.py
│   ├── train.py
│   └── lora_trainer.py
│
└── utils/
    ├── config.py                ← .env variables
    └── logger.py

.env                     ← Active config (git-ignored)
.env.custom            ← Custom model config
.env.qlora             ← QLoRA config
.env.gemini            ← Gemini config

benchmark_results.jsonl ← Benchmark history
```

---

## This Architecture = Production Grade

✅ Proven patterns (Adapter, Factory, Singleton)  
✅ Extensible (easy to add Model D, E, F)  
✅ Testable (mock adapters for unit tests)  
✅ Monitorable (metrics for each model)  
✅ Configurable (environment-based)  
✅ Observable (logging and benchmarking)  
✅ Safe (health checks, fallbacks)  
✅ Pragmatic (handles multiple use cases)

This is how real systems handle model management at scale.
