# Chapter 4: Inference Optimization & Deployment

## What We Built

### 1. Inference Engine
- **Model Loading**: Checkpoint restoration (full precision)
- **Generation Strategy**: Autoregressive text generation
- **Sampling Methods**: Temperature, top-k, top-p
- **Performance**: ~1-5 seconds per generation on CPU

### 2. Interactive Interface
- **Chat Loop**: Continuous conversation
- **Streaming**: Word-by-word or batch output
- **Context Management**: Remember conversation history
- **Error Handling**: Graceful failures

### 3. Optimization Goals
- **Latency**: Reduce time-to-first-token (TTFT)
- **Throughput**: Generate multiple responses in parallel
- **Memory**: Keep model loaded, minimize allocation
- **Cost**: CPU inference vs GPU cost trade-off

---

## Inference Architecture

```
User Input
    ↓
Tokenize (input_ids)
    ↓
Model Forward Pass (inference mode)
    ↓
Sampling (temperature/top-k/top-p)
    ↓
Detokenize (string output)
    ↓
Display to User
    ↓
(Feedback for logging/monitoring)
```

---

## Generation Strategies Comparison

| Strategy | Speed | Quality | Determinism | Use Case |
|----------|-------|---------|------------|----------|
| **Greedy** | ⚡⚡⚡ | Poor | 100% | Speed testing, debug |
| **Top-K** | ⚡⚡ | Good | Tunable | Production-grade |
| **Top-P** | ⚡⚡ | Excellent | Tunable | Creative responses |
| **Beam Search** | 🐌 | Best | Tunable | Accuracy-critical apps |
| **Temperature Only** | ⚡⚡ | Medium | Tunable | Baseline |

---

## Interview Questions & Answers (STAR Format)

### Q1: "Walk us through the inference pipeline step-by-step. From input text to output tokens."

**STAR Response**:

**Situation**: 
User enters prompt "What is my account balance?" and expects model response in 2-5 seconds. Needed to understand the full inference pipeline to explain bottlenecks and optimize latency.

**Task**: 
Trace a complete inference request through our banking LLM.

**Action**:

**Step 1: Model Load** (happens once)
```python
# Load from checkpoint
model = BankingLLM(config)
model.load_state_dict(torch.load('checkpoint.pt'))
model.eval()  # Switch to evaluation mode
model.to('cpu')

# Memory allocated
# - Model weights: 19.24M params × 4 bytes (float32) = 77MB
# - Buffers (layer norms): ~5MB
# Total: ~82MB resident
```

**Step 2: User Input Received**
```
Raw input: "What is my account balance?"
Length: 33 characters
```

**Step 3: Tokenization**
```python
# Convert string to token IDs
input_text = "What is my account balance?"

tokens = tokenizer.encode(input_text)
# Output: [2, 532, 45, 189, 402, 782, 3]
#          BOS  What is my account balance EOS

input_ids = torch.tensor([tokens])  # shape: [1, 7]
```

**Step 4: Padding** (optional, for our single sample)
```
input_ids: [1, 7]
Max sequence: 512
Padding needed: 512 - 7 = 505 tokens

padded_ids: [1, 512]
mask:       [1, 512]  (all 1s for input, 0s for padding)
```

**Step 5: Forward Pass Through Model**
```
Step 5a: Token Embedding
    [1, 512] → [1, 512, 512]
    Each token ID converted to 512-dim vector

Step 5b: Position Embedding
    Add positional information (which token position)
    Still [1, 512, 512]

Step 5c: Transformer Blocks (6 layers)
    Layer 1: Attention (each token "reads" previous tokens)
    Layer 1: Feed-forward (element-wise MLP)
    Layer 2-6: Repeat
    Output: [1, 512, 512]

Step 5d: Layer Norm
    Normalize to stable range
    [1, 512, 512]

Step 5e: LM Head (projection)
    Project 512 → 119 (vocab size)
    [1, 512, 119]
    
logits[0, 6, :] = [0.15, -0.83, 0.42, ..., -0.12]
                   (scores for next token after EOS)
```

**Step 6: Generate First Token**
```python
# Get logits for last position (end of input)
next_token_logits = logits[0, -1, :]  # [119]

# Apply sampling strategy
with torch.no_grad():
    if strategy == "greedy":
        next_token = torch.argmax(next_token_logits)
    elif strategy == "top_k":
        # Keep only top-k tokens
        top_k = 10
        topk_logits, topk_indices = torch.topk(
            next_token_logits, k=top_k
        )
        # Sample from top-k
        probs = torch.softmax(topk_logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        next_token = topk_indices[idx]

# Result: next_token = 15 (token ID)
# Text: "agent"
```

**Step 7: Autoregressive Loop** (generate more tokens)
```python
generated_tokens = []
max_new_tokens = 50

for step in range(max_new_tokens):
    # Append newly generated token
    input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
    
    # Forward pass again (KV cache would skip this for old tokens)
    logits = model(input_ids, attention_mask)
    
    # Sample next token
    next_token_logits = logits[0, -1, :]
    next_token = sample(next_token_logits, strategy)
    
    generated_tokens.append(next_token)
    
    # Stop if EOS token
    if next_token == EOS_TOKEN:
        break

# Result: [agent_token, colon_token, your_token, ..., balance_amount]
```

**Step 8: Detokenization**
```python
token_ids = [15, 58, 200, 45, 189, 402, 782, 8, 3]
decoded = tokenizer.decode(token_ids)

# Output: "agent: your account balance is $ 2450.32"
```

**Step 9: Return to User**
```
Display on screen in ~2 seconds total:
  "agent: your account balance is $ 2450.32"
```

**Performance Breakdown** (CPU, Codespaces):
```
1. Model load: 500ms (first request only)
2. Tokenization: 5ms
3. Padding: 0ms (in-memory)
4. Forward pass (one): 200ms
5. Sampling: 5ms
6. Generate 30 tokens (30× forward): 6000ms
7. Detokenization: 5ms
8. Display: 0ms
─────────────────────────────
Total: ~6700ms (6.7 seconds)

Per-token latency: 200ms (forward) + 5ms (sample) = 205ms
```

**Result**:
- **Per-token time**: 205ms on CPU (reasonable for inference)
- **First token time**: 200ms (low)
- **Throughput**: ~5 tokens/second
- **Bottleneck**: Forward pass (95% of time)
- **Optimization opportunity**: KV cache, quantization

---

### Q2: "What is KV cache in attention? Why does it matter for inference?"

**STAR Response**:

**Situation**: 
During inference, notice we recompute the entire forward pass for each new token generated. This seems wasteful: we already computed attention for previous tokens. Generating 50 tokens requires 50× full forward passes. Wanted to understand if we could reuse computation.

**Task**: 
Understand KV caching in attention and its impact on inference speed.

**Action**:

**Without KV Cache** (naive, what we currently do):

```
Generate token 1:
    Forward pass on sequence [BOS, "What", "is", "my", "account"]
    Compute attention for all 5 tokens
    Attention matrix: [5, 5]  (each token attends to all 5)
    Output: logits for position 5
    
Generate token 2:
    Forward pass on sequence [BOS, "What", "is", "my", "account", 
                             "agent", ...]
    Compute attention for all 6 tokens (including new token)
    But positions 0-4 attention is IDENTICAL to step 1!
    Wasteful: recomputing 83% of work
```

**Attention Mechanism Recap**:
```
Q = input @ W_q          [seq_len, dim]
K = input @ W_k          [seq_len, dim]
V = input @ W_v          [seq_len, dim]

attention_scores = Q @ K.T / sqrt(dim)   [seq_len, seq_len]
attention_weights = softmax(scores)

output = attention_weights @ V           [seq_len, dim]
```

**Key Insight**: 
- K and V only depend on input sequence
- During generation, input sequence only APPENDS tokens
- Previous K,V values never change
- **We can cache them!**

**With KV Cache** (optimization):

```
Generate token 1:
    Forward pass: [BOS, "What", "is", "my", "account"]
    Cache K,V values: [[k0,v0], [k1,v1], ..., [k4,v4]]
    Total cache: 5 * (dim=512) * 2 values * 4 bytes = 20.5KB
    
Generate token 2:
    New token: "agent"
    Compute K,V for NEW token only: [k5, v5]
    Append to cache: cache += [k5, v5]
    
    For attention:
    Q_new = new_token @ W_q              (compute)
    K_all = [cached K_values] + [k5]     (reuse + append)
    V_all = [cached V_values] + [v5]     (reuse + append)
    
    attention_scores = Q_new @ K_all.T   (now fast!)
    
    Result: 95% faster! (only compute for new token)
```

**Mathematical Comparison**:

```
Without KV Cache:
    Per token: seq_len * hidden * hidden operations
    Token 30: 512 * 512 * 512 = 134M operations
    
With KV Cache:
    Per token: hidden * hidden operations (only new Q·K computation)
    Token 30: 512 * 512 = 262K operations
    
Speedup: 134M / 262K = 512× faster!
    (approximately seq_len speedup)
```

**Memory Trade-off**:

```
Without cache:
    Memory: model weights (82MB) + activations (varies)
    
With cache:
    Memory: model weights (82MB) 
            + all K,V values [seq_len, hidden, num_layers]
            + 6 layers * 512 hidden * 512 seq_len * 4 bytes * 2(K,V)
            ≈ 82MB + 12.5MB = 94.5MB
    
Trade-off: +12.5MB memory → 512× speedup ✓ (excellent)
```

**Our Implementation** (currently without cache):

```python
# Current: Recompute everything
for i in range(num_tokens):
    logits = model(input_ids)  # Recomputes positions 0-i
    next_token = sample(logits[-1])
    input_ids = append(input_ids, next_token)
```

**Optimized Implementation** (with KV cache):

```python
# Optimized: Cache K,V values
kv_cache = None

for i in range(num_tokens):
    logits, kv_cache = model.forward_with_cache(
        input_ids[-1:],      # Only last token
        kv_cache=kv_cache    # Reuse previous K,V
    )
    next_token = sample(logits[-1])
    input_ids = append(input_ids, next_token)
```

**Result**:
- **Speed improvement**: 512× faster (for long sequences)
- **Memory overhead**: +12.5MB (negligible)
- **Implementation complexity**: Medium (requires attention changes)
- **Current bottleneck**: Being resolved by KV cache
- **Lesson**: Recognize redundant computation, cache selectively

---

### Q3: "We generate at 5 tokens/second. What's the minimum latency we can achieve?"

**STAR Response**:

**Situation**: 
Current generation speed: 5 tokens/second on CPU (205ms per token). User wants realistic latency goals for production. Wondering if we can reach 10, 50, or 100 tokens/second.

**Task**: 
Identify bottlenecks and estimate achievable latency with various optimizations.

**Action**:

**Current Latency Breakdown** (CPU, Codespaces):
```
Per token costs:
    Forward pass:           195ms (95% of time)
    Sampling:               5ms
    Tokenization:           5ms (first token only)
    Memory allocation:      ~0ms
    ──────────────────────
    Total per token:        ~205ms
    ──────────────────────
    Throughput:             4.9 tokens/second
```

**Bottleneck Analysis**:

```
Forward pass = matrix multiplications
    19.24M parameters × 2 ops each = 38.5 GFLOP per token
    
CPU speed: ~10 GFLOP/sec (modern CPU)
    Expected: 38.5 GFLOP / 10 GFLOP/s = 3.85 seconds per token
    
Actual: 0.195 seconds per token
    Reason: CPU parallelism, cache optimization, MKL library
```

**Optimization Strategy 1: KV Cache** (applies here)
```
Current: 195ms per token (full recompute)
With cache: 195ms / 512 ≈ 0.38ms (attention only on new token)

Reality: ~10ms per token
    (K,V computation still non-trivial, not just matmul)
```

**Optimization Strategy 2: Quantization** (4-bit)
```
Theory:
    FP32: 1 operation per value
    INT8: ~4 operations per value (requires conversion)
    INT4: ~8 operations, but memory halves
    
    Net: ~4× speedup possible
    New latency: 195ms / 4 = 48ms per token
    New throughput: 20 tokens/second
```

**Optimization Strategy 3: Tensor Parallelism** (multi-GPU)
```
Theory:
    1 GPU: 195ms
    4 GPUs: 195ms / 4 = 48ms (perfect scaling)
    
Reality: ~80% scaling efficiency (communication overhead)
    4 GPUs: 195ms / 3.2 = 61ms
    New throughput: 16 tokens/second
```

**Optimization Strategy 4: Batch Inference** (multiple users)
```
Single request: 195ms per token
10 requests simultaneously (batch):
    Computation: 195ms × (19M / 10) ≈ 195ms same
    (GPU fully utilized by batch, amortized)
    
    Per-user latency: Still 195ms
    Per-user throughput: 50 tokens/sec (batch amplification)
```

**Combined Optimizations:**

| Technique | Speedup | Cumulative | Latency |
|-----------|---------|-----------|---------|
| **Baseline** | - | 1× | 195ms |
| **KV Cache** | 19× | 19× | 10ms |
| **Quantization** | 4× | 76× | 2.5ms |
| **GPU (A100)** | 200× | 15,200× | 0.013ms |
| **Batch (100)** | 100× | 1,520,000× | 0.13μs (theoretical) |

**Realistic Targets:**

**Conservative** (KV cache only):
```
Latency: ~10ms/token
Throughput: 100 tokens/second
Device: Current CPU
Time to response: 50 tokens × 10ms = 500ms ✓ (good)
```

**Moderate** (KV cache + quantization):
```
Latency: ~2ms/token
Throughput: 500 tokens/second
Device: GPU (T4 or V100)
Time to response: 50 tokens × 2ms = 100ms ✓ (excellent)
```

**Optimized** (all techniques + batch):
```
Latency: ~0.5ms/token per user
Throughput: 2000 tokens/second (system)
Device: A100 GPU with 100 concurrent users
Time to response: 50 tokens × 0.5ms = 25ms ✓ (ultra-low)
```

**Why Not Reach Theoretical Max?**:
```
Reasons:
1. Communication overhead (PCIe/NVLink)
2. Scheduler overhead (context switching)
3. Memory bandwidth limits (higher than compute limits)
4. Numerical precision trade-offs
5. Batching overhead (synchronization)

Rule of thumb:
    Theoretical × 0.5-0.8 = Practical performance
```

**Result**:
- **Current speed**: 5 tokens/second (baseline CPU)
- **With KV cache**: 100 tokens/second (19× speedup)
- **With GPU + quant**: 500+ tokens/second (100× speedup)
- **Production target**: 100-1000 tokens/second (depends on SLA)
- **Lesson**: Multiple small optimizations > one large one

---

### Q4: "Should we batch inference requests? What are the trade-offs?"

**STAR Response**:

**Situation**: 
Single user gets fast response (~2 seconds). But what if 10 users request simultaneously? Should we queue them and process as a batch, or serve each immediately? Need to understand batch inference trade-offs.

**Task**: 
Analyze batching strategy for multi-user inference workload.

**Action**:

**Scenario A: No Batching** (serve immediately)

```
Requests arrive:
    T=0ms: User1 requests
    T=10ms: User2 requests
    T=20ms: User3 requests
    
Execution:
    T=0-2000ms: Process User1 (no batching)
        Generate 50 tokens, 5 tok/sec = 2000ms
    T=2000-4000ms: Process User2
    T=4000-6000ms: Process User3
    
Results:
    User1 latency: 2000ms ✓ (fast)
    User2 latency: 2010ms + 2000ms = 4010ms (slow!)
    User3 latency: 2020ms + 4000ms = 6020ms (very slow!)
    Average latency: (2000 + 4010 + 6020) / 3 = 4010ms ✗
```

**Scenario B: Batch Processing** (wait 100ms, then batch)

```
Requests collected:
    T=0-100ms: Buffer incoming requests
    User1, User2, User3 ready
    
Batched execution:
    T=100-2100ms: Process batch of 3
        Forward pass: same computation
        (GPU processes 3× data simultaneously)
        Per-token latency: ~195ms / 3 ≈ 65ms
        Generate 50 tokens: 50 × 65ms = 3250ms
    
Results:
    User1 latency: 100ms (wait) + 3250ms (compute) = 3350ms
    User2 latency: 100ms (wait) + 3250ms (compute) = 3350ms
    User3 latency: 100ms (wait) + 3250ms (compute) = 3350ms
    Average latency: 3350ms (lower variance!) ✓
    System throughput: 3 × 50 tokens / 3.35s = 45 tok/s ✓ (3.3× better)
```

**Batching Mathematics**:

```
Single request:
    Time per token: T_single
    Tokens per request: N
    Latency: N × T_single

Batch of B requests:
    Time per token: T_single / B (amortized)
    Tokens per request: N
    Latency: N × (T_single / B) + Wait_time
          = N × T_single / B + W

Throughput improvement: B × (N × T_single) / (N × T_single / B + W)
                       = B² × T_single / (T_single + W×B)
```

**Trade-off Analysis**:

| Factor | No Batching | Batching |
|--------|-------------|----------|
| **Latency (p50)** | 2000ms | 3100ms (55% higher) |
| **Latency (p99)** | 6000ms | 3150ms (47% lower!) |
| **Throughput** | 0.5 users/sec | 3 users/sec (6× better) |
| **Variance** | High | Low |
| **Fairness** | Yes (FIFO) | Moderate (batches) |
| **Complexity** | Low | High |

**Optimal Batch Settings**:

```
If wait tolerance = 100ms:
    Max batch size = requests_in_100ms
    
If throughput critical (chatbot):
    Larger batches = better throughput
    Batch size: 32-128 (GPU dependent)
    
If latency critical (interactive):
    Smaller batches = lower p99 latency
    Batch size: 1-8
    Wait timeout: 50ms
```

**Real-World Deployment**:

```
Strategy: Dynamic batching
    - Wait up to 50ms or batch_size=32
    - Whichever comes first
    - Prioritize user requests by waiting time
    
Results:
    Average latency: 1500ms (50ms wait + fast compute)
    p99 latency: 2000ms (max 32 requests)
    Throughput: 25 requests/sec (good)
    Fairness: Good (not all users wait equally)
```

**Our Banking LLM** (current strategy):

```python
# Current: Single request
response = model.generate(prompt, max_tokens=50)

# Better: Batch if multiple requests queued
async def batch_generate(requests):
    wait_time = 0
    batch = []
    
    while len(batch) < MAX_BATCH_SIZE and wait_time < 50ms:
        if request_queue.has_items():
            batch.append(request_queue.get_nowait())
        else:
            await asyncio.sleep(5ms)
            wait_time += 5ms
    
    # Process batch
    responses = model.batch_generate(batch)
    return responses
```

**Result**:
- **No batching**: Low latency (p50), but poor throughput
- **With batching**: Higher latency (p50), but 6× throughput
- **Optimal**: Dynamic batching with timeout
- **For our scale**: Batching worth it if >2 concurrent users
- **Lesson**: Throughput vs latency trade-off; choose wisely

---

### Q5: "What metrics should we monitor in production? How do we know if inference is degrading?"

**STAR Response**:

**Situation**: 
Deployed model to production. Users report "sometimes slow, sometimes fast" but no clear visibility into what's happening. Need metrics to monitor system health and catch regressions early.

**Task**: 
Define key inference metrics and monitoring strategy for production.

**Action**:

**Core Metrics to Track**:

**1. Latency Metrics** (critical for user experience)

```
Latency = time from request → first token output

Percentiles to track:
    p50 (median):    50% of requests faster
    p99 (tail):      99% of requests faster (catches outliers)
    p999 (extreme):  Catches catastrophic failures
    
For our system:
    p50:  2.0 seconds (target)
    p99:  5.0 seconds (SLA)
    p999: 15.0 seconds (alert)
    
Tracking:
    latencies = [2.1, 1.9, 2.0, 15.2, 2.0, ...]
    p50 = 2.0
    p99 = 15.2 (alert triggered!)
```

**2. Throughput Metrics** (system capacity)

```
Throughput = tokens generated per second

Track:
    Tokens/sec (total system)
    Requests/sec (concurrent users)
    Avg tokens per request
    
For our system:
    20 tokens/sec baseline
    If drops to 10 tokens/sec → bottleneck detected
    
Calculation:
    Throughput = (total_tokens_generated) / (elapsed_time)
    
Tracking:
    rollover each minute
    report = "540 tokens in 27 seconds = 20 tok/s"
```

**3. Resource Utilization** (hardware efficiency)

```
CPU usage:
    Should be 80-90% during inference
    <50% = underutilized (inefficient)
    >95% = overloaded (risk of timeouts)
    
GPU memory:
    Should be 60-80% of capacity
    >90% = OOM risk
    
System memory:
    Monitor for memory leaks
    Should stay stable over hours
    
Tracking:
    cpu_percent = (cpu_time / wall_time) × 100
    memory_mb = process.memory_info().rss / 1024**2
```

**4. Model-Specific Metrics**

```
Cache hit rate (if using KV cache):
    Measure: sequences reusing cached K,V
    Target: >95%
    Low rate = wasted computation
    
Token prediction quality:
    Measure: confidence of predicted token
    logit_score = max(softmax(logits))
    Low confidence (<0.5) = uncertain predictions
    
Input token distribution:
    Measure: actual vocab usage vs training dist
    Detect: distribution shift (new queries)
    Action: retrain if distribution changes >20%
    
Tracking:
    cache_hits = 95%
    avg_confidence = 0.78
    max_new_vocab_rate = 0.05
```

**5. Error Metrics**

```
Error rate:
    rate = errors / total_requests
    Target: <0.1%
    
Error types:
    OOM errors: memory limit exceeded
    Timeout errors: request took >SLA
    NaN/Inf errors: numerical instability
    
Tracking:
    errors_per_minute: [0, 0, 1, 0, 0, 2, ...]
    error_types = {"timeout": 5, "oom": 0, "nan": 1}
```

**Dashboard Layout** (what to monitor):

```
┌─────────────────────────────────┐
│ Production Inference Dashboard  │
├─────────────────────────────────┤
│                                 │
│ Latency                         │
│   p50: 2.1s  ✓                 │
│   p99: 4.8s  ✓                 │
│   p999: 12s  ⚠                 │
│                                 │
│ Throughput: 19 tok/s  ✓        │
│ Requests: 8 concurrent         │
│                                 │
│ Resources                       │
│   CPU: 87% ✓                   │
│   Memory: 1.2GB / 4GB          │
│   Errors: 1 in 10min  ⚠        │
│                                 │
│ Model Health                    │
│   Avg confidence: 0.79  ✓      │
│   Cache hit rate: 94%  ⚠       │
│   New vocab rate: 0.08  ⚠      │
│                                 │
└─────────────────────────────────┘
```

**Alerting Thresholds**:

```
Critical (page on-call):
    ✓ p99 latency > 10s
    ✓ Error rate > 1%
    ✓ Memory > 90%
    ✓ Consecutive errors > 5
    
Warning (notify team):
    ⚠ p99 latency > 5s
    ⚠ Throughput < 15 tok/s (30% drop)
    ⚠ Error rate > 0.5%
    ⚠ Cache hit rate < 90%
    
Info (log only):
    ℹ New vocabulary words appearing
    ℹ Unusual prompt patterns detected
    ℹ Gradual latency drift (+5% per week)
```

**Instrumentation Code** (pseudocode):

```python
import time
from prometheus_client import Histogram, Counter, Gauge

# Define metrics
latency_histogram = Histogram(
    'inference_latency_seconds',
    'Time to generate full response'
)
throughput_counter = Counter(
    'tokens_generated_total',
    'Total tokens generated'
)
cache_hit_rate = Gauge(
    'cache_hit_rate',
    'Proportion of requests using cache'
)

# Usage
@latency_histogram.time()  # Auto-measure
def generate_response(prompt):
    start = time.time()
    response = model.generate(prompt)
    tokens = len(tokenizer.encode(response))
    
    # Update metrics
    throughput_counter.inc(tokens)
    elapsed = time.time() - start
    
    return response

# Monitoring loop
def monitor():
    while True:
        cpu_usage = get_cpu_percent()
        memory_mb = get_memory_mb()
        
        if p99_latency > 10.0:
            alert("CRITICAL: High latency detected")
        if error_rate > 0.01:
            alert("CRITICAL: Error rate spike")
        
        time.sleep(60)
```

**Result**:
- **Metrics to track**: Latency, throughput, resources, model health, errors
- **Alert thresholds**: 3 levels (critical, warning, info)
- **Monitoring interval**: 1 minute (or real-time streaming)
- **Dashboard**: Real-time visibility into system health
- **Lesson**: "You can't improve what you don't measure"

---

## Optimization Roadmap

### Phase 1: Current (No Changes)
- **Latency**: 200ms/token (baseline)
- **Throughput**: 5 tokens/sec
- **Cost**: Minimal (CPU only)

### Phase 2: Quick Wins (1-2 weeks)
- **KV Cache**: 19× speedup → 10ms/token
- **Batch inference**: 3-5× throughput improvement
- **Impact**: 100 tokens/sec possible

### Phase 3: Medium Investment (1 month)
- **Quantization**: 4× speedup
- **GPU deployment**: 50× speedup
- **Impact**: 500 tokens/sec

### Phase 4: Production Ready (2-3 months)
- **Multi-GPU parallelism**: 10× more
- **Model distillation**: 2× smaller model
- **RAG system**: Knowledge-grounded responses
- **Impact**: 5000+ tokens/sec per server

---

## Key Takeaways

1. **Attention is the bottleneck** - KV cache is the highest-ROI optimization
2. **Batch inference trades latency for throughput** - know your SLA
3. **Monitor early, optimize late** - measure before optimizing
4. **Multi-level optimization** - combine several small improvements
5. **GPU is 50-100× faster** - consider for production

---

## Questions to Reflect On

- [ ] Why is attention computation expensive during inference?
- [ ] What's the relationship between batch size and latency?
- [ ] Why would KV cache help attention but not feed-forward layers?
- [ ] How would batching work with different sequence lengths?
- [ ] What happens to latency if we add KV cache but double model size?
