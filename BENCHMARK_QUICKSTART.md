# 🚀 Quick Start: Benchmark Guide

## What to Benchmark?

You'll test these fine-tuning methods on the **Banking77 intent classification dataset**:

### 1. **LoRA** (Baseline)
- Rank: 4, 8, 16, 32
- Measures: Accuracy, Memory, Speed, Parameter efficiency

### 2. **QLoRA** (4-bit quantization)
- Variant of LoRA with quantization
- 50-70% memory reduction
- Slightly slower inference

### 3. **Vector Database Comparison**
- FAISS (local, fast)
- ChromaDB (lightweight)
- Pinecone (cloud, production)
- Milvus (scalable)

### 4. **Feasibility Algorithms**
- Active learning (uncertainty sampling)
- Core-set selection
- Diversity sampling

---

## Option A: Run on Google Colab (RECOMMENDED FOR BEGINNERS)

### Step 1: Open Notebook
1. Go to: https://colab.research.google.com
2. Click "File" → "Open notebook" → "GitHub"
3. Paste: `https://github.com/fnuAshutosh/llm_bank_usecase`
4. Open: `LoRA_Benchmark_Colab.ipynb`

### Step 2: Enable GPU
- Runtime → Change runtime type → GPU (T4 or better)

### Step 3: Run Cells Sequentially
```
1. Setup & Install (5 min)
2. Load Data (2 min)
3. LoRA Benchmarks (30-60 min depending on config)
4. Visualize Results (2 min)
5. Save to Drive (1 min)
```

### Expected Results
```
LoRA Rank 8 Results:
- Accuracy: 91.2%
- Training: 45 sec
- Memory: 2.1 GB
- Parameters: 0.03% of base model
```

### Save to Drive
All results auto-save to Google Drive in `/Banking_LLM_Benchmarks/`

---

## Option B: Run Locally (Advanced)

### Prerequisites
```bash
# Python 3.8+
python --version

# GPU (NVIDIA CUDA 11.8+, optional but recommended)
nvidia-smi

# 8GB+ RAM
# 10GB+ disk space
```

### Installation

```bash
# Clone repo
git clone https://github.com/fnuAshutosh/llm_bank_usecase.git
cd llm_bank_usecase

# Create venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install peft transformers datasets evaluate tensorboard bitsandbytes scikit-learn pandas numpy matplotlib seaborn
```

### Run Benchmarks

```bash
# Method 1: Python script
python benchmark_suite.py

# Method 2: Jupyter notebook
jupyter notebook

# Then open: LoRA_Benchmark_Colab.ipynb
```

### Monitor Progress
```bash
# Watch GPU usage (in another terminal)
watch nvidia-smi

# Watch CPU/memory
htop
```

---

## Option C: Run on Kaggle

### Step 1: Create Account
- Go to: https://www.kaggle.com
- Sign up (free)

### Step 2: Create Notebook
- New Notebook
- Select GPU (if available)

### Step 3: Copy Code
Paste the code from `benchmark_suite.py` into Kaggle cells

### Step 4: Run & Download Results
Download CSV and PNG results when complete

---

## What to Measure?

### Training Metrics
```python
metrics = {
    "training_time": "seconds",
    "peak_memory": "GB",
    "trainable_params": "count",
    "param_ratio": "% of total",
}
```

### Quality Metrics
```python
metrics = {
    "accuracy": "% correct",
    "f1_score": "0-1 scale",
    "precision": "0-1 scale",
    "recall": "0-1 scale",
}
```

### Inference Metrics
```python
metrics = {
    "latency": "ms per token",
    "throughput": "tokens/sec",
    "memory": "MB resident",
}
```

---

## Experiment Workflow

### Week 1: LoRA Sweep
```
Try: r ∈ [4, 8, 16, 32, 64]
Measure: Accuracy vs Memory curve
Goal: Find optimal rank
```

### Week 2: QLoRA Testing
```
Try: 4-bit, 8-bit quantization
Compare: Speed vs accuracy vs memory
Goal: Is it worth the latency?
```

### Week 3: Vector DB Testing
```
Try: FAISS, ChromaDB, Pinecone
Measure: Indexing time, query latency, accuracy
Goal: Best DB for production
```

### Week 4: Active Learning
```
Try: Uncertainty sampling, core-set
Measure: Samples needed for 90% accuracy
Goal: Reduce data requirements 30-50%
```

---

## Key Questions to Answer

1. **What's the optimal LoRA rank?**
   - Plot accuracy vs rank
   - Find knee point

2. **Does QLoRA save enough memory?**
   - Compare 4-bit vs full precision
   - Measure inference latency

3. **Which vector DB for banking?**
   - Pinecone: Fast, cloud, $
   - FAISS: Local, free, fast
   - ChromaDB: Python, embedded

4. **Can active learning reduce data 50%?**
   - Train on 500 samples + active learning
   - Compare to 1000 random samples

5. **What's the production-ready stack?**
   - LoRA (r=8) + FAISS + Uncertainty sampling
   - Cost: $0
   - Accuracy: ~90%
   - Speed: <50ms per query

---

## Expected Timeline

| Phase | Duration | Effort |
|-------|----------|--------|
| Setup | 30 min | Low |
| LoRA baseline | 1 hour | Low |
| LoRA sweep (r=4,8,16,32) | 2 hours | Low |
| QLoRA | 1 hour | Low |
| Vector DB | 1 hour | Low |
| Active learning | 2 hours | Medium |
| Report + analysis | 1 hour | Low |
| **TOTAL** | **~8 hours** | **Low** |

---

## Troubleshooting

### Out of Memory (OOM)
```python
# Reduce batch size
per_device_train_batch_size = 8  # was 32

# Or use gradient accumulation
gradient_accumulation_steps = 4

# Or enable QLoRA (4-bit)
load_in_4bit = True
```

### GPU Not Available
```python
# Use CPU (slower, but works)
device = "cpu"

# Or use Colab GPU (recommended)
```

### Slow Training
```python
# Reduce epochs
num_train_epochs = 1  # was 3

# Or reduce dataset size
dataset = dataset["train"].select(range(500))  # was 5000
```

---

## Next Steps After Benchmarking

1. **Document findings** in a markdown report
   - Best configs for each method
   - Trade-off analysis
   - Recommendations

2. **Choose production stack**
   - LoRA rank?
   - Vector DB?
   - Inference framework?

3. **Implement in production**
   - Use best config from benchmarks
   - Deploy to Fly.io / Railway / AWS

4. **Monitor & iterate**
   - Track real-world accuracy
   - Retrain monthly
   - Optimize based on user feedback

---

## Resources

- **Banking77 Paper**: https://arxiv.org/abs/2003.04807
- **LoRA Paper**: https://arxiv.org/abs/2106.09685
- **QLoRA Paper**: https://arxiv.org/abs/2305.14314
- **PEFT Library**: https://github.com/huggingface/peft
- **Colab Setup**: https://colab.research.google.com/notebooks/welcome.ipynb

---

## Questions?

Post in `/workspaces/llm_bank_usecase/`:
- Share your results in GitHub Discussions
- Compare with others
- Discuss findings

---

**Good luck! 🚀 Let's find the best LoRA config for banking!**
