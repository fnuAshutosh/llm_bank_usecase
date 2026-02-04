# Hybrid Architecture: Codespace + Colab + Google Drive

## Status: ✅ ACTIVE (February 2026)

This document describes your distributed intelligent solution for managing LLM development with limited resources.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Repository                         │
│              (Code, Docs, Configuration)                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          │            │            │
    ┌─────▼──────┐ ┌──▼──────┐ ┌──▼──────────┐
    │ Codespaces │ │ Colab   │ │ Google      │
    │ (Dev)      │ │ (GPU)   │ │ Drive       │
    │            │ │         │ │ (Storage)   │
    │ CPU only   │ │ T4/A100 │ │ 2TB         │
    │ 15GB       │ │ GPU     │ │             │
    │ 180hrs/mo  │ │ 12hrs   │ │ Persistent  │
    └─────┬──────┘ └──┬──────┘ └──┬──────────┘
          │           │           │
          └───────────┼───────────┘
                      │
            ┌─────────▼──────────┐
            │   PostgreSQL +     │
            │   Redis (Cloud)    │
            │   (Metadata/Cache) │
            └────────────────────┘
```

---

## Resource Allocation

### Codespaces (Primary Development)
- **Purpose**: Development, testing, API
- **Storage**: 15GB (clean, no GPU bloat)
- **Requirements**: CPU-only Python packages
- **Workflow**:
  ```bash
  dev@codespace:~$ uvicorn src.api.main:app --reload
  # Now test against models loaded from Drive or APIs
  ```
- **Models**: Reference only (loaded on-demand from Drive/APIs)
- **Cost**: $0 (included in GitHub Pro)

### Google Colab (GPU Compute)
- **Purpose**: Model training, fast inference, experimentation
- **Hardware**: Free T4 GPU (can upgrade to A100)
- **Duration**: 12 hours per session
- **Workflow**:
  1. Mount Google Drive
  2. Load data from Drive
  3. Train model with GPU
  4. Save results back to Drive
- **Cost**: $0 (or $9.99/month for priority GPU)

### Google Drive (Persistent Storage)
- **Purpose**: Model checkpoints, datasets, results
- **Storage**: 2TB (you have Pro)
- **Folder Structure**:
  ```
  Banking LLM/
  ├── models/
  │   ├── base-models/
  │   │   └── llama2-7b/
  │   │       ├── model.bin (1.5GB)
  │   │       └── metadata.json
  │   ├── finetuned/
  │   │   ├── banking-v1/
  │   │   │   ├── adapter_config.json
  │   │   │   ├── adapter_model.bin
  │   │   │   └── metadata.json
  │   │   └── banking-v2/
  │   └── deployed/
  │       └── current → symlink to banking-v2
  ├── datasets/
  │   ├── banking77_train.csv
  │   └── banking77_test.csv
  ├── notebooks/
  │   ├── training.ipynb
  │   └── inference.ipynb
  └── results/
      ├── benchmarks/
      ├── training_logs/
      └── inference_results/
  ```
- **Cost**: $2.99/month (you have Pro)

---

## Daily Workflow

### Morning: Start Development (Codespaces)
```bash
# 1. Open Codespaces (web or VS Code)
# 2. Terminal:
python -m pip install -r requirements/base.txt

# 3. Start API server (CPU inference OK for testing)
uvicorn src.api.main:app --reload

# 4. Test endpoints with pre-loaded models
curl http://localhost:8000/health
```

**Key**: Models are loaded on-demand from Hugging Face Hub or downloaded from Drive.

### Need Fast Inference or GPU? → Use Colab
```
Steps:
1. Open https://colab.research.google.com
2. File → Open → GitHub
3. Paste: https://github.com/fnuAshutosh/llm_bank_usecase
4. Open: LoRA_Benchmark_Colab.ipynb
5. Runtime → GPU (T4 or A100)
6. Run cells (inference runs 3-5x faster on GPU)
```

### Train New Model? → Use Colab + Drive
```
In Colab:
1. Mount Drive: drive.mount('/content/drive')
2. Load training data: df = pd.read_csv('/content/drive/MyDrive/Banking LLM/datasets/banking77_train.csv')
3. Train model with GPU acceleration
4. Upload results:
   from src.cloud import GoogleDriveManager
   manager = GoogleDriveManager()
   manager.upload_model('./output/model', 'banking-v2', 'latest')
```

### Back in Codespaces: Use New Model
```python
from src.cloud import GoogleDriveManager

manager = GoogleDriveManager()
result = manager.download_model('banking-v2', './models', 'latest')

# Now use the model
from transformers import AutoModelForSequenceClassification
model = AutoModelForSequenceClassification.from_pretrained('./models/banking-v2')
```

---

## Python Package Strategy

### ✅ In Codespaces (CPU-only)
- `torch==2.2.0` (CPU version) - lightweight, fast installation
- `transformers` - model management
- `peft` - LoRA adapter support
- `sentence-transformers` - embeddings (CPU OK)
- No: `bitsandbytes`, `triton`, `nvidia-*`

**Install with**:
```bash
pip install -r requirements/base.txt
```

### ✅ In Colab (GPU version)
- `torch==2.2.0` with CUDA support
- `bitsandbytes` - quantization
- `triton` - GPU kernels
- All GPU optimizations

**Install with**:
```bash
# In Colab cell 1:
%pip install -r requirements/gpu.txt --quiet
```

---

## Google Drive Integration

### Module: `src/cloud/google_drive.py`

**For Colab (uploading models)**:
```python
from src.cloud import init_drive_manager

manager = init_drive_manager()  # Auto-mounts Drive
manager.upload_model(
    './trained_model',
    'banking-lora',
    'v1'
)
manager.save_metadata('banking-lora', {
    'loss': 0.12,
    'eval_acc': 0.94,
    'training_time_hours': 2.5
}, 'v1')
```

**For Codespaces (downloading models)**:
```python
from src.cloud import GoogleDriveManager

# Read-only mode (Drive not mounted in Codespaces)
manager = GoogleDriveManager()
result = manager.download_model('banking-lora', './models', 'v1')
print(f"Model saved to: {result['local_path']}")
```

---

## Cost Analysis

### Current Setup
| Service | You Pay | Monthly |
|---------|---------|---------|
| GitHub Pro | ✓ | $4.00 |
| Google Drive Pro | ✓ | $2.99 |
| Codespaces | Included | $0 |
| Google Colab | FREE | $0 |
| **TOTAL** | - | **$6.99** |

### What You Get
- ✅ 15GB dev environment (vs 2.5GB before)
- ✅ Free GPU training (Colab)
- ✅ 2TB persistent storage
- ✅ Production-ready CI/CD
- ✅ Unlimited Colab notebooks
- ✅ Model versioning on Drive

### Alternative Costs (Not Recommended)
- **External SSD**: $100-200 one-time ❌ Still no GPU
- **AWS EC2 Instance**: $50-200/month ❌ Ongoing cost
- **Google Cloud VM**: $100-300/month ❌ Ongoing cost

---

## Troubleshooting

### "Out of memory in Colab"
```python
# Use QLoRA (already in LoRA_Benchmark_Colab.ipynb)
from peft import get_peft_model, LoraConfig, TaskType
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="TOKEN_CLASSIFICATION"
)
```

### "Model not found in Codespaces"
- Explicitly download from Drive first:
  ```bash
  python -c "from src.cloud import GoogleDriveManager; m = GoogleDriveManager(); m.download_model('model-name', './models')"
  ```
- Or use Hugging Face Hub directly:
  ```python
  from transformers import AutoModel
  model = AutoModel.from_pretrained('meta-llama/Llama-2-7b-hf')
  ```

### "Colab session timeout"
- Train models in shorter batches
- Save checkpoints frequently to Drive
- Use `dvc` (data version control) for large datasets

---

## Next Steps

### Week 2-3: Optimization
- [ ] Benchmark CPU vs GPU inference latency
- [ ] Set up CI/CD for model validation
- [ ] Create automatic model versioning
- [ ] Configure alerts for training job status

### Week 4: Production
- [ ] Deploy API to Railway/Render
- [ ] Set up model serving layer
- [ ] Configure Pinecone for embeddings
- [ ] Document for team collaboration

### Month 2: Scale
- [ ] Add RunPod for intensive training runs
- [ ] Implement DVC for data versioning
- [ ] Set up MLflow for experiment tracking
- [ ] Create Colab job scheduling

---

## Key Takeaway

```
Limited Resources (Codespace) ≠ Limited Capabilities

Codespaces (15GB, CPU) → Development & API Testing
+ Colab (12hrs, GPU) → Fast Training & Inference  
+ Drive (2TB) → Persistent Storage
= Production-Ready LLM Platform @ $0 Extra Cost
```

You now have a professional, scalable setup without expensive hardware.

---

**Questions?** Check:
- [Codespaces Docs](https://docs.github.com/en/codespaces)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)  
- [HYBRID_SETUP_GUIDE.md](./HYBRID_SETUP_GUIDE.md)
