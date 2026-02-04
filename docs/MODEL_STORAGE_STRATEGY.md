# Model Storage Strategy for Hybrid Architecture

## Codespace Storage Layout (15GB)

### Recommended Directory Structure
```
/workspaces/llm_bank_usecase/
├── models/                          # Local model cache (~2-3GB max)
│   ├── base/                        # Base models (download on-demand)
│   │   └── .gitkeep                 # Don't commit actual models!
│   └── cache/                       # HF model cache
│
├── src/                             # Code (~100MB)
├── data/
│   ├── banking77_finetuning/        # Dataset (~50MB)
│   └── finetuning/                  # Training configs (~500KB)
├── notebooks/                       # Jupyter notebooks (~10MB)
├── .git/                            # Git history (~2MB after cleanup)
└── [Other code/docs]                # Remaining space
```

### Total Breakdown
| Item | Size | Status |
|------|------|--------|
| Code + Docs | ~300MB | ✓ Essential |
| Dependencies | ~2GB | ✓ Required |
| Data | ~100MB | ✓ Datasets |
| Models (local) | ~2GB | ⚠ On-demand only |
| Free Space | ~10GB | ✓ Build buffer |
| **TOTAL** | **15GB** | ✓ Sustainable |

---

## Google Drive Storage Layout (2TB)

### Folder Structure for Models
```
Banking LLM/
│
├── models/                          # All model versions
│   ├── base-models/
│   │   ├── llama2-7b/
│   │   │   ├── model.bin (1.5GB)
│   │   │   ├── config.json
│   │   │   └── metadata.json
│   │   └── mistral-7b/
│   │       └── ...
│   │
│   ├── finetuned/
│   │   ├── banking-v1/              # LoRA adapter (checkpoint)
│   │   │   ├── lora_config.json
│   │   │   ├── adapter_model.bin (50-100MB)
│   │   │   ├── metadata.json
│   │   │   └── benchmark.json
│   │   ├── banking-v2/
│   │   │   └── ...
│   │   └── banking-latest/ → symlink
│   │
│   └── deployed/
│       └── current → symlink to banking-v2
│
├── datasets/
│   ├── banking77_train.csv (5MB)
│   ├── banking77_test.csv (2MB)
│   ├── banking77_augmented.csv (20MB)
│   └── domain-specific.csv (50MB)
│
├── notebooks/
│   ├── 01_training.ipynb
│   ├── 02_inference.ipynb
│   ├── 03_benchmark.ipynb
│   └── 04_analysis.ipynb
│
├── results/
│   ├── training_logs/
│   │   ├── run_v1_2026-02-04.json
│   │   └── run_v2_2026-02-10.json
│   ├── benchmarks/
│   │   ├── inference_speed.csv
│   │   ├── quality_metrics.json
│   │   └── memory_usage.csv
│   └── inference_results/
│       └── test_batch_001.json
│
└── docs/
    ├── model_registry.json
    ├── training_log.md
    └── architecture.md
```

### Storage Estimation
| Category | Files | Size | Notes |
|----------|-------|------|-------|
| Base LLMs | 2-3 | 3-5GB | Download once, keep forever |
| Fine-tuned | 10-20 | 500MB-2GB | All versions, including failed runs |
| Datasets | 5-10 | 100MB-500MB | Training/test splits |
| Notebooks | 5-10 | 10-50MB | Colab experiments |
| Results/Logs | 50+ | 100-500MB | Training history |
| **TOTAL** | - | **~5-8GB** | Out of 2TB 🎉 |

---

## Model Download/Cache Strategy

### Scenario 1: First Time Setup (Codespaces)
```python
# Week 1: Download base model once, store on Drive
from transformers import AutoModel, AutoTokenizer

model_name = "meta-llama/Llama-2-7b-hf"
model = AutoModel.from_pretrained(model_name)  # 1.5GB download
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Cache location: ~/.cache/huggingface/hub/ (~2GB)
# ⚠️ Don't commit to git, clear after testing
```

### Scenario 2: Reuse in Codespaces
```python
# Cache still exists, fast load
# If cache cleared, download again from HF Hub (only once per session)
from transformers import AutoModel

model = AutoModel.from_pretrained(model_name)  # Instant (from cache)
```

### Scenario 3: Train in Colab
```python
# In Colab:
from src.cloud import GoogleDriveManager
from transformers import get_peft_model, LoraConfig

manager = init_drive_manager()

# Load base model from HF Hub (cache will be local to Colab)
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# Train with LoRA (produces 50-100MB adapter)
...training...

# Save entire directory to Drive (~1.2GB)
manager.upload_model('./output', 'banking-v2', 'latest')
```

### Scenario 4: Use Trained Model in Codespaces
```python
# In Codespaces:
from src.cloud import GoogleDriveManager
from peft import PeftModel

manager = GoogleDriveManager()

# This downloads from Drive (fake mode - just for testing structure)
result = manager.download_model('banking-v2', './models')

# Load in Codespaces
base_model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
model = PeftModel.from_pretrained(base_model, './models/banking-v2/latest')

# Now use model for inference/testing
```

---

## Local Cache Management (Codespaces)

### View Current Cache
```bash
# HF Hub cache
du -sh ~/.cache/huggingface/hub/
# pip cache
du -sh ~/.cache/pip/

# Expected after cleanup: <1GB
```

### Clear Cache When Needed
```bash
# After testing, before committing
rm -rf ~/.cache/huggingface/hub/models*
huggingface-cli cache-system
# Saves ~2GB, models re-downloaded on next use (from cache)
```

### .gitignore - Never Commit Models
```
# models/base/ - NO! Don't commit
# ~/.cache/ - NO! Don't commit

# DO commit:
models/.gitkeep
.gitignore
requirements/
src/
```

---

## Cloud Model Registry (Track Everything)

### File: Drive → Banking LLM/docs/model_registry.json
```json
{
  "models": [
    {
      "id": "llama2-7b-base",
      "source": "meta-llama/Llama-2-7b-hf",
      "size_gb": 1.5,
      "cached_in_codespaces": false,
      "cached_in_colab": true,
      "last_used": "2026-02-04",
      "purpose": "base model"
    },
    {
      "id": "banking-v1",
      "type": "LoRA adapter",
      "size_mb": 87,
      "base_model": "llama2-7b-base",
      "training_date": "2026-02-05",
      "metrics": {
        "eval_loss": 0.124,
        "accuracy": 0.938,
        "f1": 0.935
      },
      "status": "archived"
    },
    {
      "id": "banking-v2",
      "type": "LoRA adapter",
      "size_mb": 92,
      "base_model": "llama2-7b-base",
      "training_date": "2026-02-10",
      "metrics": {
        "eval_loss": 0.098,
        "accuracy": 0.952,
        "f1": 0.950
      },
      "status": "deployed",
      "deployed_at": "2026-02-10T14:30:00Z"
    }
  ]
}
```

---

## Decision Matrix: Where to Store What?

| Item | Codespaces | Drive | HF Hub | Notes |
|------|-----------|-------|--------|-------|
| Source Code | ✓ | - | - | In git repo |
| Requirements | ✓ | - | - | In git repo |
| Base Model | ⚠ Cache only | ✓ Large | ✓ preferred | Download on-demand |
| Fine-tuned | ⚠ Temp | ✓ Always | ✓ Optional | Your models |
| Datasets | ✓ Link only | ✓ If large | - | Reference or cache |
| Notebooks | - | ✓ | - | Colab experiments |
| Training Logs | - | ✓ | ✓ MLflow | Track history |
| Configs | ✓ | ✓ | - | In git + backup |

**Legend**: ✓ = store here, ⚠ = temporary, - = not recommended

---

## Maintenance Schedule

### Daily (Codespaces)
- No action needed, caches grow naturally
- Keep free space >5GB for builds

### Weekly  
- Review Drive space usage
- Archive old experiment notebooks

### Monthly
- Clean Codespaces cache if >3GB
- Consolidate model versions

### Quarterly
- Delete failed training runs
- Update model registry
- Archive old notebooks to local backup

---

## Cost Impact

### Current Setup
```
GitHub Codespaces: $0 (included in Pro)
  - Storage: 15GB (sufficient with smart caching)
  - Cache: Auto-managed, can exceed temporarily

Google Drive Pro: $2.99/month
  - Storage: 2TB for models/datasets
  - Auto-sync to Colab
  
Total: $2.99/month (vs $100-300/month for cloud VMs)
```

### Storage Math
```
Codespaces: 15GB
  - Code/Docs: 300MB
  - Python packages: 2GB
  - HF Cache: 2GB (temporary, cleared weekly)
  - Free buffer: ~10GB ✓ Safe

Google Drive: 2TB
  - Used: 5-8GB for all models + datasets
  - Free: 1.99TB remaining ✓ Plenty
  
Total: 2TB+ storage, $0 extra cost
```

---

## Quick Reference Commands

```bash
# Check Codespaces space
df -h
du -sh ~/.cache/huggingface/

# Clear cache
rm -rf ~/.cache/huggingface/hub/models*

# In Colab: Mount drive
from google.colab import drive
drive.mount('/content/drive')

# In Colab: Upload model
from src.cloud import init_drive_manager
manager = init_drive_manager()
manager.upload_model('./output', 'banking-v2', 'latest')

# In Codespaces: List Drive models (simulation)
from src.cloud import GoogleDriveManager
manager = GoogleDriveManager()
manager.list_models()
```

---

## Success Criteria

✅ Codespaces never exceeds 15GB  
✅ Models stored on Drive, referenced in code  
✅ Colab experiments auto-saved to Drive  
✅ Model registry kept up-to-date  
✅ Git repo stays <100MB (no model files)  
✅ Easy to download any model version  
✅ Zero additional cost  

---

This strategy keeps you ***lean, fast, and scalable***.
