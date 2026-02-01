# Hybrid Development Setup - GitHub Codespaces + Google Cloud

**Decision Made:** ✅ HYBRID APPROACH WITH GITHUB PRO + GOOGLE DRIVE PRO  
**Status:** IMPLEMENTATION READY  
**Timeline:** Today → Production in 4-6 weeks

---

## Your Setup

### Available Resources
```
GitHub Pro:         180 hours Codespaces/month (unlimited for your use)
Google Drive Pro:   2 TB storage
Google Colab:       Free GPU access (unlimited)
Local Mac:          FastAPI testing/browsing (optional)
```

### Architecture Overview
```
┌─────────────────┐
│  GitHub Desktop │
│   (Code Sync)   │
└────────┬────────┘
         │
    ┌────v────┐
    │ GitHub  │
    │   Repo  │
    └────┬────┘
         │
    ┌────v────────────────────┐
    │  Codespaces (15GB dev)   │  ← Your main dev environment
    │  • FastAPI Server        │
    │  • Code development      │
    │  • Testing               │
    │  • PostgreSQL            │
    │  • Redis                 │
    └────┬────────────────────┘
         │
    ┌────v──────────────┐
    │ Google Colab      │     ← Model inference & training
    │ (Free GPU access) │
    └────┬──────────────┘
         │
    ┌────v──────────────┐
    │ Google Drive Pro  │     ← 2TB for models/datasets
    │ (Backup & Data)   │
    └───────────────────┘
```

---

## STEP-BY-STEP SETUP (60 minutes)

### STEP 1: Initialize GitHub Repository (5 minutes)

**On your Mac:**
```bash
cd /Users/ashu/Projects/LLM

# Initialize git
git init

# Configure git
git config user.name "Your Name"
git config user.email "your-email@gmail.com"

# Add all files
git add .

# Create initial commit
git commit -m "Initial Banking LLM project - server working, ready for Codespaces"

# Add remote (replace with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/banking-llm.git

# Create main branch and push
git branch -M main
git push -u origin main
```

**⚠️ Important:** Create empty repo on GitHub first (don't initialize with README)

---

### STEP 2: Set Up Codespaces Development Container (10 minutes)

**Create `.devcontainer/devcontainer.json`:**

```json
{
  "name": "Banking LLM Development",
  "image": "mcr.microsoft.com/devcontainers/python:3.11",
  "features": {
    "ghcr.io/devcontainers/features/github-cli:1": {}
  },
  "forwardPorts": [8000, 5432, 6379],
  "portsAttributes": {
    "8000": {
      "label": "FastAPI",
      "onAutoForward": "notify",
      "requireLocalPort": false
    },
    "5432": {
      "label": "PostgreSQL",
      "onAutoForward": "silent",
      "requireLocalPort": false
    },
    "6379": {
      "label": "Redis",
      "onAutoForward": "silent",
      "requireLocalPort": false
    }
  },
  "postCreateCommand": "pip install -r requirements/base.txt",
  "customizations": {
    "vscode": {
      "extensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "ms-python.debugpy",
        "GitHub.copilot",
        "charliermarsh.ruff",
        "ms-python.black-formatter",
        "eamodio.gitlens",
        "ms-azuretools.vscode-docker"
      ],
      "settings": {
        "python.defaultInterpreterPath": "/usr/local/bin/python",
        "python.linting.enabled": true,
        "python.linting.ruffEnabled": true,
        "python.formatting.provider": "black",
        "[python]": {
          "editor.defaultFormatter": "ms-python.python",
          "editor.formatOnSave": true,
          "editor.codeActionsOnSave": {
            "source.organizeImports": true
          }
        }
      }
    }
  }
}
```

**Create `.devcontainer/docker-compose.yml` (for PostgreSQL + Redis):**

```yaml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: banking_user
      POSTGRES_PASSWORD: dev_password
      POSTGRES_DB: banking_llm
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

---

### STEP 3: Create GitHub Actions CI/CD Pipeline (5 minutes)

**Create `.github/workflows/test.yml`:**

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_USER: banking_user
          POSTGRES_PASSWORD: test_password
          POSTGRES_DB: banking_llm
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements/base.txt
        pip install -r requirements/dev.txt
    
    - name: Run tests
      run: pytest tests/ -v
      
    - name: Run linting
      run: ruff check src/
```

---

### STEP 4: Push to GitHub

```bash
# Create these files first (see above)
mkdir -p .devcontainer
mkdir -p .github/workflows

# Add to git
git add .devcontainer/
git add .github/
git add .gitignore

# Commit and push
git commit -m "Add Codespaces and CI/CD configuration"
git push origin main
```

---

### STEP 5: Launch Codespaces (3 minutes)

**In browser:**
1. Go to https://github.com/YOUR_USERNAME/banking-llm
2. Click green "Code" button
3. Select "Codespaces" tab
4. Click "Create codespace on main"
5. Wait 2-3 minutes for environment to spin up
6. VS Code opens in browser with your project

---

### STEP 6: Verify Everything Works in Codespaces

```bash
# In Codespaces terminal:

# Test Python
python --version

# Test dependencies
python -c "import fastapi, torch, transformers; print('✓ All dependencies loaded')"

# Start FastAPI server
uvicorn src.api.main:app --reload --port 8000

# In another terminal tab:
# Test endpoints
curl http://localhost:8000/health/
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"What is my balance?","customer_id":"CUST-123"}'
```

**Expected Result:** Browser opens preview showing your API ✅

---

### STEP 7: Set Up Google Drive for Models/Data

**Create folder structure on Google Drive:**
```
Banking LLM/
├── Models/
│   ├── llama2-7b/
│   ├── mistral-7b/
│   └── training-outputs/
├── Datasets/
│   ├── banking-transactions/
│   ├── customer-interactions/
│   └── training-data/
├── Backups/
│   └── database-exports/
└── Notebooks/
    └── colab-experiments/
```

**Mount Google Drive in Colab:**
```python
# In Google Colab notebook:
from google.colab import drive
drive.mount('/content/drive')

# Access your files:
import os
os.listdir('/content/drive/MyDrive/Banking LLM/Models/')
```

---

### STEP 8: Create Google Colab Notebook for Inference

**Save to:** Google Drive → Banking LLM → Notebooks → colab-experiments.ipynb

**Key cells:**
```python
# Cell 1: Setup
!pip install transformers torch ollama

# Cell 2: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Download model
!ollama pull llama2:7b

# Cell 4: Test inference
import ollama
response = ollama.generate(model='llama2:7b', prompt='You are a banking assistant...')
print(response)

# Cell 5: Save outputs to Drive
import os
with open('/content/drive/MyDrive/Banking LLM/Models/test_output.txt', 'w') as f:
    f.write(response)
```

---

## Daily Workflow

### Morning: Start Development
```bash
1. Go to GitHub Codespaces dashboard
2. Click "Resume" on your codespace
3. It reopens exactly where you left off
4. Terminal history and all files intact
```

### During Day: Code & Test
```bash
In Codespaces:
├── Edit code in VS Code (full IDE)
├── FastAPI runs on :8000 (auto-preview)
├── Git commits saved to GitHub
├── All work synced automatically
└── Switch to Colab tab when testing inference
```

### When Done: Pause or Stop
```bash
# Codespaces auto-pauses after 30 min of inactivity
# Manually pause from GitHub dashboard to save hours
# No data lost when paused/stopped
```

---

## Project Timeline with This Setup

### Week 1: Foundation ✅ (Already Done)
```
✅ API scaffolding
✅ Database schema planning
✅ Security layer (PII detection)
✅ Server running
→ Now: Move to Codespaces + Colab
```

### Week 2-3: Integration
```
→ Real PostgreSQL in Codespaces
→ Redis caching
→ Google Colab inference bridge
→ End-to-end testing
```

### Week 4-5: Models & Training
```
→ Download/train models on Colab
→ Store on Google Drive
→ Fine-tune for banking domain
→ Deploy to Hugging Face Hub
```

### Week 6: Production
```
→ Deploy API (Railway/Render free tier)
→ Set up monitoring
→ Document everything
→ Ready for demo/production
```

---

## Cost Breakdown - With GitHub Pro + Drive Pro

| Service | Your Cost | Notes |
|---------|-----------|-------|
| GitHub Pro | $4/mo | You already have (covers Codespaces 180hrs) |
| Google Drive Pro | $2.99/mo | You already have |
| Codespaces | $0 | Included in GitHub Pro |
| Google Colab | $0 | Free (can pay $9.99 for priority GPU) |
| Hugging Face Hub | $0 | Free storage |
| Railway/Render | $0 | Free tier (scales to $5-10) |
| **TOTAL** | **$7/mo** | You already pay this! |

**Compare:**
- External SSD: $100-200 one-time
- Cloud instances: $50-200/mo
- Your hybrid: $7/mo (you already have)

---

## Files to Create/Commit Now

```bash
# On your Mac, create these files:

1. .devcontainer/devcontainer.json
2. .devcontainer/docker-compose.yml
3. .github/workflows/test.yml
4. .gitignore (if not exists)
5. README.md (update with Codespaces instructions)

# Commit all:
git add .
git commit -m "Add Codespaces dev environment configuration"
git push origin main
```

---

## Next Commands to Run

```bash
# 1. Initialize git (if not done)
cd /Users/ashu/Projects/LLM
git init

# 2. Add remote
git remote add origin https://github.com/YOUR_USERNAME/banking-llm.git

# 3. Create files above

# 4. Commit and push
git add .
git commit -m "Initial commit with Codespaces setup"
git push -u origin main

# 5. Go to GitHub → Create Codespace
# https://github.com/YOUR_USERNAME/banking-llm
# Click "Code" → "Codespaces" → "Create codespace on main"
```

---

## Cleanup Your Local Mac (Optional)

Once Codespaces is working, you can free up your Mac:

```bash
# Optional: Keep code repo, delete venv to free space
rm -rf /Users/ashu/Projects/LLM/venv

# This frees ~1.7GB on your local drive
# Your Codespaces has fresh environment with 15GB

# You can keep the /Users/ashu/Projects/LLM folder for:
# - Git operations
# - Pulling/pushing code
# - Local testing if needed
```

---

## Success Criteria

After setup, you should be able to:

```
☑ Push code to GitHub from Mac
☑ See code in GitHub repo
☑ Open Codespaces from GitHub
☑ Run FastAPI in Codespaces
☑ Access API via browser preview
☑ See PostgreSQL + Redis running
☑ Run tests with GitHub Actions
☑ Access Google Drive from Colab
☑ Train models on Colab GPU
☑ Resume Codespace after pause
```

---

## Quick Reference Commands

```bash
# Codespaces terminal:
python --version                    # Check Python
pip list                           # Check packages
uvicorn src.api.main:app --reload  # Start API
pytest tests/ -v                   # Run tests
git status                         # Check changes
git push origin main               # Push code

# When done:
# Click blue pause button in top-right
# Or use: gh codespace stop
```

---

## Support & Documentation

- **Codespaces Docs:** https://docs.github.com/codespaces
- **Devcontainer Spec:** https://containers.dev
- **Google Colab:** https://colab.research.google.com
- **Your Repository:** https://github.com/YOUR_USERNAME/banking-llm

---

## Summary

**What you're getting:**
- ✅ 15GB development environment (vs 2.5GB locally)
- ✅ Instant resume (no setup every time)
- ✅ Free GPU access via Colab
- ✅ 2TB Google Drive storage
- ✅ Production-ready CI/CD
- ✅ Collaboration-ready setup
- ✅ Cost: $0 extra (you already have the tools)

**What you lose:**
- ❌ Need internet (almost always have it anyway)
- ❌ Slightly more latency (minimal)
- ❌ Local debugging in VS Code (but can use remote SSH)

**Net result:** Better, faster, more professional setup with ZERO additional cost.

---

**Ready to start? Let me help you:**
1. Create the GitHub repo
2. Set up dev container files
3. Push to GitHub
4. Launch your first Codespace

Just let me know! 🚀
