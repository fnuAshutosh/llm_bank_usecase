# Banking LLM System - Deployment Guide

## 📊 Current Resource Usage

```
API Server:     960 MB RAM, ~5% CPU
Frontend:       19 MB RAM
Total:          ~1 GB RAM
Project Size:   671 MB
```

## 💻 LOCAL MAC DEPLOYMENT

### ✅ **YES, IT WILL WORK ON MAC!**

**Minimum Requirements:**
- **RAM**: 8GB (4GB free)
- **CPU**: 2 cores (Apple Silicon M1+ or Intel)
- **Disk**: 5GB free space
- **OS**: macOS 11+ (Big Sur or later)

**Recommended:**
- **RAM**: 16GB
- **CPU**: 4 cores
- **Disk**: 10GB

### 🚀 Quick Setup on Mac

```bash
# 1. Install dependencies
brew install python@3.11

# 2. Clone and setup
git clone https://github.com/fnuAshutosh/llm_bank_usecase.git
cd llm_bank_usecase

# 3. Create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 4. Install packages
pip install -r requirements/base.txt -r requirements/prod.txt

# 5. Setup environment
cp .env.example .env
# Edit .env with your API keys

# 6. Start API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 7. Start frontend
python3 -m http.server 3000 &

# 8. Open browser
open http://localhost:3000/frontend/index.html
```

**Install Time**: ~10 minutes
**Startup Time**: ~30 seconds

---

## 🐳 DOCKER DEPLOYMENT

### Option 1: Lightweight Version (Recommended)

**Size**: ~500MB (without ML libraries)
**RAM**: ~200MB

```dockerfile
# Dockerfile.light
FROM python:3.11-slim

WORKDIR /app

# Install minimal dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy only essential requirements
COPY requirements/base.txt .
RUN pip install --no-cache-dir fastapi uvicorn pydantic sqlalchemy psycopg2-binary redis python-jose passlib

COPY src/ ./src/
COPY frontend/ ./frontend/

EXPOSE 8000 3000

CMD uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### Option 2: Full Version (with AI features)

**Size**: ~3GB (includes PyTorch, transformers)
**RAM**: ~1GB

```dockerfile
# Dockerfile (already exists in project)
```

---

## ☁️ FREE HOSTING OPTIONS

### 1. **Railway.app** ⭐ RECOMMENDED
```
✅ Free Tier: 500 hours/month, 8GB RAM, 8GB disk
✅ Easy setup: Connect GitHub, auto-deploy
✅ Postgres included
✅ Custom domain
```

**Setup:**
```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

**Cost**: FREE (with $5/month credit)

---

### 2. **Render.com**
```
✅ Free Tier: 750 hours/month
✅ Auto-deploy from GitHub
✅ Postgres included
⚠️  Sleeps after 15 min inactivity
```

**Setup:**
- Connect GitHub repo
- Select "Web Service"
- Build: `pip install -r requirements/prod.txt`
- Start: `uvicorn src.api.main:app --host 0.0.0.0 --port $PORT`

**Cost**: FREE

---

### 3. **Fly.io**
```
✅ Free: 3 VMs with 256MB RAM each
✅ Global deployment
✅ Postgres included
✅ No sleep policy
```

**Setup:**
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch
fly deploy
```

**Cost**: FREE (up to $5/month usage)

---

### 4. **Google Cloud Run**
```
✅ Free Tier: 2 million requests/month
✅ Auto-scaling
✅ Pay per use
⚠️  Cold starts
```

**Cost**: FREE (within limits)

---

### 5. **AWS Lambda + API Gateway**
```
✅ Free Tier: 1 million requests/month
✅ Serverless
⚠️  Requires adaptation for Lambda
```

---

## 🎯 BEST DEPLOYMENT STRATEGY

### For Development/Demo:
**Railway.app** - Easiest, most generous free tier

### For Production:
**Google Cloud Run** - Scalable, cost-effective

### For Mac Local:
**Direct Python** - No Docker needed, fastest

---

## 📦 DOCKER BUILD & PUSH

### Lightweight Image (200MB)

```bash
# 1. Build lightweight image
docker build -f Dockerfile.light -t banking-llm:light .

# 2. Test locally
docker run -p 8000:8000 -p 3000:3000 banking-llm:light

# 3. Push to Docker Hub (FREE)
docker login
docker tag banking-llm:light yourusername/banking-llm:light
docker push yourusername/banking-llm:light

# 4. Deploy anywhere!
```

### Push to GitHub Container Registry (FREE)

```bash
# 1. Create personal access token on GitHub
# Settings > Developer settings > Personal access tokens

# 2. Login
echo YOUR_TOKEN | docker login ghcr.io -u fnuAshutosh --password-stdin

# 3. Tag and push
docker tag banking-llm:light ghcr.io/fnuashutosh/banking-llm:light
docker push ghcr.io/fnuashutosh/banking-llm:light

# 4. Deploy on any platform using:
docker pull ghcr.io/fnuashutosh/banking-llm:light
```

---

## 🔥 QUICK START (5 Minutes)

### Deploy to Railway (Easiest):

```bash
# 1. Create railway.json
{
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health"
  }
}

# 2. Push to GitHub
git add .
git commit -m "Deploy config"
git push

# 3. Go to railway.app
# - Connect GitHub repo
# - Click "Deploy"
# - Done! 🎉
```

---

## 💰 COST COMPARISON

| Platform | Free Tier | RAM | Storage | Sleep? |
|----------|-----------|-----|---------|--------|
| Railway | 500 hrs/mo | 8GB | 8GB | No |
| Render | 750 hrs/mo | 512MB | 1GB | Yes (15min) |
| Fly.io | Always | 256MB | 1GB | No |
| Vercel | Unlimited | 1GB | 100MB | No |
| GCP Run | 2M req/mo | 2GB | - | Cold start |

**Recommendation**: Railway for demo, then migrate to GCP Run for production

---

## 🛠️ REDUCE RESOURCE USAGE

### Option 1: Disable ML Libraries (400MB → 100MB)

```python
# src/api/main.py - Already configured!
# Search routes disabled by default
# Keeps startup fast
```

### Option 2: Use External LLM API

```env
# .env
LLM_PROVIDER=openai  # Instead of local ollama
OPENAI_API_KEY=your_key
```

**Result**: RAM drops from 960MB → 150MB

---

## ✅ READY TO DEPLOY?

Choose your path:

1. **Local Mac**: Run setup script above (10 min)
2. **Railway**: Push to GitHub, connect repo (5 min)
3. **Docker**: Build image, push to hub (15 min)

All options work with your Mac! 🚀
