# 🚀 Quick Start - Hybrid Banking LLM Setup

## You Have 15 Minutes to Get Started

### Why This Works for You
- **GitHub Pro**: 180 Codespaces hours/month (2x free tier)
- **Google Drive Pro**: 2TB storage (instead of 15GB free)
- **Cost**: $0 additional (you already own these!)
- **Resources**: 4-core, 16GB RAM Codespaces environment

---

## Step 1: Create GitHub Repository (2 min)

1. Go to [github.com/new](https://github.com/new)
2. Name: `banking-llm`
3. Description: "Enterprise Banking LLM with FastAPI, PII Detection, and Cloud Infrastructure"
4. **Don't** initialize with README/gitignore (we have them)
5. Click "Create repository"
6. Copy the HTTPS URL (you'll need it)

---

## Step 2: Push Code to GitHub (3 min)

Run these commands in the `/Users/ashu/Projects/LLM` directory:

```bash
# Make setup script executable
chmod +x scripts/setup-github.sh

# Run interactive setup
bash scripts/setup-github.sh
```

The script will:
- Initialize git
- Configure your name/email
- Commit all files
- Add GitHub remote
- Show you the push command

Then run:
```bash
git branch -M main
git push -u origin main
```

**Expected output**: Files uploading, then "Branch 'main' set up to track 'origin/main'"

---

## Step 3: Launch Codespaces (3 min)

1. Go to your new repo on GitHub
2. Click **Code** button (green)
3. Click **Codespaces** tab
4. Click **"Create codespace on main"**
5. Wait 2-3 minutes for VS Code to open

**What's happening**: GitHub is creating a Linux container with:
- Python 3.11
- All dependencies pre-installed
- PostgreSQL database
- Redis cache
- FastAPI ready to run

---

## Step 4: Verify It Works (4 min)

In the Codespaces terminal, run:

```bash
# Start FastAPI server
uvicorn src.api.main:app --reload --port 8000
```

You should see:
```
Uvicorn running on http://0.0.0.0:8000
```

Click the notification "Open in Browser" or visit:
- **API Docs**: https://your-codespace-url/docs
- **Health Check**: https://your-codespace-url/health/

Test the chat endpoint:
```bash
curl -X POST https://your-codespace-url/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my account balance?"}'
```

**Expected response**: Banking context response with inference

---

## Step 5: Set Up Database (3 min)

In another Codespaces terminal:

```bash
# Check PostgreSQL is running
docker-compose ps

# Create database tables
python scripts/setup-db.py
```

---

## Your New Development Workflow

### Daily Startup (30 seconds)
```bash
# 1. Open Codespaces
# 2. Terminal: uvicorn src.api.main:app --reload
# 3. Click "Open in Browser"
# Done!
```

### For GPU Tasks (e.g., model training)
1. Go to Google Colab
2. Create new notebook
3. Connect to your Google Drive
4. Upload models and run training
5. Save trained models to Drive

### Code Sync
- Changes in Codespaces auto-sync to GitHub
- Google Drive integrates via Python SDK
- Use git pull in Codespaces to get Drive changes

---

## What You Now Have

| Resource | Local Mac | Codespaces | Google Drive | Google Colab |
|----------|-----------|-----------|--------------|-------------|
| **Storage** | 2.5GB free ❌ | 15GB | 2TB ✅ | Unlimited (for notebooks) |
| **CPU** | Intel 12-core | 4-core 🔧 | - | - |
| **RAM** | 16GB (shared) | 16GB dedicated | - | 12GB+ free GPU |
| **Models** | Can't fit ❌ | Store references | Store everything ✅ | Run on GPU ✅ |
| **Development** | Limited ❌ | Full IDE ✅ | - | Experimentation ✅ |

---

## Troubleshooting

**"git: command not found"**
- Run: `brew install git`

**"Codespaces timeout"**
- Default is 30 min. Change in settings for longer idle time

**"PostgreSQL not starting"**
- Run: `docker-compose up -d postgres redis`

**"Port 8000 already in use"**
- Run: `lsof -i :8000` and kill the process, or use port 8001

---

## Next: 8-Week Roadmap

See `HYBRID_SETUP_GUIDE.md` for the complete development timeline.

**This week:**
1. ✅ Codespaces running
2. Implement database models
3. Connect to PostgreSQL
4. Write API tests

**Next week:**
5. Google Colab GPU setup
6. Model inference testing
7. Integration testing

**Week 3:**
8. Fine-tuning workflow
9. Monitoring setup
10. Documentation

**Week 4+:**
11. Production deployment
12. Performance optimization
13. Cost analysis

---

## Pro Tips 💡

**Save Codespaces Hours:**
- Close when not using (saves ~1hr/week)
- Use secrets for API keys (not in repo)
- Commit often to auto-save state

**Speed Up Development:**
- Use VS Code extensions (already pre-installed)
- GitHub Copilot works in Codespaces
- Terminal shortcuts: `Ctrl+`` to toggle terminal

**Store Models Right:**
- Small (<500MB): GitHub releases
- Medium (500MB-2GB): Google Drive
- Large (>2GB): Hugging Face Hub
- For real-time: Together.ai API

---

## Questions?

- **About Codespaces**: See `.devcontainer/devcontainer.json`
- **About Database**: See `docs/03-DATA_MODELS.md` (coming soon)
- **About Architecture**: See `docs/02-ARCHITECTURE.md`
- **About Infrastructure**: See `HYBRID_SETUP_GUIDE.md`

---

**You're ready! Push to GitHub and launch Codespaces now.** 🚀
