# 📋 Step-by-Step: Push to GitHub & Launch Codespaces

## Part 1: Create GitHub Repository (2 minutes)

### 1.1 Go to GitHub
- Open [github.com/new](https://github.com/new)
- Make sure you're logged in

### 1.2 Configure Repository
Fill in these fields:

| Field | Value |
|-------|-------|
| **Repository name** | `banking-llm` |
| **Description** | Enterprise Banking LLM with FastAPI, PII Detection & Codespaces |
| **Visibility** | Public (or Private if you prefer) |
| **Initialize with** | ☐ README ☐ .gitignore ☐ LICENSE |

⚠️ **Important**: Do NOT check any of the "Initialize" boxes (we already have these files!)

### 1.3 Create Repository
- Click **"Create repository"** button
- Copy the HTTPS URL (looks like: `https://github.com/YOUR_USERNAME/banking-llm.git`)

---

## Part 2: Push Code to GitHub (3 minutes)

### 2.1 Add Remote (run in terminal)
```bash
cd /Users/ashu/Projects/LLM

# Replace YOUR_USERNAME with your actual GitHub username
git remote add origin https://github.com/YOUR_USERNAME/banking-llm.git
```

### 2.2 Rename Branch to Main
```bash
git branch -M main
```

### 2.3 Push to GitHub
```bash
git push -u origin main
```

**Wait for**: 
```
Enumerating objects: 39, done.
Counting objects: 100% (39/39), done.
...
To https://github.com/YOUR_USERNAME/banking-llm.git
 * [new branch]      main -> main
Branch 'main' set up to track 'origin/main'.
```

✅ Your code is now on GitHub!

---

## Part 3: Verify GitHub Actions (1 minute)

### 3.1 Check Your Repository
- Go to your repo: `https://github.com/YOUR_USERNAME/banking-llm`
- Click the **"Actions"** tab
- You should see the workflow running

### 3.2 Wait for Tests
- Tests run automatically on push
- You'll see: 
  - ✅ Python linting (Ruff)
  - ✅ Code formatting (Black)  
  - ✅ Type checking (mypy)
  - ✅ Unit tests (pytest)

---

## Part 4: Launch GitHub Codespaces (5 minutes)

### 4.1 Open Repository
- Go to: `https://github.com/YOUR_USERNAME/banking-llm`

### 4.2 Create Codespace
- Click **Code** button (top right, green)
- Click **Codespaces** tab
- Click **"Create codespace on main"** button

**Wait**: GitHub will create your cloud environment (takes 2-3 minutes)

You'll see a loading screen, then VS Code opens in your browser!

### 4.3 What's Happening
GitHub is provisioning:
```
✓ Linux container (Ubuntu 22.04)
✓ Python 3.11 with all 124 packages
✓ PostgreSQL 15 database
✓ Redis 7 cache server
✓ VS Code with Python extensions
✓ Git configuration
✓ Port forwarding for API (8000)
```

---

## Part 5: Test Codespaces (3 minutes)

### 5.1 Start FastAPI Server

In the **Codespaces Terminal** (bottom of screen):

```bash
# Start the server
uvicorn src.api.main:app --reload --port 8000
```

You should see:
```
Uvicorn running on http://0.0.0.0:8000
```

### 5.2 Open API Documentation
- Click the notification: **"Open in Browser"**
- Or go to: `https://your-codespace-name.preview.app.github.dev`
- Add `/docs` to the end: `https://your-codespace-name.preview.app.github.dev/docs`

### 5.3 Test Endpoints

Click **"Try it out"** on any endpoint:

**Test 1: Health Check**
```
GET /health/
```
Expected response:
```json
{"status": "ok"}
```

**Test 2: Detailed Health**
```
GET /health/detailed
```
Expected response:
```json
{
  "status": "healthy",
  "components": {
    "api": "healthy",
    "inference": "healthy",
    "database": "unknown",
    "cache": "unknown"
  }
}
```

**Test 3: Chat Endpoint**
```
POST /api/v1/chat
```

Request body:
```json
{
  "message": "What services do you provide?",
  "user_id": "user123",
  "session_id": "session456"
}
```

Expected response:
```json
{
  "response": "I am a banking assistant for Bank of America...",
  "inference_model": "ollama-mock",
  "tokens_used": 150,
  "timestamp": "2024-01-XX..."
}
```

---

## Part 6: Set Up Database (2 minutes)

### 6.1 Check Services
In a **new Codespaces terminal** (press `Ctrl+Shift+``):

```bash
docker-compose ps
```

Should show:
```
NAME              STATUS
postgres          Up (healthy)
redis             Up (healthy)
```

### 6.2 Initialize Database
```bash
# Create database schema (when available)
python scripts/setup-db.py
```

---

## Part 7: Daily Workflow Setup

### Your Morning Routine (30 seconds)
```bash
1. Go to github.com/YOUR_USERNAME/banking-llm
2. Click Code → Codespaces → "Create codespace on main"
3. Wait for environment (~2 min)
4. Terminal: uvicorn src.api.main:app --reload
5. Open browser to /docs
6. Start coding!
```

### Code Sync
```bash
# Pull latest changes
git pull origin main

# Make changes
# (auto-saves in Codespaces)

# Commit and push
git add .
git commit -m "Your message"
git push origin main
```

---

## Part 8: Key Codespaces Features

### Available to You (GitHub Pro)

| Feature | Your Limit |
|---------|-----------|
| **Hours/month** | 180 hours |
| **CPU cores** | 4 cores |
| **RAM** | 16 GB |
| **Storage** | 15 GB per codespace |
| **Concurrent codespaces** | 5 |
| **Idle timeout** | 30 minutes (default) |

### Pro Tips

**To save hours:**
```bash
# Stop codespace (not delete)
# Click → Codespaces → Stop codespace
# This pauses the clock!
```

**To get more space:**
```bash
# Use Google Drive Pro (2TB)
# Mount in Codespaces with:
# python -m pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

**To use GPU:**
```bash
# GitHub Codespaces: CPU only
# But you have free Google Colab access!
# Perfect for model training
```

---

## Troubleshooting

### "Repository not found" when pushing
**Fix**: Check your GitHub token
```bash
git remote -v  # Show current remote
# Should show: https://github.com/YOUR_USERNAME/banking-llm.git
```

### "Codespaces won't start"
**Fix**: Try again or check for errors in setup logs

### "Port 8000 already in use"
**Fix**: Use different port
```bash
uvicorn src.api.main:app --reload --port 8001
```

### "Docker services not starting"
**Fix**: Restart services
```bash
docker-compose down
docker-compose up -d postgres redis
```

---

## Next Steps After Codespaces Works

### ✅ When you see `/docs` working:

1. **Database Integration**
   - Create schema in PostgreSQL
   - Connect ORM to models

2. **Redis Integration**
   - Add caching layer
   - Session management

3. **Model Management**
   - Set up Google Colab notebook
   - Configure model storage on Google Drive

4. **Testing**
   - Write comprehensive test suite
   - Set up coverage reporting

5. **Documentation**
   - Complete remaining docs (8 more files)
   - Add API specifications

---

## Success Criteria ✅

When you've completed this guide:

- [ ] ✅ GitHub repository created
- [ ] ✅ Code pushed to GitHub
- [ ] ✅ GitHub Actions running tests
- [ ] ✅ Codespaces launched
- [ ] ✅ FastAPI server running
- [ ] ✅ `/docs` API page loading
- [ ] ✅ Health endpoints responding
- [ ] ✅ Chat endpoint working
- [ ] ✅ PostgreSQL connected
- [ ] ✅ Redis connected

---

## Questions?

- **VS Code in Codespaces**: Built-in help (Ctrl+H)
- **GitHub Codespaces**: See [github.com/features/codespaces](https://github.com/features/codespaces)
- **API Documentation**: See `docs/` folder
- **Database Schema**: See `HYBRID_SETUP_GUIDE.md`

---

**You've now got a professional, cloud-based development environment!** 🎉

**Total time: ~20 minutes**

Next time you open Codespaces, it'll take <30 seconds since the environment is cached.
