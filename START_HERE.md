# 🚀 SETUP CHECKLIST - Start Here!

## ✅ Everything is Committed & Pushed to GitHub!

**Repository**: https://github.com/fnuAshutosh/llm_bank_usecase
**Commit**: d742db1 - Complete banking LLM system

---

## 📋 BEFORE YOU START - REQUIRED STEPS

### 1️⃣ **Copy Environment File** ⚠️ IMPORTANT

```bash
cp .env.example .env
```

Then edit `.env` and add your API keys:

```bash
# Required for AI features
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Required for database
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here

# Optional - Vector search (Pinecone)
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=banking-assistant

# Optional - LLM provider (if using Ollama locally)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
```

**Where to get API keys:**
- **OpenAI**: https://platform.openai.com/api-keys (FREE $5 credit)
- **Supabase**: https://supabase.com (FREE tier - create project)
- **Pinecone**: https://www.pinecone.io (FREE tier - 1 index)
- **Ollama**: https://ollama.ai (FREE - local LLM)

---

### 2️⃣ **Setup Supabase Database** (5 minutes)

1. Go to https://supabase.com
2. Create new project
3. Copy URL and anon key to `.env`
4. Run database setup:

```bash
python setup_database.py
```

This creates all tables (customers, accounts, transactions, etc.)

---

### 3️⃣ **Install Dependencies**

**On Mac:**
```bash
./setup-mac.sh
source venv/bin/activate
```

**On Linux/Windows:**
```bash
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements/base.txt -r requirements/prod.txt
```

---

## 🎯 QUICK START PATHS

### Path A: Local Development (Recommended for Mac)

```bash
# 1. Setup (one time)
cp .env.example .env
# Edit .env with your keys
./setup-mac.sh
source venv/bin/activate

# 2. Start servers
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &
python3 -m http.server 3000 &

# 3. Open browser
open http://localhost:3000/frontend/index.html
```

**Login with:**
- Email: `demo@bank.com`
- Password: `Demo@123456`

---

### Path B: Deploy to Railway.app (5 minutes)

```bash
# 1. Make sure .env is NOT committed
echo ".env" >> .gitignore

# 2. Go to railway.app
# 3. Connect GitHub repo: fnuAshutosh/llm_bank_usecase
# 4. Add environment variables from .env
# 5. Click Deploy
# Done! Your app is live 🎉
```

---

### Path C: Docker (Lightweight - 200MB)

```bash
# 1. Make sure .env exists
cp .env.example .env
# Edit with your keys

# 2. Build & run
docker build -f Dockerfile.light -t banking-llm:light .
docker run -p 8000:8000 --env-file .env banking-llm:light

# 3. Open http://localhost:8000/docs
```

---

## 📝 WHAT EACH FILE NEEDS

### ✅ Already Done (Committed to Git)
- All source code
- Frontend HTML/CSS/JS
- Docker configs
- Railway deployment config
- Mac setup script
- Documentation

### ⚠️ YOU NEED TO CREATE

1. **`.env`** - Copy from `.env.example` and fill in:
   - ✅ SUPABASE_URL (from supabase.com)
   - ✅ SUPABASE_KEY (from supabase.com)
   - ✅ OPENAI_API_KEY (optional - for GPT models)
   - ✅ PINECONE_API_KEY (optional - for vector search)

2. **Supabase Database** - Run:
   ```bash
   python setup_database.py
   ```

That's it! Everything else is ready.

---

## 🔑 API KEYS - What You Need

| Service | Required? | Free Tier | Purpose |
|---------|-----------|-----------|---------|
| **Supabase** | ✅ YES | ✅ Free 500MB | Database for users, accounts, transactions |
| **OpenAI** | ⚠️ Recommended | ✅ $5 credit | AI chat responses (GPT-4) |
| **Ollama** | 🔄 Alternative | ✅ Free | Local LLM (no API key needed) |
| **Pinecone** | ❌ Optional | ✅ Free 1 index | Vector search for similar queries |
| **Anthropic** | ❌ Optional | ✅ $5 credit | Claude AI (alternative to GPT) |

**Minimum to run**: Just Supabase + Ollama (both free!)

---

## 🎮 DEMO ACCOUNTS

The system auto-creates these test accounts:

```
Email: demo@bank.com
Password: Demo@123456
```

Or register new accounts through signup page!

---

## 📊 FEATURES INCLUDED

✅ **Frontend**
- Login/Signup pages
- BofA-style dashboard
- AI chat interface
- Real-time metrics
- Transaction history
- Account management

✅ **Backend API**
- 30+ REST endpoints
- JWT authentication
- Fraud detection
- KYC compliance
- PII detection
- Audit logging
- Rate limiting

✅ **AI/ML**
- OpenAI/Anthropic integration
- Local Ollama support
- Pinecone vector search
- QLoRA fine-tuning ready
- Sentiment analysis

✅ **Observability**
- Prometheus metrics
- Grafana dashboards
- Jaeger tracing
- Structured logging

✅ **Security**
- JWT tokens
- Password hashing
- PII redaction
- Audit logs
- Rate limiting
- CORS protection

---

## 🆘 TROUBLESHOOTING

### "ModuleNotFoundError"
```bash
pip install -r requirements/base.txt -r requirements/prod.txt
```

### "Connection refused" on port 8000
```bash
# API not running, start it:
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### "No such file: .env"
```bash
cp .env.example .env
# Then edit .env with your API keys
```

### "Supabase connection error"
```bash
# Check .env has correct URL and key
# Make sure you ran: python setup_database.py
```

---

## 📚 DOCUMENTATION

- **Getting Started**: [README.md](README.md)
- **Deployment Guide**: [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
- **API Docs**: http://localhost:8000/docs (when running)
- **Architecture**: [docs/02-ARCHITECTURE.md](docs/02-ARCHITECTURE.md)
- **Banking Use Cases**: [docs/03-BANKING-USECASES.md](docs/03-BANKING-USECASES.md)

---

## ✅ READY TO GO!

1. ✅ Code committed and pushed to GitHub
2. ⚠️ Copy `.env.example` to `.env` and fill in API keys
3. ⚠️ Setup Supabase database
4. ✅ Choose deployment path (Local/Railway/Docker)
5. 🚀 Start coding!

**Next step**: Copy .env and get your Supabase keys!

---

## 💬 NEED HELP?

- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for detailed instructions
- All commands are in this file
- Default credentials: `demo@bank.com` / `Demo@123456`

🎉 **Everything is ready to deploy!**
