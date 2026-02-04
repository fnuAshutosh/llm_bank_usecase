# Audit Fix Plan

**Date:** 2026-02-04  
**Estimated Time:** 6-12 hours  
**Priority Order:** Critical → High → Medium

---

## 🎯 Priority 1: Critical Issues (2-4 hours)

### Step 1: Create Missing `src/llm/models/` Directory (1-2 hours)

```bash
# Create directory structure
mkdir -p src/llm/models

# Create base adapter interface
touch src/llm/models/__init__.py
touch src/llm/models/base_adapter.py
touch src/llm/models/custom_transformer_adapter.py
touch src/llm/models/gemini_adapter.py
touch src/llm/models/qlora_adapter.py
```

**Implementation Tasks:**
- [ ] Create `base_adapter.py` with `ModelAdapter` abstract base class
- [ ] Implement `CustomTransformerAdapter` for custom models
- [ ] Implement `GeminiAdapter` for Google Gemini API
- [ ] Implement `QLoRASambaNovaAdapter` for QLoRA models
- [ ] Add proper error handling and logging
- [ ] Test imports: `python -c "from src.llm import pipeline"`

### Step 2: Install Development Dependencies (15 minutes)

```bash
# Install dev tools
pip install -r requirements/dev.txt

# Verify installation
pytest --version
ruff --version
black --version
mypy --version
```

**Verification:**
- [ ] pytest installed and working
- [ ] ruff linting available
- [ ] black formatting available
- [ ] mypy type checking available

### Step 3: Configure Environment (15 minutes)

```bash
# Copy example file
cp .env.example .env

# Edit .env with your values
nano .env  # or vim, code, etc.
```

**Required Configuration:**
- [ ] Set `SUPABASE_URL` and `SUPABASE_KEY` (if using Supabase)
- [ ] Set `DATABASE_URL` (if using local PostgreSQL)
- [ ] Set `REDIS_URL` (if using Redis)
- [ ] Set `SECRET_KEY` and `JWT_SECRET` (generate secure keys)
- [ ] Set LLM provider keys (TOGETHER_API_KEY, OPENAI_API_KEY, etc.)
- [ ] Set `PINECONE_API_KEY` (if using vector search)

**Generate Secure Keys:**
```bash
# Generate SECRET_KEY (32 characters)
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate JWT_SECRET
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Generate ENCRYPTION_KEY (32 bytes)
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

### Step 4: Setup Infrastructure (30-60 minutes)

**Option A: GitHub Codespaces (Recommended)**
```bash
# Already have PostgreSQL and Redis in Codespaces
# Just configure connection strings in .env
```

**Option B: Docker Compose (Local)**
```bash
# Start infrastructure services
docker-compose up -d postgres redis

# Verify services
docker-compose ps
docker-compose logs postgres
docker-compose logs redis
```

**Option C: Local Installation**
```bash
# macOS
brew install postgresql@15 redis
brew services start postgresql@15
brew services start redis

# Ubuntu/Debian
sudo apt install postgresql-15 redis-server
sudo systemctl start postgresql redis-server
```

**Database Setup:**
```bash
# Create database
createdb banking_llm

# Run migrations
python setup_database.py

# Verify
psql banking_llm -c "\dt"
```

---

## 🔧 Priority 2: High Priority Issues (4-6 hours)

### Step 5: Implement Authentication (2-3 hours)

**File:** `src/api/middleware/auth.py`

- [ ] Replace TODO with actual JWT validation
- [ ] Add token extraction from headers
- [ ] Implement user authentication
- [ ] Add role-based access control (RBAC)
- [ ] Write tests for auth middleware

**Example Implementation:**
```python
# Verify JWT token
# Check user permissions
# Add user context to request
```

### Step 6: Implement Rate Limiting (1 hour)

**File:** `src/api/middleware/rate_limit.py`

- [ ] Connect to Redis
- [ ] Implement sliding window rate limiting
- [ ] Add per-user and per-IP limits
- [ ] Return proper 429 responses
- [ ] Write tests

### Step 7: Add Database Queries (2 hours)

**Files:** 
- `src/services/banking_service.py`
- `src/api/routes/health.py`

- [ ] Implement account balance queries
- [ ] Implement transaction history
- [ ] Add fraud investigation workflow
- [ ] Add health check queries (database connection, Redis)
- [ ] Add metrics collection endpoint
- [ ] Write integration tests

### Step 8: Run and Fix Tests (1 hour)

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Fix failing tests
# Add new tests for new functionality
```

**Target:** >90% code coverage

---

## 📝 Priority 3: Medium Priority Issues (2-4 hours)

### Step 9: Complete TODO Items (2 hours)

- [ ] `chat_v2.py` - Implement escalation logic (webhook/queue)
- [ ] `admin.py` - Add admin role verification
- [ ] `inference.py` - Load config from file instead of hardcoding

### Step 10: Improve Test Coverage (1-2 hours)

- [ ] Add unit tests for all services
- [ ] Add integration tests for API endpoints
- [ ] Add E2E tests for critical flows
- [ ] Aim for >90% coverage

### Step 11: Fix Deprecation Warnings (30 minutes)

```bash
# Update transformers if needed
pip install --upgrade transformers torch

# Or suppress warnings if they're from dependencies
```

### Step 12: Update Dependencies (30 minutes)

```bash
# Update pip
pip install --upgrade pip

# Update other dependencies
pip list --outdated

# Update as needed (be careful with major versions)
```

---

## ✅ Verification Checklist

After completing all fixes, verify:

- [ ] All imports work without errors
- [ ] API starts successfully: `uvicorn src.api.main:app --reload`
- [ ] Health endpoint responds: `curl http://localhost:8000/health`
- [ ] All tests pass: `pytest tests/`
- [ ] Test coverage >90%: `pytest --cov=src`
- [ ] Code passes linting: `ruff check src/`
- [ ] Code is formatted: `black src/`
- [ ] Type checking passes: `mypy src/`
- [ ] No critical security issues: `bandit -r src/`
- [ ] Docker compose works: `docker-compose up`
- [ ] Database migrations run: `alembic upgrade head`
- [ ] Frontend loads: `http://localhost:8000/`
- [ ] API docs accessible: `http://localhost:8000/docs`

---

## 🚀 Quick Start After Fixes

```bash
# 1. Activate environment
source venv/bin/activate  # or your venv path

# 2. Install dependencies (if not done)
pip install -r requirements/dev.txt

# 3. Configure environment
cp .env.example .env
# Edit .env with your values

# 4. Start infrastructure (if using Docker)
docker-compose up -d postgres redis

# 5. Setup database
python setup_database.py

# 6. Run tests
pytest tests/

# 7. Start API
uvicorn src.api.main:app --reload --port 8000

# 8. Open browser
# http://localhost:8000/docs
```

---

## 📞 Need Help?

- **Issues File:** [AUDIT_ISSUES.md](AUDIT_ISSUES.md)
- **Documentation:** See `docs/` directory
- **Quick Start:** [QUICK_START.md](QUICK_START.md)
- **Setup Guide:** [HYBRID_SETUP_GUIDE.md](HYBRID_SETUP_GUIDE.md)

---

**Status:** Ready to implement fixes
