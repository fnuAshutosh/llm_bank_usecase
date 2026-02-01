#!/bin/bash
# Setup script for GitHub Codespaces development environment
# Run this after Codespaces starts if needed

set -e

echo "🚀 Setting up Banking LLM Development Environment..."

# Update system
echo "📦 Updating system packages..."
apt-get update
apt-get upgrade -y

# Python dependencies already installed by postCreateCommand

# Create required directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p data
mkdir -p tests
mkdir -p docs

# Environment file
if [ ! -f .env.development ]; then
    echo "⚙️ Creating .env.development..."
    cat > .env.development << EOF
APP_NAME=banking-llm
APP_ENV=development
APP_DEBUG=true

# Database
DATABASE_URL=postgresql://banking_user:dev_password_change_in_prod@localhost:5432/banking_llm

# Redis
REDIS_URL=redis://localhost:6379/0

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json
LOG_FILE=logs/banking_llm.log

# Model
MODEL_PROVIDER=ollama
MODEL_NAME=llama2:7b
MODEL_TIMEOUT=30

# Security (change these!)
SECRET_KEY=dev-secret-key-change-in-production
ALLOWED_HOSTS=localhost,127.0.0.1,*.github.dev
EOF
    echo "✅ .env.development created"
fi

# Start services
echo "🐘 Starting PostgreSQL..."
docker-compose -f .devcontainer/docker-compose.yml up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check services
echo "✅ Services status:"
docker-compose -f .devcontainer/docker-compose.yml ps

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start FastAPI: uvicorn src.api.main:app --reload --port 8000"
echo "2. Run tests: pytest tests/ -v"
echo "3. Check health: curl http://localhost:8000/health/"
echo ""
