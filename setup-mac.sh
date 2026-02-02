#!/bin/bash
# Mac Setup Script - Run this on your Mac

set -e

echo "🏦 Banking LLM - Mac Setup"
echo "=========================="
echo ""

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 not found. Installing..."
    brew install python@3.11
else
    echo "✅ Python 3.11 found"
fi

# Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
echo "   This may take 5-10 minutes..."
pip install --upgrade pip
pip install -r requirements/base.txt -r requirements/prod.txt

# Setup environment
echo ""
echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "✅ Created .env file - Please edit with your API keys"
fi

# Create logs directory
mkdir -p logs

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start the application:"
echo ""
echo "   1. Activate environment:"
echo "      source venv/bin/activate"
echo ""
echo "   2. Start API:"
echo "      uvicorn src.api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "   3. In another terminal, start frontend:"
echo "      python3 -m http.server 3000"
echo ""
echo "   4. Open browser:"
echo "      http://localhost:3000/frontend/index.html"
echo ""
echo "📖 See DEPLOYMENT_GUIDE.md for more options"
