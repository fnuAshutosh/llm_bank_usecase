#!/bin/bash
# Initialize Banking LLM project with GitHub
# Run this once to set up GitHub repository

set -e

echo "🚀 Banking LLM - GitHub Repository Setup"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git initialized"
else
    echo "✅ Git already initialized"
fi

# Configure git if not done
if [ -z "$(git config user.name)" ]; then
    echo ""
    echo "⚙️ Configuring Git..."
    read -p "Enter your name: " git_name
    read -p "Enter your email: " git_email
    git config user.name "$git_name"
    git config user.email "$git_email"
    echo "✅ Git configured"
fi

# Add files
echo ""
echo "📝 Adding files to Git..."
git add .

# Create initial commit
git commit -m "Initial Banking LLM project with FastAPI, PII detection, audit logging, and Codespaces configuration"

echo "✅ Initial commit created"

# Get current remote
current_remote=$(git remote get-url origin 2>/dev/null || echo "")

if [ -z "$current_remote" ]; then
    echo ""
    echo "🔗 Adding GitHub remote..."
    read -p "Enter your GitHub username: " github_user
    repo_name="banking-llm"
    
    git remote add origin "https://github.com/$github_user/$repo_name.git"
    echo "✅ Remote added: https://github.com/$github_user/$repo_name.git"
    
    echo ""
    echo "📤 Ready to push! Run:"
    echo "   git branch -M main"
    echo "   git push -u origin main"
else
    echo ""
    echo "ℹ️ Remote already set: $current_remote"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Create empty repository on GitHub (https://github.com/new)"
echo "2. Run: git push -u origin main"
echo "3. Enable GitHub Actions (already configured)"
echo "4. Create first Codespace!"
echo ""
