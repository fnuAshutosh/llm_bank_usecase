# Colab Error Fix: FileNotFoundError - models/tokenizer.json

## The Problem ❌
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/tokenizer.json'
```

**Cause**: The `models/` directory doesn't exist in your Colab environment.

---

## The Solution ✅

### **Option 1: Quick Fix (Use This NOW)**

Add this cell **RIGHT AFTER Step 2** (after installing dependencies):

```python
# ===== NEW CELL: Create Directories =====
import os

# Create all necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('data/finetuning', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print("✅ All directories created successfully!")
print(f"   - models/")
print(f"   - data/finetuning/")
print(f"   - logs/")
```

Then continue with Step 3. 

**This is the fastest way to fix it right now!**

---

### **Option 2: Update Your Notebook (Better)**

The notebook has been updated with auto-directory creation. If you re-download it:

1. Go to: https://github.com/fnuAshutosh/llm_bank_usecase
2. Download: `Custom_Banking_LLM_Training.ipynb`
3. Upload to Colab
4. Run Step 2 - it now automatically creates directories ✅

---

## How to Apply the Fix in Your Colab (Right Now)

### **If you're already running the notebook:**

**Just add this one cell** between Step 2 and Step 3:

```python
import os
os.makedirs('models', exist_ok=True)
os.makedirs('data/finetuning', exist_ok=True)
os.makedirs('logs', exist_ok=True)
print("✅ Directories created!")
```

Then re-run the tokenizer saving cell (Step 4) and it will work! ✅

---

## Gemini Prompt (If You Want AI Help)

Copy-paste this into Colab's Gemini chat:

```
I'm getting FileNotFoundError when saving files to models/tokenizer.json in Google Colab.

The error is:
FileNotFoundError: [Errno 2] No such file or directory: 'models/tokenizer.json'

I need Python code to:
1. Create the 'models' directory if it doesn't exist
2. Create the 'data/finetuning' directory
3. Create the 'logs' directory
4. Make sure these persist for the rest of my notebook

Give me a Python cell I can run in Colab to fix this.
```

**Expected Gemini Response:**
```python
import os

# Create directories
os.makedirs('models', exist_ok=True)
os.makedirs('data/finetuning', exist_ok=True)
os.makedirs('logs', exist_ok=True)

print("✅ Directories created!")
```

---

## Verification ✅

After running the directory creation cell, verify it worked:

```python
import os

# Check if directories exist
print("Directory Check:")
print(f"  models/ exists: {os.path.exists('models')}")
print(f"  data/finetuning/ exists: {os.path.exists('data/finetuning')}")
print(f"  logs/ exists: {os.path.exists('logs')}")

# List what's in each
print("\nContents:")
print(f"  models/: {os.listdir('models') if os.path.exists('models') else 'empty'}")
```

---

## Common Colab Path Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| FileNotFoundError: models/ | Directory doesn't exist | `os.makedirs('models', exist_ok=True)` |
| Permission denied | Wrong file permissions | Use `exist_ok=True` in makedirs |
| Path not found | Working directory changed | Use `os.getcwd()` to check current path |
| File not saved | Directory permissions | Make sure path is absolute or relative to cwd |

---

## Future-Proof Your Colab Notebook

Add this at the very beginning (Cell 1) to prevent all path issues:

```python
import os
import sys

# Setup working directory and paths
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)
os.makedirs('data/finetuning', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Add project to path
sys.path.insert(0, '/content/llm_bank_usecase')

print("✅ Environment setup complete")
print(f"   Working directory: {os.getcwd()}")
print(f"   Project path: /content/llm_bank_usecase")
```

---

## Summary

| Step | Action | Code |
|------|--------|------|
| 1 | Create directories | `os.makedirs('models', exist_ok=True)` |
| 2 | Verify they exist | `os.path.exists('models')` |
| 3 | Continue training | No changes needed after this |

**Time to fix**: 1 minute ⏱️  
**Difficulty**: Easy ✅  
**Impact**: Prevents all file-saving errors in Colab 🎯

---

## Still Having Issues?

If you get the error again, try:

```python
import os

# Debug: Print current working directory
print(f"Current directory: {os.getcwd()}")

# Force create with full path
full_path = os.path.join(os.getcwd(), 'models')
os.makedirs(full_path, exist_ok=True)

print(f"✅ Created: {full_path}")
print(f"✅ Exists: {os.path.exists(full_path)}")
```

This will show you exactly what's happening! 🔍

---

## Need Help?

Paste this Gemini prompt and Gemini will give you more specific advice:

```
I'm still getting FileNotFoundError in Colab even after creating directories.
The error happens when I try: tokenizer.save('models/tokenizer.json')
Can you help me debug this?
```

Gemini can help you troubleshoot further! ✨
