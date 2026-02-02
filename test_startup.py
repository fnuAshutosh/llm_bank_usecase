"""Simple API startup test"""

import sys

print("="*80)
print("🚀 Banking LLM API - Startup Test")
print("="*80)

try:
    print("\n1. Testing FastAPI import...")
    from fastapi import FastAPI
    print("   ✅ FastAPI imported")
    
    print("\n2. Testing configuration...")
    from src.utils.config import settings
    print(f"   ✅ Environment: {settings.ENVIRONMENT}")
    print(f"   ✅ LLM Provider: {settings.LLM_PROVIDER}")
    print(f"   ✅ Supabase URL: {settings.SUPABASE_URL}")
    
    print("\n3. Testing main app import...")
    from src.api.main import app
    print("   ✅ App initialized successfully")
    
    print("\n4. Checking registered routes...")
    routes = [route.path for route in app.routes]
    key_routes = [r for r in routes if r.startswith('/api/')]
    print(f"   ✅ {len(key_routes)} API routes registered")
    print(f"   Sample routes: {key_routes[:5]}")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80)
    print("\n🎉 Ready to start API server!")
    print("\nRun this command:")
    print("  python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
    print("\nThen visit:")
    print("  http://localhost:8000/docs")
    print("="*80)
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
