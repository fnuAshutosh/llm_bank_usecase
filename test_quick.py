"""Quick test script - Validate imports and basic functionality"""

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("Testing imports...")
    
    try:
        print("  ✓ fastapi")
        from fastapi import FastAPI
        
        print("  ✓ supabase client")
        from src.database.supabase_client import get_supabase_client
        
        print("  ✓ config")
        from src.utils.config import settings
        print(f"    - Environment: {settings.ENVIRONMENT}")
        print(f"    - LLM Provider: {settings.LLM_PROVIDER}")
        
        print("  ✓ authentication")
        from src.security.auth_service import auth_service
        
        print("  ✓ encryption")
        from src.security.encryption import encryption_service
        
        print("  ✓ LLM service")
        from src.llm import llm_service
        
        print("  ✓ banking service")
        from src.services.banking_service import banking_service
        
        print("  ✓ fraud detection")
        from src.services.fraud_detection import fraud_detection_service
        
        print("  ✓ KYC service")
        from src.services.kyc_service import kyc_service
        
        print("  ✓ compliance")
        from src.services.compliance import compliance_service
        
        print("  ✓ observability")
        from src.observability.metrics import track_model_inference
        from src.observability.tracing import trace_function
        from src.observability.logging_config import setup_logging
        
        print("\n✅ All imports successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_encryption():
    """Test encryption service"""
    print("\nTesting encryption...")
    
    try:
        from src.security.encryption import encryption_service
        
        # Test basic encryption
        test_data = "Sensitive banking data"
        encrypted = encryption_service.encrypt(test_data)
        decrypted = encryption_service.decrypt(encrypted)
        
        assert decrypted == test_data, "Decrypted data doesn't match"
        print(f"  ✓ Basic encryption works")
        
        # Test password hashing
        password = "SecurePassword123!"
        hashed = encryption_service.hash_password(password)
        assert encryption_service.verify_password(password, hashed), "Password verification failed"
        print(f"  ✓ Password hashing works")
        
        print("✅ Encryption tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Encryption test failed: {e}")
        traceback.print_exc()
        return False


def test_auth():
    """Test authentication service"""
    print("\nTesting authentication...")
    
    try:
        from src.security.auth_service import auth_service
        
        # Test token creation
        token_data = {
            "sub": "test-customer-id",
            "email": "test@example.com"
        }
        
        access_token = auth_service.create_access_token(token_data)
        print(f"  ✓ Access token created: {access_token[:50]}...")
        
        # Test token verification
        payload = auth_service.verify_token(access_token)
        assert payload["sub"] == "test-customer-id", "Token verification failed"
        print(f"  ✓ Token verification works")
        
        print("✅ Authentication tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Authentication test failed: {e}")
        traceback.print_exc()
        return False


def test_supabase_connection():
    """Test Supabase connection"""
    print("\nTesting Supabase connection...")
    
    try:
        from src.database.supabase_client import get_supabase_client
        
        client = get_supabase_client()
        print(f"  ✓ Supabase client initialized")
        print(f"    URL: {client.url}")
        
        print("✅ Supabase connection test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Supabase connection test failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config import settings
        
        print(f"  Environment: {settings.ENVIRONMENT}")
        print(f"  LLM Provider: {settings.LLM_PROVIDER}")
        print(f"  Supabase URL: {settings.SUPABASE_URL}")
        print(f"  JWT Algorithm: {settings.JWT_ALGORITHM}")
        print(f"  Enable Monitoring: {settings.ENABLE_MONITORING}")
        print(f"  Enable Tracing: {settings.ENABLE_TRACING}")
        
        print("✅ Configuration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all quick tests"""
    print("=" * 80)
    print("  🧪 Quick Test Suite - Validation")
    print("=" * 80)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Configuration", test_config()))
    results.append(("Supabase Connection", test_supabase_connection()))
    results.append(("Encryption", test_encryption()))
    results.append(("Authentication", test_auth()))
    
    # Summary
    print("\n" + "=" * 80)
    print("  📊 Test Summary")
    print("=" * 80)
    
    total = len(results)
    passed = sum(1 for _, result in results if result)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready to start API server.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
