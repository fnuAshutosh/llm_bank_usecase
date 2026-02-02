#!/usr/bin/env python3
"""
Quick API Test - Verify basic endpoints work
"""

import json

import requests

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n1. Testing Health Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("   ✅ Health check passed")

def test_docs():
    """Test API documentation"""
    print("\n2. Testing API Documentation...")
    response = requests.get(f"{BASE_URL}/docs")
    print(f"   Status: {response.status_code}")
    assert response.status_code == 200
    print("   ✅ API docs accessible")

def test_openapi():
    """Test OpenAPI spec"""
    print("\n3. Testing OpenAPI Specification...")
    response = requests.get(f"{BASE_URL}/openapi.json")
    print(f"   Status: {response.status_code}")
    data = response.json()
    routes = list(data["paths"].keys())
    print(f"   ✅ {len(routes)} routes defined")
    print(f"   Sample routes: {routes[:5]}")

def test_registration():
    """Test user registration endpoint (will fail until DB is set up)"""
    print("\n4. Testing Registration Endpoint...")
    print("   NOTE: This will fail until Supabase tables are created")
    
    payload = {
        "email": "test@example.com",
        "password": "Test123!@#",
        "first_name": "Test",
        "last_name": "User",
        "phone_number": "+1234567890"
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/api/v1/auth/register",
            json=payload
        )
        print(f"   Status: {response.status_code}")
        print(f"   Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 500:
            print("   ⚠️  Expected failure - Database tables not created yet")
            print("   ℹ️  Run SQL commands in Supabase to create tables")
        elif response.status_code == 201:
            print("   ✅ Registration successful!")
        else:
            print(f"   ⚠️  Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    print("="*80)
    print("🧪 Quick API Test")
    print("="*80)
    
    try:
        test_health()
        test_docs()
        test_openapi()
        test_registration()
        
        print("\n" + "="*80)
        print("✅ Basic API Tests Complete!")
        print("="*80)
        print("\nNext Steps:")
        print("1. Create database tables in Supabase SQL Editor")
        print("2. Run: python test_api.py")
        print("3. View API docs: http://localhost:8000/docs")
        print("="*80)
        
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to API server")
        print("   Start server with: python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload")
