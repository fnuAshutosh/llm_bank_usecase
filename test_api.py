"""Comprehensive test script - Test all functionality"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal

import httpx

from src.api.main import app

BASE_URL = "http://test"
USE_ASGI = True

# Colors for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_section(title):
    """Print section header"""
    print(f"\n{BLUE}{'=' * 80}")
    print(f"  {title}")
    print(f"{'=' * 80}{RESET}\n")


def print_success(msg):
    """Print success message"""
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg):
    """Print error message"""
    print(f"{RED}❌ {msg}{RESET}")


def print_info(msg):
    """Print info message"""
    print(f"{YELLOW}ℹ️  {msg}{RESET}")


def get_client(timeout: float | None = None) -> httpx.AsyncClient:
    """Return an AsyncClient for in-process or external testing."""
    if USE_ASGI:
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url=BASE_URL, timeout=timeout)
    return httpx.AsyncClient(base_url=BASE_URL, timeout=timeout)


async def test_health_check():
    """Test health check endpoint"""
    print_section("Health Check")
    
    async with get_client() as client:
        try:
            response = await client.get(f"{BASE_URL}/")
            if response.status_code == 200:
                data = response.json()
                print_success(f"Health Check: {data.get('status', 'unknown')}")
                print_info(f"Response: {json.dumps(data, indent=2)}")
                return True
            else:
                print_error(f"Health check failed: {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Health check error: {e}")
            return False


async def test_user_registration():
    """Test user registration"""
    print_section("User Registration")
    
    user_data = {
        "email": "john.doe@example.com",
        "password": "SecurePassword123!",
        "first_name": "John",
        "last_name": "Doe",
        "phone_number": "+1234567890",
        "date_of_birth": "1990-01-15"
    }
    
    async with get_client() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/auth/register",
                json=user_data
            )
            
            if response.status_code == 201:
                data = response.json()
                print_success(f"User registered: {data['customer_id']}")
                print_info(f"Email: {data['email']}")
                print_info(f"KYC Status: {data['kyc_status']}")
                return data['customer_id']
            elif response.status_code == 400:
                print_error("User already exists (expected if running tests multiple times)")
                return None
            else:
                print_error(f"Registration failed: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Registration error: {e}")
            return None


async def test_user_login(email, password):
    """Test user login"""
    print_section("User Login")
    
    login_data = {
        "username": email,  # OAuth2 uses 'username' field
        "password": password
    }
    
    async with get_client() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/auth/token",
                data=login_data  # OAuth2 uses form data
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success("Login successful!")
                print_info(f"Access Token: {data['access_token'][:50]}...")
                print_info(f"Token Type: {data['token_type']}")
                return data['access_token']
            else:
                print_error(f"Login failed: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Login error: {e}")
            return None


async def test_create_account(token):
    """Test account creation"""
    print_section("Create Bank Account")
    
    account_data = {
        "account_type": "checking",
        "initial_balance": "1000.00",
        "currency": "USD"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with get_client() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/accounts/",
                json=account_data,
                headers=headers
            )
            
            if response.status_code == 201:
                data = response.json()
                print_success(f"Account created: {data['account_number']}")
                print_info(f"Type: {data['account_type']}")
                print_info(f"Balance: ${data['balance']}")
                return data['account_id']
            else:
                print_error(f"Account creation failed: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Account creation error: {e}")
            return None


async def test_list_accounts(token):
    """Test listing accounts"""
    print_section("List Accounts")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with get_client() as client:
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/accounts/",
                headers=headers
            )
            
            if response.status_code == 200:
                accounts = response.json()
                print_success(f"Found {len(accounts)} accounts")
                for acc in accounts:
                    print_info(f"  {acc['account_type']}: ${acc['balance']} ({acc['account_number']})")
                return accounts
            else:
                print_error(f"List accounts failed: {response.status_code}")
                return []
        except Exception as e:
            print_error(f"List accounts error: {e}")
            return []


async def test_create_transaction(token, account_id):
    """Test transaction creation"""
    print_section("Create Transaction")
    
    transaction_data = {
        "from_account_id": account_id,
        "transaction_type": "deposit",
        "amount": "500.00",
        "description": "Test deposit"
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with get_client() as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/transactions/",
                json=transaction_data,
                headers=headers
            )
            
            if response.status_code == 201:
                data = response.json()
                print_success(f"Transaction created: {data['transaction_id']}")
                print_info(f"Type: {data['transaction_type']}")
                print_info(f"Amount: ${data['amount']}")
                print_info(f"Status: {data['status']}")
                print_info(f"Fraud Score: {data['fraud_score']}")
                return data['transaction_id']
            else:
                print_error(f"Transaction failed: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Transaction error: {e}")
            return None


async def test_fraud_detection(token, transaction_id):
    """Test fraud detection analysis"""
    print_section("Fraud Detection Analysis")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with get_client() as client:
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/transactions/{transaction_id}/fraud-analysis",
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success(f"Fraud analysis completed")
                print_info(f"Risk Score: {data['risk_score']}")
                print_info(f"Risk Level: {data['risk_level']}")
                print_info(f"Signals detected: {len(data['fraud_signals'])}")
                for signal in data['fraud_signals']:
                    print_info(f"  - {signal['signal_name']}: {signal['description']}")
                return True
            else:
                print_error(f"Fraud analysis failed: {response.status_code}")
                return False
        except Exception as e:
            print_error(f"Fraud analysis error: {e}")
            return False


async def test_chat(token):
    """Test chat endpoint with LLM"""
    print_section("Chat with LLM")
    
    chat_data = {
        "message": "What is my current account balance?",
        "conversation_id": None
    }
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with get_client(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{BASE_URL}/api/v1/chat/",
                json=chat_data,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                print_success("Chat response received!")
                print_info(f"Response: {data['response']}")
                print_info(f"Model: {data['model']}")
                print_info(f"Latency: {data['latency_ms']}ms")
                print_info(f"PII Detected: {data['pii_detected']}")
                return data['conversation_id']
            else:
                print_error(f"Chat failed: {response.status_code}")
                print_error(f"Response: {response.text}")
                return None
        except Exception as e:
            print_error(f"Chat error: {e}")
            return None


async def test_admin_endpoints(token):
    """Test admin endpoints"""
    print_section("Admin Endpoints")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    async with get_client() as client:
        # List customers
        try:
            response = await client.get(
                f"{BASE_URL}/api/v1/admin/customers",
                headers=headers
            )
            
            if response.status_code == 200:
                customers = response.json()
                print_success(f"Found {len(customers)} customers")
                for cust in customers[:3]:  # Show first 3
                    print_info(f"  {cust['first_name']} {cust['last_name']}: {cust['kyc_status']}")
            else:
                print_error(f"List customers failed: {response.status_code}")
        except Exception as e:
            print_error(f"Admin endpoints error: {e}")


async def run_all_tests():
    """Run all tests in sequence"""
    print(f"\n{BLUE}{'=' * 80}")
    print(f"  🧪 Banking LLM API - Comprehensive Test Suite")
    print(f"  📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 80}{RESET}\n")
    
    # Test 1: Health Check
    health_ok = await test_health_check()
    if not health_ok:
        print_error("Health check failed - stopping tests")
        return
    
    # Test 2: User Registration
    customer_id = await test_user_registration()
    
    # Test 3: User Login (use existing user if registration failed)
    token = await test_user_login("john.doe@example.com", "SecurePassword123!")
    if not token:
        print_error("Login failed - stopping tests")
        return
    
    # Test 4: Create Account
    account_id = await test_create_account(token)
    if not account_id:
        print_error("Account creation failed - skipping transaction tests")
    else:
        # Test 5: List Accounts
        accounts = await test_list_accounts(token)
        
        # Test 6: Create Transaction
        transaction_id = await test_create_transaction(token, account_id)
        
        # Test 7: Fraud Detection
        if transaction_id:
            await test_fraud_detection(token, transaction_id)
    
    # Test 8: Chat with LLM
    conversation_id = await test_chat(token)
    
    # Test 9: Admin Endpoints
    await test_admin_endpoints(token)
    
    # Summary
    print_section("Test Summary")
    print_success("All tests completed!")
    print_info("Check logs above for detailed results")


if __name__ == "__main__":
    print(f"\n{YELLOW}Starting API tests...{RESET}")
    print(f"{YELLOW}Make sure the API is running on {BASE_URL}{RESET}\n")
    
    try:
        asyncio.run(run_all_tests())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Tests interrupted by user{RESET}")
    except Exception as e:
        print(f"\n{RED}Test suite error: {e}{RESET}")
