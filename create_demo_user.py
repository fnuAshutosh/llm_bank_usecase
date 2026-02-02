
import asyncio
import os
from dotenv import load_dotenv

# Load env before imports
load_dotenv()

from src.database.supabase_client import get_supabase_client
from src.security.encryption import encryption_service

async def main():
    print("Checking for demo user...")
    client = get_supabase_client()
    
    # User 1: matches frontend default
    email = "demo@bankai.com"
    password = "demo123"
    
    existing = await client.get_customer_by_email(email)
    if existing:
        print(f"✅ User {email} already exists.")
    else:
        print(f"Creating user {email}...")
        hashed = encryption_service.hash_password(password)
        user_data = {
            "email": email,
            "hashed_password": hashed,
            "full_name": "Demo User",
            "first_name": "Demo",
            "last_name": "User",
            "is_active": True,
            "kyc_status": "verified",
            "risk_score": 0.0
        }
        try:
            res = await client.create_customer(user_data)
            print(f"✅ Created user: {res.get('customer_id')}")
        except Exception as e:
            print(f"❌ Error creating user {email}: {e}")

    # User 2: matches START_HERE.md (backup)
    email2 = "demo@bank.com"
    password2 = "Demo@123456"
    
    existing2 = await client.get_customer_by_email(email2)
    if existing2:
        print(f"✅ User {email2} already exists.")
    else:
        print(f"Creating user {email2}...")
        hashed2 = encryption_service.hash_password(password2)
        user_data2 = {
            "email": email2,
            "hashed_password": hashed2,
            "full_name": "Bank Demo",
            "first_name": "Bank",
            "last_name": "Demo",
            "is_active": True,
            "kyc_status": "verified"
        }
        try:
            res = await client.create_customer(user_data2)
            print(f"✅ Created user: {res.get('customer_id')}")
        except Exception as e:
            print(f"❌ Error creating user {email2}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
