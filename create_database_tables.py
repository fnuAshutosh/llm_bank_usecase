#!/usr/bin/env python3
"""
Create database tables in Supabase
"""

import os

import requests
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
SUPABASE_ACCESS_TOKEN = os.getenv("SUPABASE_ACCESS_TOKEN")

# Extract project ref from URL
PROJECT_REF = SUPABASE_URL.split("//")[1].split(".")[0] if SUPABASE_URL else None

# SQL schema
SQL_SCHEMA = """
-- customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name VARCHAR(255),
    first_name VARCHAR(255),
    last_name VARCHAR(255),
    phone_number VARCHAR(20),
    date_of_birth DATE,
    address TEXT,
    kyc_status VARCHAR(50) DEFAULT 'pending',
    kyc_level INTEGER DEFAULT 0,
    risk_score DECIMAL(5,2) DEFAULT 0.00,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- ensure missing columns exist on existing deployments
ALTER TABLE customers ADD COLUMN IF NOT EXISTS first_name VARCHAR(255);
ALTER TABLE customers ADD COLUMN IF NOT EXISTS last_name VARCHAR(255);
ALTER TABLE customers ADD COLUMN IF NOT EXISTS date_of_birth DATE;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- accounts table
CREATE TABLE IF NOT EXISTS accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    account_number VARCHAR(50) UNIQUE NOT NULL,
    account_type VARCHAR(50) NOT NULL,
    balance DECIMAL(15,2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'active',
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE accounts ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE;

-- transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES accounts(account_id) ON DELETE CASCADE,
    from_account_id UUID,
    to_account_id UUID,
    transaction_type VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'pending',
    description TEXT,
    merchant VARCHAR(255),
    merchant_name VARCHAR(255),
    merchant_category VARCHAR(255),
    location VARCHAR(255),
    fraud_score DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE transactions ADD COLUMN IF NOT EXISTS from_account_id UUID;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS to_account_id UUID;
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_name VARCHAR(255);
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS merchant_category VARCHAR(255);

-- fraud_alerts table
CREATE TABLE IF NOT EXISTS fraud_alerts (
    alert_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    account_id UUID REFERENCES accounts(account_id) ON DELETE SET NULL,
    transaction_id UUID REFERENCES transactions(transaction_id) ON DELETE SET NULL,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'active',
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- conversations table
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    status VARCHAR(20) DEFAULT 'active',
    escalation_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- messages table
CREATE TABLE IF NOT EXISTS messages (
    message_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- audit_logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES customers(customer_id) ON DELETE SET NULL,
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    ip_address VARCHAR(45),
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(email);
CREATE INDEX IF NOT EXISTS idx_accounts_customer_id ON accounts(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_account_id ON transactions(account_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_customer_id ON fraud_alerts(customer_id);
CREATE INDEX IF NOT EXISTS idx_conversations_customer_id ON conversations(customer_id);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_customer_id ON audit_logs(customer_id);
"""

def _run_sql_via_management_api(sql: str) -> bool:
    if not SUPABASE_ACCESS_TOKEN:
        print("❌ SUPABASE_ACCESS_TOKEN not set. Management API cannot be used.")
        return False

    if not PROJECT_REF:
        print("❌ SUPABASE_URL is missing or invalid.")
        return False

    api_url = f"https://api.supabase.com/v1/projects/{PROJECT_REF}/database/query"
    headers = {
        "Authorization": f"Bearer {SUPABASE_ACCESS_TOKEN}",
        "Content-Type": "application/json",
    }

    print(f"\n✅ Using Supabase Management API: {api_url}")

    statements = [s.strip() for s in sql.split(";") if s.strip()]
    for i, statement in enumerate(statements, 1):
        payload = {"query": statement + ";"}
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code >= 400:
            print(f"   ❌ Statement {i}/{len(statements)} failed: {response.text[:200]}")
            return False
        print(f"   ✅ Statement {i}/{len(statements)} executed")

    return True


def create_tables():
    """Create all database tables"""
    print("="*80)
    print("🗄️  Creating Database Tables in Supabase")
    print("="*80)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        return False
    
    print(f"\n✅ Connecting to Supabase: {SUPABASE_URL}")
    
    try:
        # Create Supabase client (prefer service role for verification)
        client_key = SUPABASE_SERVICE_KEY or SUPABASE_KEY
        supabase = create_client(SUPABASE_URL, client_key)

        print("\n📝 Executing SQL schema via Supabase Management API...")
        executed = _run_sql_via_management_api(SQL_SCHEMA)
        if not executed:
            print("\n❌ Failed to execute SQL via Management API.")
            print("   This API requires a Supabase Personal Access Token (PAT).")
            print("\n✅ Next step to enable full automation:")
            print("   1) Create a PAT at https://supabase.com/dashboard/account/tokens")
            print("   2) Add to .env: SUPABASE_ACCESS_TOKEN=your_pat_here")
            print("   3) Re-run: python create_database_tables.py")
            return False
        
        # Verify tables exist by trying to query them
        print("\n🔍 Verifying tables...")
        tables_to_check = ['customers', 'accounts', 'transactions', 'fraud_alerts', 
                          'conversations', 'messages', 'audit_logs']
        
        verified = 0
        for table in tables_to_check:
            try:
                # Try to select from table (will fail if doesn't exist)
                result = supabase.table(table).select("*").limit(1).execute()
                print(f"   ✅ Table '{table}' exists")
                verified += 1
            except Exception as e:
                print(f"   ❌ Table '{table}' not found: {str(e)[:100]}")
        
        print(f"\n{'='*80}")
        if verified == len(tables_to_check):
            print("✅ SUCCESS! All 7 tables created and verified!")
            print("="*80)
            print("\n🎉 Database is ready! You can now:")
            print("   1. Run: python test_api.py")
            print("   2. Register users and test all endpoints")
            print("   3. View data in Supabase Dashboard")
            print("="*80)
            return True
        elif verified > 0:
            print(f"⚠️  PARTIAL SUCCESS: {verified}/{len(tables_to_check)} tables verified")
            print("="*80)
            print("\n💡 Some tables may already exist or need manual creation")
            print("   Check Supabase Dashboard → SQL Editor")
            print("="*80)
            return True
        else:
            print("❌ FAILED: Unable to create tables via API")
            print("="*80)
            print("\n💡 Alternative: Create tables manually:")
            print("   1. Go to Supabase Dashboard")
            print("   2. Click 'SQL Editor'")
            print("   3. Copy SQL from API_STARTUP_SUCCESS.md")
            print("   4. Run the SQL commands")
            print("="*80)
            return False
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Please create tables manually in Supabase Dashboard")
        return False

if __name__ == "__main__":
    create_tables()
