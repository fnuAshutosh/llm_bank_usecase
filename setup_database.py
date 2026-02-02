#!/usr/bin/env python3
"""
Create database tables in Supabase using Management API
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", SUPABASE_KEY)

# Extract project ref from URL
if SUPABASE_URL:
    # Format: https://PROJECT_REF.supabase.co
    project_ref = SUPABASE_URL.split("//")[1].split(".")[0]
else:
    project_ref = None

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

def save_sql_file():
    """Save SQL to file for manual execution"""
    print("\n📝 Saving SQL schema to file...")
    with open('database_schema.sql', 'w') as f:
        f.write(SQL_SCHEMA)
    print("   ✅ Saved to: database_schema.sql")
    return True

def print_instructions():
    """Print manual instructions"""
    print("\n" + "="*80)
    print("📋 MANUAL DATABASE SETUP INSTRUCTIONS")
    print("="*80)
    print("\n🔧 Option 1: Use Supabase Dashboard (Recommended)")
    print("   1. Go to: https://supabase.com/dashboard")
    print(f"   2. Select your project (ref: {project_ref})")
    print("   3. Click 'SQL Editor' in the left sidebar")
    print("   4. Click 'New Query'")
    print("   5. Copy the SQL below and paste it")
    print("   6. Click 'Run' button")
    print("\n" + "-"*80)
    print(SQL_SCHEMA)
    print("-"*80)
    
    print("\n🔧 Option 2: Use Local File")
    print("   1. The SQL has been saved to: database_schema.sql")
    print("   2. Go to Supabase Dashboard → SQL Editor")
    print("   3. Click 'New Query'")
    print("   4. Copy contents from database_schema.sql")
    print("   5. Paste and Run")
    
    print("\n🔧 Option 3: Use psql (if you have direct DB access)")
    print(f"   psql postgresql://postgres:[PASSWORD]@db.{project_ref}.supabase.co:5432/postgres < database_schema.sql")
    
    print("\n" + "="*80)
    print("⏭️  AFTER CREATING TABLES:")
    print("="*80)
    print("   1. Run: python test_api.py")
    print("   2. Or test registration:")
    print("      curl -X POST http://localhost:8000/api/v1/auth/register \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{")
    print("          \"email\": \"test@example.com\",")
    print("          \"password\": \"Test123!@#\",")
    print("          \"first_name\": \"Test\",")
    print("          \"last_name\": \"User\"")
    print("        }'")
    print("="*80 + "\n")

def main():
    print("="*80)
    print("🗄️  Supabase Database Setup")
    print("="*80)
    
    if not SUPABASE_URL or not project_ref:
        print("\n❌ Error: SUPABASE_URL not configured properly")
        return
    
    print(f"\n✅ Project: {project_ref}")
    print(f"✅ URL: {SUPABASE_URL}")
    
    print("\n⚠️  Note: Supabase REST API doesn't support CREATE TABLE commands")
    print("   Tables must be created via Dashboard or direct database connection")
    
    save_sql_file()
    print_instructions()

if __name__ == "__main__":
    main()
