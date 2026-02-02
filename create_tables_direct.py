#!/usr/bin/env python3
"""
Create database tables using direct PostgreSQL connection
"""

import os

import psycopg2
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")

# Extract project ref from URL
if SUPABASE_URL:
    project_ref = SUPABASE_URL.split("//")[1].split(".")[0]
else:
    project_ref = None

# Construct direct database URL
DB_HOST = f"db.{project_ref}.supabase.co"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = SUPABASE_DB_PASSWORD
DB_PORT = 5432

SQL_SCHEMA = """
-- customers table
CREATE TABLE IF NOT EXISTS customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password TEXT NOT NULL,
    full_name VARCHAR(255) NOT NULL,
    phone_number VARCHAR(20),
    address TEXT,
    kyc_status VARCHAR(50) DEFAULT 'pending',
    kyc_level INTEGER DEFAULT 0,
    risk_score DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- accounts table
CREATE TABLE IF NOT EXISTS accounts (
    account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    customer_id UUID REFERENCES customers(customer_id) ON DELETE CASCADE,
    account_number VARCHAR(50) UNIQUE NOT NULL,
    account_type VARCHAR(50) NOT NULL,
    balance DECIMAL(15,2) DEFAULT 0.00,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- transactions table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id UUID REFERENCES accounts(account_id) ON DELETE CASCADE,
    transaction_type VARCHAR(50) NOT NULL,
    amount DECIMAL(15,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    status VARCHAR(20) DEFAULT 'pending',
    description TEXT,
    merchant VARCHAR(255),
    location VARCHAR(255),
    fraud_score DECIMAL(5,2) DEFAULT 0.00,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

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

def create_tables():
    """Create tables using direct PostgreSQL connection"""
    print("="*80)
    print("🗄️  Creating Database Tables via Direct PostgreSQL Connection")
    print("="*80)
    
    if not project_ref or not DB_PASSWORD or DB_PASSWORD == "your-db-password-here":
        print("\n❌ Error: Database credentials not configured")
        print(f"   Project Ref: {project_ref}")
        print(f"   DB Password: {'***' if DB_PASSWORD and DB_PASSWORD != 'your-db-password-here' else 'NOT SET'}")
        print("\n💡 Check your .env file:")
        print("   SUPABASE_URL=https://xxx.supabase.co")
        print("   SUPABASE_DB_PASSWORD=your_actual_password")
        print("\n   Get password from: Supabase Dashboard → Settings → Database")
        return False
    
    connection_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} sslmode=require"
    
    print(f"\n✅ Connecting to: {DB_HOST}")
    print(f"✅ Database: {DB_NAME}")
    print(f"✅ User: {DB_USER}")
    
    try:
        # Connect to database
        print("\n📡 Establishing connection...")
        conn = psycopg2.connect(connection_string)
        conn.autocommit = True
        cursor = conn.cursor()
        print("   ✅ Connected successfully!")
        
        # Execute SQL schema
        print("\n📝 Executing SQL schema...")
        
        # Split into individual statements
        statements = [s.strip() + ';' for s in SQL_SCHEMA.split(';') if s.strip()]
        
        success_count = 0
        for i, statement in enumerate(statements, 1):
            try:
                cursor.execute(statement)
                # Get statement type
                stmt_type = statement.split()[0:2]
                print(f"   ✅ Statement {i}/{len(statements)}: {' '.join(stmt_type)}")
                success_count += 1
            except Exception as e:
                print(f"   ⚠️  Statement {i} error: {str(e)[:100]}")
        
        print(f"\n📊 Results: {success_count}/{len(statements)} statements executed successfully")
        
        # Verify tables
        print("\n🔍 Verifying tables...")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        expected_tables = {'customers', 'accounts', 'transactions', 'fraud_alerts', 
                          'conversations', 'messages', 'audit_logs'}
        found_tables = {t[0] for t in tables}
        
        for table in expected_tables:
            if table in found_tables:
                print(f"   ✅ Table '{table}' created")
            else:
                print(f"   ❌ Table '{table}' missing")
        
        # Get row counts
        print("\n📊 Table Statistics:")
        for table in sorted(expected_tables & found_tables):
            cursor.execute(f"SELECT COUNT(*) FROM {table};")
            count = cursor.fetchone()[0]
            print(f"   {table}: {count} rows")
        
        cursor.close()
        conn.close()
        
        if expected_tables.issubset(found_tables):
            print("\n" + "="*80)
            print("✅ SUCCESS! All 7 tables created successfully!")
            print("="*80)
            print("\n🎉 Database is ready! You can now:")
            print("   1. Run: python test_api.py")
            print("   2. Test registration:")
            print("      curl -X POST http://localhost:8000/api/v1/auth/register \\")
            print("        -H 'Content-Type: application/json' \\")
            print("        -d '{")
            print('          "email": "test@example.com",')
            print('          "password": "Test123!@#",')
            print('          "first_name": "Test",')
            print('          "last_name": "User"')
            print("        }'")
            print("="*80)
            return True
        else:
            print("\n⚠️  Some tables are missing. Check errors above.")
            return False
            
    except psycopg2.OperationalError as e:
        print(f"\n❌ Connection Error: {e}")
        print("\n💡 Possible issues:")
        print("   1. Wrong database password - Check Supabase Dashboard → Settings → Database")
        print("   2. Database not accessible from Codespaces")
        print("   3. SSL/firewall issues")
        print("\n💡 Alternative: Use Supabase Dashboard SQL Editor")
        print("   1. Go to: https://supabase.com/dashboard/project/vdrcjlglcxbrbfhxmiai/sql")
        print("   2. Copy SQL from database_schema.sql")
        print("   3. Run the query")
        return False
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Alternative: Use Supabase Dashboard SQL Editor")
        return False

if __name__ == "__main__":
    create_tables()
