# Data Models - Banking LLM System

**Database:** PostgreSQL 15+  
**ORM:** SQLAlchemy 2.0  
**Schema Version:** 1.0  
**Last Updated:** February 1, 2026

---

## Entity Relationship Diagram (ERD)

```
┌─────────────────┐
│   customers     │
├─────────────────┤
│ customer_id (PK)│
│ email           │
│ phone           │
│ name            │
│ date_of_birth   │
│ created_at      │
└────────┬────────┘
         │ 1
         │
         │ N
┌────────▼─────────────┐
│   accounts          │
├─────────────────────┤
│ account_id (PK)     │
│ customer_id (FK)    │
│ account_type        │
│ balance             │
│ status              │
│ created_at          │
└────────┬─────────────┘
         │ 1
         │
         │ N
┌────────▼──────────────┐
│   transactions       │
├──────────────────────┤
│ transaction_id (PK)  │
│ account_id (FK)      │
│ amount               │
│ transaction_type     │
│ status               │
│ created_at           │
└──────────────────────┘


┌─────────────────┐
│  users (staff)  │
├─────────────────┤
│ user_id (PK)    │
│ email           │
│ role            │
│ department      │
│ created_at      │
└────────┬────────┘
         │ 1
         │
         │ N
┌────────▼──────────────┐
│   audit_logs        │
├──────────────────────┤
│ log_id (PK)          │
│ user_id (FK)         │
│ action               │
│ entity_type          │
│ entity_id            │
│ changes              │
│ created_at           │
└──────────────────────┘

```

---

## Core Tables

### 1. Customers Table

```sql
CREATE TABLE customers (
  customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) NOT NULL UNIQUE,
  phone VARCHAR(20),
  
  -- Personal Information
  first_name VARCHAR(100) NOT NULL,
  last_name VARCHAR(100) NOT NULL,
  middle_name VARCHAR(100),
  date_of_birth DATE NOT NULL,
  gender CHAR(1), -- M, F, O, N
  
  -- Address
  street_address VARCHAR(255) NOT NULL,
  city VARCHAR(100) NOT NULL,
  state_province VARCHAR(100),
  postal_code VARCHAR(20) NOT NULL,
  country_code CHAR(2) NOT NULL, -- ISO 3166-1 alpha-2
  
  -- KYC/Compliance
  kyc_status ENUM('pending', 'approved', 'rejected', 'expired') DEFAULT 'pending',
  kyc_verification_date TIMESTAMP,
  aml_status ENUM('clear', 'flagged', 'blocked') DEFAULT 'clear',
  pep_status ENUM('clear', 'match', 'rejected') DEFAULT 'clear',
  risk_level ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
  
  -- SSN/Tax ID (encrypted)
  ssn_encrypted BYTEA NOT NULL,
  ssn_hash VARCHAR(256) UNIQUE NOT NULL, -- SHA-256 hash for uniqueness check
  
  -- Identification Documents
  id_document_type ENUM('passport', 'drivers_license', 'national_id'),
  id_document_number VARCHAR(50),
  id_expiry_date DATE,
  
  -- Status
  status ENUM('active', 'inactive', 'suspended', 'closed') DEFAULT 'active',
  customer_segment ENUM('retail', 'sme', 'corporate') DEFAULT 'retail',
  
  -- Timestamps
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  deleted_at TIMESTAMP, -- Soft delete
  
  -- Metadata
  preferred_language VARCHAR(10) DEFAULT 'en',
  marketing_consent BOOLEAN DEFAULT false,
  communication_preferences JSONB,
  
  CONSTRAINT age_check CHECK (
    DATE_PART('year', AGE(date_of_birth)) >= 18
  )
);

-- Indexes
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_ssn_hash ON customers(ssn_hash);
CREATE INDEX idx_customers_status ON customers(status);
CREATE INDEX idx_customers_kyc_status ON customers(kyc_status);
CREATE INDEX idx_customers_created_at ON customers(created_at);
```

**Python Pydantic Model:**
```python
from pydantic import BaseModel, EmailStr
from datetime import date
from typing import Optional

class CustomerCreate(BaseModel):
    email: EmailStr
    phone: Optional[str] = None
    first_name: str
    last_name: str
    date_of_birth: date
    street_address: str
    city: str
    state_province: Optional[str]
    postal_code: str
    country_code: str
    ssn: str  # Encrypted before storage
    
    class Config:
        schema_extra = {
            "example": {
                "email": "john@example.com",
                "first_name": "John",
                "last_name": "Doe",
                "date_of_birth": "1985-03-20",
                "ssn": "XXX-XX-XXXX"
            }
        }

class Customer(CustomerCreate):
    customer_id: str
    kyc_status: str
    aml_status: str
    risk_level: str
    status: str
    created_at: datetime
    updated_at: datetime
```

---

### 2. Accounts Table

```sql
CREATE TABLE accounts (
  account_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID NOT NULL REFERENCES customers(customer_id),
  
  -- Account Details
  account_number VARCHAR(20) NOT NULL UNIQUE,
  account_type ENUM(
    'checking',
    'savings',
    'money_market',
    'credit_card',
    'loan',
    'investment'
  ) NOT NULL,
  account_subtype VARCHAR(50),
  
  -- Balances
  balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
  available_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
  pending_balance DECIMAL(15, 2) NOT NULL DEFAULT 0,
  credit_limit DECIMAL(15, 2),
  
  -- Rates & Terms
  interest_rate DECIMAL(5, 3),
  annual_percentage_rate DECIMAL(5, 3),
  minimum_balance DECIMAL(15, 2),
  overdraft_protection BOOLEAN DEFAULT false,
  
  -- Ownership
  primary_owner UUID NOT NULL REFERENCES customers(customer_id),
  secondary_owner UUID REFERENCES customers(customer_id),
  ownership_type ENUM('individual', 'joint', 'trust', 'business'),
  
  -- Status
  status ENUM('active', 'closed', 'suspended', 'dormant') DEFAULT 'active',
  
  -- Dates
  opened_date DATE NOT NULL,
  closed_date DATE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  CONSTRAINT balance_check CHECK (
    balance >= 0 OR account_type = 'credit_card'
  ),
  CONSTRAINT joint_account_check CHECK (
    (ownership_type = 'joint' AND secondary_owner IS NOT NULL) OR
    (ownership_type != 'joint' AND secondary_owner IS NULL)
  )
);

-- Indexes
CREATE INDEX idx_accounts_customer_id ON accounts(customer_id);
CREATE INDEX idx_accounts_account_number ON accounts(account_number);
CREATE INDEX idx_accounts_status ON accounts(status);
CREATE INDEX idx_accounts_opened_date ON accounts(opened_date);
```

---

### 3. Transactions Table

```sql
CREATE TABLE transactions (
  transaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  account_id UUID NOT NULL REFERENCES accounts(account_id),
  
  -- Transaction Details
  transaction_type ENUM(
    'debit',
    'credit',
    'transfer',
    'payment',
    'withdrawal',
    'deposit',
    'fee',
    'interest'
  ) NOT NULL,
  
  amount DECIMAL(15, 2) NOT NULL,
  currency_code CHAR(3) DEFAULT 'USD',
  
  -- Counterparty (if applicable)
  counterparty_account_id UUID REFERENCES accounts(account_id),
  counterparty_name VARCHAR(255),
  counterparty_bank_name VARCHAR(255),
  
  -- Merchant (if applicable)
  merchant_id UUID,
  merchant_name VARCHAR(255),
  merchant_category_code VARCHAR(10),
  merchant_location VARCHAR(255),
  
  -- Status & Timestamps
  status ENUM(
    'pending',
    'completed',
    'failed',
    'reversed',
    'scheduled'
  ) DEFAULT 'pending',
  
  initiated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  processed_at TIMESTAMP,
  settlement_date DATE,
  
  -- Fees
  transaction_fee DECIMAL(10, 2) DEFAULT 0,
  
  -- Reference
  reference_number VARCHAR(50),
  description VARCHAR(255),
  
  -- Metadata
  tags JSONB,
  metadata JSONB,
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_transactions_account_id ON transactions(account_id);
CREATE INDEX idx_transactions_status ON transactions(status);
CREATE INDEX idx_transactions_initiated_at ON transactions(initiated_at);
CREATE INDEX idx_transactions_processed_at ON transactions(processed_at);
CREATE INDEX idx_transactions_amount ON transactions(amount);
CREATE PARTIAL INDEX idx_transactions_pending 
  ON transactions(account_id) WHERE status = 'pending';
```

---

### 4. Audit Logs Table

```sql
CREATE TABLE audit_logs (
  log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- User Information
  user_id UUID REFERENCES users(user_id),
  service_account VARCHAR(100),
  
  -- Action Details
  action ENUM(
    'create',
    'read',
    'update',
    'delete',
    'approve',
    'reject',
    'export',
    'verify'
  ) NOT NULL,
  
  -- Entity Information
  entity_type VARCHAR(100) NOT NULL,
  entity_id VARCHAR(100) NOT NULL,
  
  -- Changes
  old_values JSONB,
  new_values JSONB,
  changes_summary JSONB,
  
  -- Context
  ip_address INET,
  user_agent VARCHAR(500),
  request_id VARCHAR(100),
  
  -- Status
  status ENUM('success', 'failure') DEFAULT 'success',
  error_message TEXT,
  
  -- Timestamp
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Compliance
  compliance_relevant BOOLEAN DEFAULT false,
  retention_until DATE
);

-- Indexes
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_entity_type ON audit_logs(entity_type);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX idx_audit_logs_request_id ON audit_logs(request_id);
```

---

### 5. PII Detection Results Table

```sql
CREATE TABLE pii_detection_results (
  result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Source
  source_type ENUM('transaction', 'chat', 'document', 'api_call'),
  source_id VARCHAR(100),
  
  -- PII Information
  pii_type ENUM(
    'ssn',
    'credit_card',
    'bank_account',
    'phone',
    'email',
    'address',
    'passport',
    'drivers_license',
    'account_number'
  ),
  
  detected_value_hash VARCHAR(256), -- Hash of detected PII (not actual value)
  confidence_score DECIMAL(3, 2),
  
  -- Action Taken
  action_taken ENUM(
    'masked',
    'encrypted',
    'flagged',
    'allowed',
    'rejected'
  ) DEFAULT 'masked',
  
  -- Context
  context_before VARCHAR(500),
  context_after VARCHAR(500),
  
  -- Timestamps
  detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  
  -- Compliance
  audit_log_id UUID REFERENCES audit_logs(log_id)
);

-- Indexes
CREATE INDEX idx_pii_results_source_type ON pii_detection_results(source_type);
CREATE INDEX idx_pii_results_pii_type ON pii_detection_results(pii_type);
CREATE INDEX idx_pii_results_detected_at ON pii_detection_results(detected_at);
```

---

### 6. Chat Sessions Table

```sql
CREATE TABLE chat_sessions (
  session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  customer_id UUID NOT NULL REFERENCES customers(customer_id),
  
  -- Session Information
  session_status ENUM(
    'active',
    'completed',
    'abandoned',
    'flagged'
  ) DEFAULT 'active',
  
  conversation_type ENUM(
    'account_inquiry',
    'transaction_request',
    'complaint',
    'fraud_report',
    'kyc_verification',
    'general_inquiry'
  ),
  
  -- Conversation
  messages_count INTEGER DEFAULT 0,
  message_summary JSONB,
  
  -- ML Model Information
  model_used VARCHAR(100),
  model_version VARCHAR(50),
  confidence_scores DECIMAL(3, 2)[],
  
  -- Timestamps
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  ended_at TIMESTAMP,
  duration_seconds INTEGER,
  
  -- Outcomes
  resolution_type ENUM(
    'resolved',
    'escalated',
    'incomplete',
    'no_action'
  ),
  
  -- Metadata
  device_type VARCHAR(50),
  channel ENUM('web', 'mobile', 'api', 'voice'),
  tags JSONB
);

-- Indexes
CREATE INDEX idx_chat_sessions_customer_id ON chat_sessions(customer_id);
CREATE INDEX idx_chat_sessions_session_status ON chat_sessions(session_status);
CREATE INDEX idx_chat_sessions_started_at ON chat_sessions(started_at);
```

---

### 7. Compliance Events Table

```sql
CREATE TABLE compliance_events (
  event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Event Type
  event_type ENUM(
    'kyc_verification',
    'aml_screening',
    'suspicious_activity',
    'transaction_monitoring',
    'sanctions_hit',
    'pep_match',
    'policy_violation'
  ) NOT NULL,
  
  -- Related Entities
  customer_id UUID REFERENCES customers(customer_id),
  account_id UUID REFERENCES accounts(account_id),
  transaction_id UUID REFERENCES transactions(transaction_id),
  
  -- Event Details
  severity ENUM('low', 'medium', 'high', 'critical') DEFAULT 'medium',
  status ENUM('open', 'under_review', 'resolved', 'escalated'),
  
  description TEXT,
  findings JSONB,
  
  -- Risk Assessment
  risk_score DECIMAL(5, 2),
  recommended_action VARCHAR(255),
  
  -- Review
  reviewed_by UUID REFERENCES users(user_id),
  review_notes TEXT,
  reviewed_at TIMESTAMP,
  
  -- Reporting
  sar_filed BOOLEAN DEFAULT false, -- Suspicious Activity Report
  sar_date DATE,
  ctr_filed BOOLEAN DEFAULT false, -- Currency Transaction Report
  ctr_date DATE,
  
  -- Timestamps
  detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX idx_compliance_events_customer_id ON compliance_events(customer_id);
CREATE INDEX idx_compliance_events_event_type ON compliance_events(event_type);
CREATE INDEX idx_compliance_events_severity ON compliance_events(severity);
CREATE INDEX idx_compliance_events_status ON compliance_events(status);
```

---

### 8. API Keys & Integrations Table

```sql
CREATE TABLE api_keys (
  key_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  
  -- Key Information
  key_hash VARCHAR(256) NOT NULL UNIQUE, -- SHA-256 hash
  key_prefix VARCHAR(20), -- First 20 chars for logs
  
  -- Ownership
  organization_id UUID NOT NULL,
  created_by UUID REFERENCES users(user_id),
  
  -- Permissions
  scopes TEXT[] NOT NULL, -- read:account, write:transaction, etc.
  
  -- Status
  status ENUM('active', 'revoked', 'expired') DEFAULT 'active',
  
  -- Rate Limits
  requests_per_minute INTEGER DEFAULT 100,
  requests_per_hour INTEGER DEFAULT 5000,
  requests_per_day INTEGER DEFAULT 50000,
  
  -- Usage
  total_requests INTEGER DEFAULT 0,
  last_used_at TIMESTAMP,
  
  -- Dates
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  expires_at TIMESTAMP,
  revoked_at TIMESTAMP,
  
  -- Metadata
  name VARCHAR(255),
  tags JSONB
);

-- Indexes
CREATE INDEX idx_api_keys_key_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_organization_id ON api_keys(organization_id);
CREATE INDEX idx_api_keys_status ON api_keys(status);
```

---

## Data Validation Rules

### Amount Validation
```python
class AmountValidator:
    MIN_TRANSACTION: Decimal = Decimal("0.01")
    MAX_TRANSACTION: Decimal = Decimal("999999.99")
    
    @staticmethod
    def validate_amount(amount: Decimal) -> bool:
        return (
            AmountValidator.MIN_TRANSACTION <= amount <= 
            AmountValidator.MAX_TRANSACTION
        )
```

### Email Validation
```python
import re

EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'

def validate_email(email: str) -> bool:
    return re.match(EMAIL_PATTERN, email) is not None
```

### Account Number Validation
```python
def validate_account_number(account_number: str) -> bool:
    # Must be 16-20 characters, alphanumeric
    return (
        len(account_number) >= 16 and 
        len(account_number) <= 20 and 
        account_number.isalnum()
    )
```

---

## Encryption & Security

### Sensitive Fields
- `ssn_encrypted`: AES-256 encryption
- `credit_card_numbers`: Never stored (PCI-DSS compliance)
- `passwords`: bcrypt hashing (never stored plain)

### Encryption Key Rotation
- Keys rotated every 90 days
- Old keys retained for decryption of historical data
- HSM integration for key management

---

## Data Retention Policies

| Table | Retention Period | Deletion Method |
|-------|------------------|-----------------|
| customers | Lifetime | Anonymization |
| accounts | 7 years after closure | Hard delete |
| transactions | 7 years (regulatory) | Archive then delete |
| audit_logs | 10 years (compliance) | Archival storage |
| chat_sessions | 90 days (then archive) | Archive then delete |
| pii_detection_results | 1 year | Hard delete |
| compliance_events | 10 years | Archival storage |

---

## Database Optimization

### Query Performance Considerations
```sql
-- Optimized query for customer transactions
EXPLAIN ANALYZE
SELECT t.transaction_id, t.amount, t.status, t.initiated_at
FROM transactions t
WHERE t.account_id = $1
  AND t.initiated_at > NOW() - INTERVAL '90 days'
  AND t.status = 'completed'
ORDER BY t.initiated_at DESC
LIMIT 50;

-- Should use index: idx_transactions_pending or
-- composite index on (account_id, status, initiated_at)
```

### Connection Pooling
- Min pool size: 5
- Max pool size: 20
- Max overflow: 10
- Pool timeout: 30 seconds

---

## Backup & Disaster Recovery

**Backup Strategy:**
- Daily full backups (S3)
- Hourly incremental backups (S3)
- 30-day retention for daily backups
- 1-year retention for monthly snapshots
- RTO: 4 hours
- RPO: 1 hour

**Backup Encryption:**
- AES-256 encryption at rest
- SSL/TLS in transit
- Separate key for backup encryption

---

## Monitoring & Alerts

### Key Metrics
- Connection pool utilization
- Query execution time (p95, p99)
- Transaction volume
- Error rates
- Data growth rate

### Alerts
```yaml
database_connection_pool_low: # < 2 available connections
  severity: critical
  action: page_on_call

slow_query: # > 5s execution
  severity: high
  action: log_and_monitor

transaction_volume_spike: # > 3x normal
  severity: medium
  action: alert_ops_team
```

---

## Migration Strategy

### Adding a New Column
```sql
-- 1. Add column with default (not nullable)
ALTER TABLE transactions ADD COLUMN source_system VARCHAR(100) DEFAULT 'api';

-- 2. Remove default after backfill
ALTER TABLE transactions ALTER COLUMN source_system DROP DEFAULT;

-- 3. Add index if needed
CREATE INDEX idx_transactions_source ON transactions(source_system);
```

---

## Next Steps

1. Database setup: `scripts/init_db.py`
2. Run migrations: `alembic upgrade head`
3. Configure backups (AWS S3)
4. Set up monitoring
5. Configure audit logging
