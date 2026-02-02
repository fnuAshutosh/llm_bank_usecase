# API Specifications - Banking LLM System

**API Version:** 1.0.0  
**Base URL:** `https://api.banking-llm.com`  
**Environment URLs:**
- Development: `http://localhost:8000`
- Staging: `https://staging-api.banking-llm.com`
- Production: `https://api.banking-llm.com`

---

## Authentication

### API Key Authentication (Development)
```bash
curl -X GET http://localhost:8000/api/v1/chat \
  -H "Authorization: Bearer YOUR_API_KEY"
```

### OAuth 2.0 (Production)
```bash
# 1. Get access token
curl -X POST https://auth.banking-llm.com/oauth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "grant_type=client_credentials&client_id=YOUR_ID&client_secret=YOUR_SECRET"

# 2. Use access token
curl -X GET https://api.banking-llm.com/api/v1/chat \
  -H "Authorization: Bearer ACCESS_TOKEN"
```

### JWT Tokens (Production)
```
Header: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
Expiration: 1 hour (refresh tokens available)
Scopes: read:chat, write:chat, read:account, write:transaction
```

---

## Rate Limiting

### Rate Limit Headers
```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1643723400
```

### Tier-Based Limits
| Tier | Requests/Min | Requests/Hour | Requests/Day |
|------|--------------|---------------|--------------|
| Free | 10 | 500 | 5,000 |
| Professional | 100 | 5,000 | 50,000 |
| Enterprise | 1,000 | 50,000 | 500,000 |
| Custom | Unlimited | Unlimited | Unlimited |

### Rate Limit Response
```json
{
  "error": "rate_limit_exceeded",
  "message": "You have exceeded your rate limit of 1000 requests per minute",
  "retry_after": 60,
  "reset_time": "2026-02-01T17:05:00Z"
}
```

---

## Core Endpoints

### 1. Health Check Endpoints

#### Basic Health Check
```http
GET /health

Response 200:
{
  "status": "healthy",
  "timestamp": "2026-02-01T17:00:00Z"
}
```

#### Detailed Health Check
```http
GET /health/detailed

Response 200:
{
  "status": "degraded",
  "timestamp": "2026-02-01T17:00:00Z",
  "version": "0.1.0",
  "environment": "development",
  "components": {
    "database": {
      "status": "healthy",
      "latency_ms": 5,
      "connection_pool": {
        "active": 10,
        "idle": 5,
        "max": 20
      }
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 2,
      "memory_usage_mb": 128
    },
    "model": {
      "status": "healthy",
      "model_name": "llama2-7b",
      "loaded": true,
      "inference_latency_ms": 234
    },
    "external_apis": {
      "together_ai": {
        "status": "healthy",
        "latency_ms": 150
      }
    }
  }
}
```

---

### 2. Chat Endpoint

#### POST /api/v1/chat
**Purpose:** Send a customer query and receive LLM-powered response  
**Authentication:** Required (API Key or OAuth 2.0)  
**Rate Limit:** 100 requests/minute

**Request:**
```json
{
  "message": "What is my account balance?",
  "customer_id": "CUST_12345",
  "session_id": "SESSION_67890",
  "account_id": "ACC-2024-001",
  "context_type": "account_inquiry",
  "conversation_history": [
    {
      "role": "user",
      "content": "Hello",
      "timestamp": "2026-02-01T16:55:00Z"
    },
    {
      "role": "assistant",
      "content": "Hello! How can I help you today?",
      "timestamp": "2026-02-01T16:55:05Z"
    }
  ],
  "metadata": {
    "device": "mobile",
    "ip_address": "192.0.2.1",
    "user_agent": "MobileApp/1.0"
  }
}
```

**Response (200 OK):**
```json
{
  "status": "success",
  "message": "Your checking account balance is $5,234.67 as of Feb 1, 2026 at 3:30 PM ET.",
  "metadata": {
    "customer_id": "CUST_12345",
    "session_id": "SESSION_67890",
    "context_type": "account_inquiry",
    "timestamp": "2026-02-01T17:00:00Z",
    "processing_time_ms": 245,
    "model_used": "llama2-7b",
    "model_version": "v2.1",
    "confidence_score": 0.92,
    "pii_detected": ["account_number"],
    "pii_masked": true,
    "audit_log_id": "LOG-2024-8847-001"
  },
  "suggestions": [
    "Would you like to view recent transactions?",
    "Need help with a transfer?"
  ],
  "action_items": [
    {
      "type": "transaction_recommendation",
      "description": "Set up autopay for your monthly bill"
    }
  ]
}
```

**Response (400 Bad Request):**
```json
{
  "status": "error",
  "error_code": "INVALID_REQUEST",
  "message": "Missing required field: customer_id",
  "details": {
    "field": "customer_id",
    "reason": "required_field_missing"
  }
}
```

**Response (401 Unauthorized):**
```json
{
  "status": "error",
  "error_code": "UNAUTHORIZED",
  "message": "Invalid or expired API key",
  "details": {
    "token_type": "bearer",
    "hint": "Get a new token from the authentication endpoint"
  }
}
```

**Response (429 Too Many Requests):**
```json
{
  "status": "error",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "message": "Too many requests",
  "retry_after": 60,
  "reset_time": "2026-02-01T17:05:00Z"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "error",
  "error_code": "SERVICE_UNAVAILABLE",
  "message": "The API is temporarily unavailable. Please retry in 60 seconds.",
  "retry_after": 60
}
```

---

### 3. Account Endpoints

#### GET /api/v1/accounts/{account_id}
**Purpose:** Retrieve account details  
**Authentication:** Required  
**Parameters:**
- `account_id` (path): Account identifier
- `include_transactions` (query): Include recent transactions (default: false)

**Response (200):**
```json
{
  "account_id": "ACC-2024-001",
  "account_type": "checking",
  "holder_name": "John Doe",
  "balance": 5234.67,
  "available_balance": 5200.00,
  "currency": "USD",
  "status": "active",
  "opened_date": "2020-01-15",
  "last_activity": "2026-02-01T15:30:00Z",
  "transactions": [
    {
      "id": "TXN-2024-8847-001",
      "date": "2026-02-01T15:30:00Z",
      "description": "Coffee Shop Purchase",
      "amount": -4.50,
      "balance_after": 5234.67,
      "status": "completed"
    }
  ]
}
```

#### POST /api/v1/accounts/{account_id}/transactions
**Purpose:** Initiate a transaction (transfer, payment, etc.)  
**Authentication:** Required + MFA  

**Request:**
```json
{
  "transaction_type": "transfer",
  "amount": 250.00,
  "destination_account": "ACC-2024-002",
  "description": "Payment to friend",
  "scheduled_date": "2026-02-05",
  "mfa_token": "MFA_TOKEN_12345"
}
```

**Response (201 Created):**
```json
{
  "transaction_id": "TXN-2024-8848-001",
  "status": "scheduled",
  "amount": 250.00,
  "from_account": "ACC-2024-001",
  "to_account": "ACC-2024-002",
  "scheduled_date": "2026-02-05T00:00:00Z",
  "created_at": "2026-02-01T17:00:00Z",
  "confirmation_code": "CONF_8847_001"
}
```

---

### 4. Authentication Endpoints

#### POST /auth/login
**Purpose:** Authenticate and get session token  

**Request:**
```json
{
  "username": "user@example.com",
  "password": "secure_password_123",
  "mfa_code": "123456"
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "user": {
    "id": "USER_12345",
    "email": "user@example.com",
    "name": "John Doe",
    "mfa_enabled": true
  }
}
```

#### POST /auth/refresh
**Purpose:** Refresh access token  

**Request:**
```json
{
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Response (200):**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600
}
```

---

### 5. Admin Endpoints

#### GET /admin/models
**Purpose:** List available LLM models  
**Authentication:** Required (Admin role)  

**Response (200):**
```json
{
  "models": [
    {
      "id": "llama2-7b",
      "name": "Llama 2 7B",
      "provider": "meta",
      "status": "active",
      "inference_latency_ms": 234,
      "daily_requests": 125000,
      "accuracy_score": 0.92
    },
    {
      "id": "mistral-7b",
      "name": "Mistral 7B",
      "provider": "mistralai",
      "status": "active",
      "inference_latency_ms": 189,
      "daily_requests": 98000,
      "accuracy_score": 0.94
    }
  ]
}
```

#### POST /admin/models/{model_id}/switch
**Purpose:** Switch active model  
**Authentication:** Required (Admin role)  

**Response (200):**
```json
{
  "status": "success",
  "message": "Model switched to mistral-7b",
  "previous_model": "llama2-7b",
  "current_model": "mistral-7b",
  "switched_at": "2026-02-01T17:00:00Z"
}
```

#### GET /admin/stats
**Purpose:** Get system statistics  
**Authentication:** Required (Admin role)  

**Response (200):**
```json
{
  "timestamp": "2026-02-01T17:00:00Z",
  "uptime_hours": 720,
  "requests": {
    "total": 15234892,
    "successful": 15198234,
    "failed": 36658,
    "success_rate": 0.9976
  },
  "performance": {
    "avg_latency_ms": 234,
    "p50_latency_ms": 189,
    "p95_latency_ms": 523,
    "p99_latency_ms": 1023
  },
  "errors": {
    "rate_limit": 12,
    "auth_failed": 234,
    "service_error": 36,
    "invalid_request": 376
  },
  "pii": {
    "detected": 1234,
    "masked": 1234,
    "missed": 0
  }
}
```

---

### 6. Compliance Endpoints

#### POST /api/v1/compliance/kyc
**Purpose:** Submit KYC verification  
**Authentication:** Required  

**Request:**
```json
{
  "customer_id": "CUST_12345",
  "document_type": "passport",
  "document_url": "s3://documents/passport_123.pdf",
  "address_proof_url": "s3://documents/address_123.pdf"
}
```

**Response (202 Accepted):**
```json
{
  "status": "pending",
  "verification_id": "KYC_VER_12345",
  "message": "KYC verification submitted and queued for processing",
  "estimated_processing_time_hours": 2,
  "webhook_callback": "https://yourapp.com/webhook/kyc",
  "status_check_url": "/api/v1/compliance/kyc/KYC_VER_12345"
}
```

#### GET /api/v1/compliance/kyc/{verification_id}
**Purpose:** Check KYC verification status  
**Authentication:** Required  

**Response (200):**
```json
{
  "verification_id": "KYC_VER_12345",
  "status": "approved",
  "customer_id": "CUST_12345",
  "completed_at": "2026-02-01T16:30:00Z",
  "document_verification": {
    "status": "approved",
    "document_type": "passport",
    "validity": "valid",
    "expiry_date": "2027-06-15",
    "name_match": true
  },
  "address_verification": {
    "status": "approved",
    "address": "123 Main St, Boston, MA 02101",
    "verification_method": "utility_bill"
  },
  "aml_screening": {
    "status": "clear",
    "watchlists_checked": ["OFAC_SDN", "EU_CONSOLIDATED", "PEP"],
    "matches": 0
  }
}
```

---

### 7. Fraud Detection Endpoints

#### POST /api/v1/fraud/report
**Purpose:** Report a fraudulent transaction  
**Authentication:** Required  

**Request:**
```json
{
  "transaction_id": "TXN-2024-8847-001",
  "reason": "unauthorized_charge",
  "description": "I didn't make this $500 purchase"
}
```

**Response (201 Created):**
```json
{
  "case_id": "FRAUD_CASE_12345",
  "status": "open",
  "transaction_id": "TXN-2024-8847-001",
  "amount": 500.00,
  "reported_at": "2026-02-01T17:00:00Z",
  "provisional_credit": {
    "amount": 500.00,
    "applied_at": "2026-02-01T17:00:00Z"
  },
  "estimated_resolution": "2026-02-15T00:00:00Z"
}
```

---

## Data Types & Schemas

### Transaction Object
```json
{
  "id": "TXN-2024-8847-001",
  "date": "2026-02-01T15:30:00Z",
  "type": "purchase|transfer|payment|withdrawal",
  "amount": 250.00,
  "currency": "USD",
  "description": "Coffee Shop Purchase",
  "merchant": {
    "name": "Starbucks",
    "category": "food_beverage",
    "location": "Boston, MA"
  },
  "status": "completed|pending|failed",
  "balance_after": 5234.67,
  "fees": 0.00
}
```

### Error Object
```json
{
  "status": "error",
  "error_code": "ERROR_CODE",
  "message": "Human readable error message",
  "details": {
    "field": "field_name",
    "reason": "reason_code"
  },
  "request_id": "REQ_12345_6789"
}
```

---

## Pagination

### Query Parameters
```
?page=1&limit=20&sort=date&order=desc
```

### Response Headers
```
X-Total-Count: 1000
X-Page: 1
X-Limit: 20
X-Total-Pages: 50
```

### Response Body
```json
{
  "data": [...],
  "pagination": {
    "current_page": 1,
    "per_page": 20,
    "total_items": 1000,
    "total_pages": 50,
    "has_next": true,
    "has_prev": false
  }
}
```

---

## Webhooks

### Event Types
- `transaction.completed`
- `transaction.failed`
- `account.updated`
- `kyc.verified`
- `fraud.detected`
- `dispute.resolved`

### Webhook Payload
```json
{
  "event_id": "EVT_12345",
  "event_type": "transaction.completed",
  "timestamp": "2026-02-01T17:00:00Z",
  "data": {
    "transaction_id": "TXN-2024-8847-001",
    "amount": 250.00,
    "status": "completed"
  }
}
```

### Webhook Security
```
Header: X-Webhook-Signature
Value: sha256=abcd1234...
```

---

## Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| INVALID_REQUEST | 400 | Missing or invalid request parameters |
| UNAUTHORIZED | 401 | Authentication failed |
| FORBIDDEN | 403 | Insufficient permissions |
| NOT_FOUND | 404 | Resource not found |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| INTERNAL_ERROR | 500 | Server error |
| SERVICE_UNAVAILABLE | 503 | Service temporarily down |

---

## API Clients

### Python SDK
```python
from banking_llm import BankingLLMClient

client = BankingLLMClient(api_key="YOUR_API_KEY")

response = client.chat.send(
    message="What is my balance?",
    customer_id="CUST_12345"
)

print(response.message)
```

### JavaScript/Node.js SDK
```javascript
const BankingLLM = require('banking-llm');

const client = new BankingLLM({
  apiKey: 'YOUR_API_KEY'
});

const response = await client.chat.send({
  message: 'What is my balance?',
  customerId: 'CUST_12345'
});

console.log(response.message);
```

---

## Testing

### Using cURL
```bash
# Health check
curl -X GET http://localhost:8000/health

# Chat endpoint
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "message": "What is my balance?",
    "customer_id": "CUST_12345"
  }'
```

### Using Postman
1. Import the OpenAPI spec: `/api/v1/openapi.json`
2. Set authorization: `Bearer YOUR_API_KEY`
3. Create and run requests

### Using Swagger UI
Visit: `http://localhost:8000/docs` (development only)

---

## API Versioning

Current version: **1.0.0**

### Version Strategy
- `/api/v1/` - Current version
- `/api/v2/` - Future breaking changes

Deprecated versions will have 12-month sunset period.

---

## SLA & Support

**Uptime SLA:** 99.95% (excluding planned maintenance)  
**Response Time:** p95 < 500ms for 95% of requests  
**Support:** 24/7 for enterprise customers

**Support Channels:**
- Email: support@banking-llm.com
- Slack: #api-support
- Phone: +1-800-BANK-LLM
