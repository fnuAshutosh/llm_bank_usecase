# Banking Use Cases - Enterprise LLM System

## Overview

This document details all banking scenarios and use cases supported by the enterprise Banking LLM system. Each use case includes real-world examples, compliance requirements, and API integration patterns.

---

## 1. Customer Service & Account Inquiries

### 1.1 Account Balance & Transaction History
**Purpose:** Allow customers to check balances and recent transactions via natural language  
**Compliance:** PCI-DSS 3.2.1 (customer data access control)

```
Customer: "What's my checking account balance?"

LLM Context:
- Account ID: ACC-2024-001
- Account Type: Checking
- Last Updated: 2026-02-01 15:30 UTC

Response: "Your checking account balance is $5,234.67 as of Feb 1, 2026 at 3:30 PM ET."
```

**API Integration:**
```bash
POST /api/v1/chat
{
  "message": "What's my account balance?",
  "customer_id": "CUST_12345",
  "account_id": "ACC-2024-001",
  "context_type": "account_inquiry"
}
```

**Security Controls:**
- ✅ PII masking on account numbers (show last 4 digits only)
- ✅ Multi-factor authentication required
- ✅ Audit logging of all inquiries
- ✅ Rate limiting (10 requests/minute per customer)

---

### 1.2 Fraud Detection & Alert
**Purpose:** Identify suspicious transactions and alert customers  
**Compliance:** PCI-DSS 3.4 (fraud detection), GLBA (customer protection)

```
System: "Alert: Unusual transaction detected"
- $3,500 purchase at electronics retailer
- Location: China
- Time: 3:00 AM (unusual for this customer)
- Similarity: 0% match with typical purchases

Question: "Is this transaction authorized?"
```

**Risk Scoring:**
- Geographic anomaly: +40 points (typical: 5000km away)
- Time anomaly: +20 points (unusual hour)
- Amount anomaly: +30 points (10x typical transaction)
- Merchant anomaly: +10 points (new category)

**Total Risk Score: 100/100 → ALERT**

**API Integration:**
```bash
POST /api/v1/chat
{
  "message": "I need to report a fraudulent transaction",
  "transaction_id": "TXN-2024-8847-001",
  "severity": "high",
  "context_type": "fraud_report"
}
```

---

### 1.3 Bill Payment & Transfer Assistance
**Purpose:** Help customers make payments and transfers via conversational interface  
**Compliance:** ACH rules, wire transfer regulations

```
Customer: "I want to pay my credit card bill of $1,200"

LLM Response:
1. Confirms payment amount and due date
2. Offers payment options (immediate, scheduled)
3. Confirms account to debit from
4. Shows confirmation code for audit trail

Confirmation: "Payment of $1,200 scheduled for Feb 5, 2026"
```

**Supported Transactions:**
- P2P transfers (domestic)
- Bill payments
- ACH transfers
- Wire transfers (with authorization)
- Loan payments

---

## 2. Know Your Customer (KYC) & Anti-Money Laundering (AML)

### 2.1 Customer Onboarding & Verification
**Purpose:** Streamline KYC verification using LLM-assisted documentation review  
**Compliance:** PCI-DSS, GDPR, FinCEN regulations

```
Agent: "Please provide your government-issued ID and proof of address"

Customer: "I have my passport and recent utility bill"

LLM Verification:
- Passport scan analysis:
  * Valid: YES
  * Expiry: 2027-06-15 (valid)
  * Name match: YES
  * DOB: 1985-03-20
  
- Utility bill analysis:
  * Address: 123 Main St, Boston, MA 02101
  * Date: 2026-01-15
  * Ownership: YES (matches passport name)

Status: APPROVED ✅
KYC Risk: LOW
Processing Time: 2 minutes (vs. 2 hours manual)
```

**Risk Assessment Factors:**
- Document validity
- Age verification
- Address verification
- PEP/Sanctions database check
- Industry risk level
- Geographic risk level

**API Integration:**
```bash
POST /api/v1/chat
{
  "message": "Complete KYC verification",
  "customer_id": "NEW_CUST_2024",
  "documents": [
    {
      "type": "passport",
      "url": "s3://documents/passport_123.pdf"
    },
    {
      "type": "utility_bill",
      "url": "s3://documents/address_proof_123.pdf"
    }
  ],
  "context_type": "kyc_verification"
}
```

---

### 2.2 AML Screening & Sanctions Check
**Purpose:** Screen transactions against OFAC, PEP, and consolidated watchlists  
**Compliance:** AML Act, OFAC regulations

```
Transaction Screening:

Sender: "John Smith" from Iran
Match Score: 89% (compared to OFAC SDN list)

LLM Recommendation: 
"REJECT transaction - High probability match with OFAC Specially Designated 
Nationals list. Manual review required by compliance team."

Actions:
- Block transaction
- Flag for compliance review
- Generate SAR (Suspicious Activity Report)
- Notify management
```

**Watchlists Monitored:**
- OFAC SDN (Specially Designated Nationals)
- OFAC Consolidated Non-SDN List
- EU Consolidated List
- UK Consolidated List
- UN Sanctions Lists
- PEP (Politically Exposed Persons) databases

---

### 2.3 Customer Risk Profiling
**Purpose:** Continuously assess and update customer risk profiles  
**Compliance:** Risk-based approach (AML Act)

```
Customer Risk Profile:

Name: Jane Doe
Account Age: 3 years
Risk Level: MEDIUM

Factors:
- Transaction Volume: $50K-$100K monthly ✅ Normal
- Geographic Pattern: 5 countries/month ⚠️ High
- Sudden Account Activity: Recent spike ⚠️ Elevated
- Industry: Technology Consulting ✅ Low Risk
- Transaction Types: Diverse (good) ✅
- Age of Accounts: 15 years average ✅ Established

Risk Score: 65/100 (MEDIUM)
Recommendation: Monitor for escalations, no action needed
Review Date: 2026-05-01
```

---

## 3. Loan & Credit Products

### 3.1 Loan Application & Approval
**Purpose:** Streamline loan applications with LLM-assisted underwriting  
**Compliance:** Fair Lending Act, ECOA, FCRA

```
Customer: "I'd like to apply for a $50,000 personal loan"

LLM Process:
1. Collects application info (conversationally)
   - Loan amount: $50,000
   - Purpose: Home renovation
   - Desired term: 60 months
   - Employment: Software Engineer at TechCorp

2. Analyzes eligibility:
   - Credit score: 750 (excellent)
   - DTI ratio: 28% (good)
   - Employment history: 8 years (good)
   - Income verification: Recent tax returns confirmed

3. Calculates terms:
   - Interest rate: 5.2%
   - Monthly payment: $943
   - Total interest: $6,580
   - APR: 5.2%

4. Generates formal application

Decision: APPROVED ✅
Amount: $50,000
Term: 60 months
Rate: 5.2% APR
```

**Eligibility Criteria:**
- Minimum credit score: 620
- Maximum DTI ratio: 43%
- Minimum annual income: $30,000
- Employment verification: Required
- Bank statements: 2-3 months

---

### 3.2 Credit Card & Line of Credit
**Purpose:** Assist customers in selecting appropriate credit products  
**Compliance:** TILA (Truth in Lending Act), EFTA

```
Customer: "I'm interested in a credit card with good rewards"

LLM Recommendation:
Based on your profile:
- Annual income: $150,000
- Credit score: 780
- Preferred categories: Travel & dining

Recommended: Premium Travel Rewards Card
- Annual Fee: $150
- Sign-up Bonus: $500 (3x on $5,000 spend)
- Benefits:
  * 3x points on travel
  * 2x points on dining
  * 1x points on other purchases
  * Priority airport lounge access
  * Travel insurance included

Credit Limit: $25,000
Approval Decision: INSTANT APPROVAL ✅
```

---

### 3.3 Mortgage Assistance
**Purpose:** Guide customers through mortgage processes  
**Compliance:** RESPA, TRID, HMDA

```
Customer: "I'm looking to buy a home for $500,000"

LLM Process:
1. Qualifies customer
   - Price: $500,000
   - Down payment: 20% ($100,000)
   - Loan amount: $400,000
   - Credit score: 760

2. Calculates scenarios
   - 30-year fixed at 6.5%: $2,528/month
   - 15-year fixed at 6.0%: $4,019/month
   - 7/1 ARM at 6.0%: $2,398/month

3. Estimates costs
   - Down payment: $100,000
   - Closing costs: $12,000 (3%)
   - Property tax/year: $8,000
   - Insurance/year: $1,500

4. Connects to mortgage specialist

Next Step: Document collection
```

---

## 4. Investment & Wealth Management

### 4.1 Investment Portfolio Review
**Purpose:** Analyze and provide recommendations on investment portfolios  
**Compliance:** Reg D, suitability requirements

```
Customer: "How is my investment portfolio performing?"

LLM Analysis:
Portfolio Value: $250,000
Performance (YTD): +12.5% ($30,000 gain)
Allocation:
- US Stocks: 60% ($150,000) → +18%
- International: 20% ($50,000) → +5%
- Bonds: 15% ($37,500) → +2%
- Cash: 5% ($12,500) → +0.5%

Risk Profile: Moderate
Age: 45 years
Years to Retirement: 20

Recommendation:
Current allocation is good for moderate risk. Consider:
- Increase bonds to 20% (reduce volatility)
- Diversify international (add emerging markets)
- Review quarterly

Next Review: 2026-05-01
```

---

### 4.2 Retirement Planning
**Purpose:** Help customers plan for retirement  
**Compliance:** ERISA, retirement plan regulations

```
Customer: "Will I have enough for retirement?"

LLM Analysis:
Current Age: 45
Retirement Age: 65 (target)
Years to Retirement: 20

Current Assets:
- Retirement accounts: $450,000
- Investment accounts: $250,000
- Real estate equity: $800,000
- Total: $1,500,000

Monthly Income: $12,000
Monthly Expenses: $7,000
Annual Savings: $60,000

Projection (age 65):
- Retirement savings: ~$2.8M (with 7% annual growth)
- Expected annual spend: $84,000
- Projected lifespan: 95 years

Analysis: On track ✅
Retirement surplus: $1.2M
Safety margin: 14.5 years of expenses

Recommendations:
- Maintain current savings rate
- Consider Roth conversion
- Review beneficiary designations
```

---

## 5. Compliance & Reporting

### 5.1 Transaction Monitoring
**Purpose:** Real-time monitoring and reporting of suspicious activities  
**Compliance:** AML/CFT regulations, FinCEN reporting

```
Transaction Alert:

Customer: Robert Johnson
Amount: $9,950
Type: Wire transfer to international bank (Samoa)
Frequency: First transfer to this destination
Structured: Amount just below $10,000 threshold

LLM Assessment:
Pattern: Potential structuring (splitting amounts to avoid $10K reporting)
Risk: HIGH
Action: BLOCK transaction + Flag for compliance

Generated Report:
- Transaction flagged for suspicious pattern matching
- Potential structuring indicator
- Manual compliance review required
- Management notification sent
```

---

### 5.2 Regulatory Reporting & Audits
**Purpose:** Generate reports for regulatory requirements  
**Compliance:** FDIC reporting, Federal Reserve reports

```
LLM-Generated Report: Quarterly Compliance Summary

Quarter: Q1 2026
Period: Jan 1 - Mar 31, 2026

Key Metrics:
- Total customers: 125,000 (+5% vs Q4)
- Total transactions: 8.2M
- Average account balance: $45,200
- Fraud cases: 127 (0.0015% rate)
- KYC rejections: 8
- AML alerts: 342 (reviewed)
- SARs filed: 12
- CTRs filed: 234

Compliance Status: FULLY COMPLIANT ✅
- No regulatory violations
- All reports filed on time
- Training completion: 100%
- Audit findings: 0

Next Audit: Q2 2026
```

---

## 6. Dispute Resolution & Chargebacks

### 6.1 Chargeback Investigation
**Purpose:** Handle disputes and chargebacks efficiently  
**Compliance:** Visa/Mastercard dispute rules, EFTA

```
Customer: "This $500 transaction was unauthorized"

LLM Investigation:
Transaction Details:
- Date: 2026-01-28, 02:15 AM
- Amount: $500
- Merchant: Unknown Electronics
- Location: Nigeria (customer typically in US)

Context Analysis:
- Customer location (Feb 1): Boston, MA
- Device: Different phone (iPhone vs usual Android)
- Billing address: Boston, MA
- Shipping address: Lagos, Nigeria

LLM Recommendation: LIKELY FRAUDULENT
Evidence:
- Geographic impossibility (28 hours between transactions)
- Device change
- Unusual merchant
- Suspicious destination

Action: 
- Initiate chargeback process
- Cancel card and reissue
- Flag account for enhanced monitoring
- Refund: $500 (provisional)
```

---

### 6.2 Billing Dispute Resolution
**Purpose:** Resolve billing disputes quickly  
**Compliance:** FCRA, ECOA

```
Customer: "I was charged twice for my mortgage payment"

LLM Analysis:
Transactions Found:
- Feb 1, 2026: $2,500 (mortgage payment)
- Feb 1, 2026: $2,500 (duplicate)

Investigation:
- Both on same date (Feb 1)
- Both from checking account ending in 4532
- Both processed through ACH
- Second payment appears to be system error

LLM Decision: DUPLICATE CHARGE
Action:
- Immediately reverse duplicate payment ($2,500)
- Credit customer account
- File duplicate transaction report
- Notify servicing team

Status: RESOLVED ✅
Timeline: 5 minutes (vs typical 30-60 days)
Refund issued: $2,500
Reference: DISP-2024-001
```

---

## 7. Customer Support & Self-Service

### 7.1 FAQs & General Inquiries
**Purpose:** Answer common banking questions 24/7  

```
Customer: "What are the wire transfer fees?"

LLM Response:
Wire Transfer Fees:

Domestic Wires:
- Standard (3-5 business days): FREE
- Expedited (next business day): $15
- Overnight: $25

International Wires:
- Standard: $35 + currency exchange spread
- Expedited: $50 + currency exchange spread
- Canada/Mexico: $25

Limits:
- Daily: $10,000 (standard customers)
- Monthly: $50,000
- Can increase with verification

Would you like to initiate a wire transfer?
```

---

### 7.2 Account Management
**Purpose:** Help customers update account information  
**Compliance:** GLBA, data security requirements

```
Customer: "Update my phone number"

LLM Process:
1. Verify identity
   - Last 4 SSN: ****1234 ✅
   - Mother's maiden name: ✅
   
2. Confirm change
   - Current: (617) 555-0123
   - New: (617) 555-0456
   
3. Confirm authentication
   - Send verification code to new number
   - Code entered: 742518 ✅
   
4. Confirm update
   - Phone updated successfully
   - All communications to new number
   - Previous number removed

Status: COMPLETE ✅
Audit log: Updated on 2026-02-01 16:45 UTC
```

---

## API Response Examples

### Standard Chat Response
```json
{
  "status": "success",
  "message": "Your account balance is $5,234.67",
  "metadata": {
    "customer_id": "CUST_12345",
    "context_type": "account_inquiry",
    "timestamp": "2026-02-01T17:00:00Z",
    "processing_time_ms": 245,
    "pii_detected": ["account_number"],
    "pii_masked": true,
    "audit_log_id": "LOG-2024-8847-001"
  },
  "suggestions": [
    "Would you like to view recent transactions?",
    "Need help with a transfer?"
  ]
}
```

### Error Response
```json
{
  "status": "error",
  "error_code": "SUSPICIOUS_ACTIVITY",
  "message": "This transaction requires additional verification",
  "required_action": "contact_support",
  "timestamp": "2026-02-01T17:00:00Z"
}
```

---

## Compliance Matrix

| Use Case | PCI-DSS | GDPR | AML/CFT | Fair Lending | GLBA | SOC2 |
|----------|---------|------|---------|--------------|------|------|
| Account Inquiries | ✅ | ✅ | ✅ | - | ✅ | ✅ |
| Fraud Detection | ✅ | ✅ | ✅ | - | ✅ | ✅ |
| KYC Verification | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| AML Screening | ✅ | ✅ | ✅ | - | ✅ | ✅ |
| Loan Application | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Investment Advice | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Chargeback Handling | ✅ | ✅ | ✅ | - | ✅ | ✅ |

---

## Next Steps

1. **Phase 1:** Implement account inquiry and fraud detection (Weeks 1-4)
2. **Phase 2:** Add KYC/AML screening capabilities (Weeks 5-8)
3. **Phase 3:** Integrate loan and credit products (Weeks 9-12)
4. **Phase 4:** Add investment and wealth management (Weeks 13+)

See [Implementation Roadmap](10-IMPLEMENTATION-ROADMAP.md) for detailed timeline.
