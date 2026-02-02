# Infrastructure Setup - Banking LLM System

**Last Updated:** February 1, 2026  
**Environment Types:** Development | Staging | Production  
**Cloud Providers:** AWS (primary), Google Cloud (optional), Azure (optional)

---

## Quick Start Options

### Option 1: GitHub Codespaces (Recommended - Free)
✅ **Best for:** Learning, development, rapid prototyping  
⏱️ **Setup time:** 5 minutes  
💰 **Cost:** Free (with GitHub Pro - 180 hours/month)  
🖥️ **Specs:** 4-core vCPU, 16GB RAM, 15GB storage  

```bash
1. Go to: https://github.com/YOUR_USERNAME/banking-llm
2. Click: Code → Codespaces → Create codespace on main
3. Wait 2-3 minutes for environment
4. In terminal: uvicorn src.api.main:app --reload --port 8000
5. Click notification: "Port 8000 is now listening"
6. Visit: /docs (Swagger UI)
```

### Option 2: Local Development (macOS/Linux)
✅ **Best for:** Full control, custom setup  
⏱️ **Setup time:** 30 minutes  
💰 **Cost:** Free (hardware dependent)  
🖥️ **Specs:** Requires 8GB+ RAM, 10GB+ disk, Python 3.11+  

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/banking-llm.git
cd banking-llm

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements/dev.txt

# Setup database
docker-compose -f .devcontainer/docker-compose.yml up -d

# Run migrations
python scripts/init_db.py

# Start server
uvicorn src.api.main:app --reload --port 8000
```

### Option 3: Docker (All platforms)
✅ **Best for:** Consistent environments, deployment  
⏱️ **Setup time:** 15 minutes  
💰 **Cost:** Free (Docker Desktop free for personal use)  
🖥️ **Specs:** Requires Docker, 4GB+ RAM allocated  

```bash
# Build image
docker build -t banking-llm:latest .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://user:pass@db:5432/banking \
  -e REDIS_URL=redis://cache:6379 \
  banking-llm:latest

# Visit: http://localhost:8000/docs
```

### Option 4: Cloud Deployment (AWS)
✅ **Best for:** Production, scaling  
⏱️ **Setup time:** 45 minutes  
💰 **Cost:** $50-500/month (varies by scale)  
🖥️ **Specs:** Managed, auto-scaling, 99.95% SLA  

```bash
# See Section 3: AWS Production Deployment
```

---

## 1. Development Environment

### 1.1 Prerequisites

**macOS:**
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Install Docker
brew install docker

# Install git
brew install git

# Verify
python3 --version  # Python 3.11.x
docker --version   # Docker version 24.x+
git --version      # git version 2.x+
```

**Linux (Ubuntu/Debian):**
```bash
# Update packages
sudo apt update && sudo apt upgrade

# Install Python 3.11
sudo apt install -y python3.11 python3.11-venv python3-pip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install git
sudo apt install -y git

# Add user to docker group
sudo usermod -aG docker $USER
```

**Windows:**
```
1. Download Python 3.11 from python.org
2. Install with "Add Python to PATH" checked
3. Download Docker Desktop from docker.com
4. Download git from git-scm.com
5. Restart computer
```

### 1.2 Local Setup (Step-by-Step)

**Step 1: Clone Repository**
```bash
cd ~/Projects
git clone https://github.com/YOUR_USERNAME/banking-llm.git
cd banking-llm
```

**Step 2: Create Virtual Environment**
```bash
# macOS/Linux
python3.11 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

**Step 3: Install Dependencies**
```bash
pip install --upgrade pip
pip install -r requirements/dev.txt

# Verify
python -c "import fastapi, torch, transformers; print('✓ All dependencies installed')"
```

**Step 4: Setup Database**
```bash
# Start PostgreSQL + Redis via Docker
docker-compose -f .devcontainer/docker-compose.yml up -d

# Initialize database
python scripts/init_db.py

# Verify
psql -U postgres -d banking_llm -c "SELECT version();"
```

**Step 5: Configure Environment**
```bash
# Copy environment template
cp .env.example .env

# Edit for local development
# (.env created with safe defaults)
```

**Step 6: Start API Server**
```bash
uvicorn src.api.main:app --reload --port 8000
```

**Step 7: Verify Installation**
```bash
# In another terminal
curl http://localhost:8000/health
# Should return: {"status": "healthy", "timestamp": "..."}
```

### 1.3 Development Tools

**Pre-commit Hooks:**
```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Now on every commit:
# - Code formatting (Black)
# - Linting (Flake8, Pylint)
# - Type checking (Mypy)
# - Security scanning (Bandit)
```

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.12.0
    hooks:
      - id: black

  - repo: https://github.com/PyCQA/flake8
    rev: 6.1.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.7.1
    hooks:
      - id: mypy
        args: [--ignore-missing-imports]

  - repo: https://github.com/PyCQA/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
```

**IDE Setup (VS Code):**
```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    }
  }
}
```

---

## 2. Staging Environment

### 2.1 Staging Configuration

**Purpose:**
- Test releases before production
- User acceptance testing (UAT)
- Load testing
- Security testing

**Specs:**
```
Environment: Staging
Compute: 2x t3.large (AWS)
Memory: 16GB total
Storage: 200GB SSD
Database: PostgreSQL 15 (managed)
Cache: Redis 7.2 (managed)
Load Balancer: Application Load Balancer
DNS: staging-api.banking-llm.com
SSL: AWS Certificate Manager (free)
SLA: 95% uptime (no guarantee)
```

**Staging Deployment:**
```bash
# Build and push image
docker build -t banking-llm:staging-1.0.0 .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789.dkr.ecr.us-east-1.amazonaws.com
docker tag banking-llm:staging-1.0.0 123456789.dkr.ecr.us-east-1.amazonaws.com/banking-llm:staging-1.0.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/banking-llm:staging-1.0.0

# Deploy to ECS
aws ecs update-service \
  --cluster banking-llm-staging \
  --service banking-llm-api \
  --force-new-deployment \
  --region us-east-1

# Verify deployment
curl https://staging-api.banking-llm.com/health
```

---

## 3. Production Environment (AWS)

### 3.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Production (Multi-AZ)                │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Route 53 (DNS + Health Checks)                       │  │
│  │ - api.banking-llm.com → ALB                          │  │
│  │ - Failover to secondary region (optional)            │  │
│  └───────────────┬──────────────────────────────────────┘  │
│                  │                                            │
│  ┌──────────────▼──────────────────────────────────────┐  │
│  │ CloudFront (CDN) + WAF (Web Application Firewall)   │  │
│  │ - DDoS protection                                    │  │
│  │ - Rate limiting                                      │  │
│  │ - Bot detection                                      │  │
│  └───────────────┬──────────────────────────────────────┘  │
│                  │                                            │
│  ┌──────────────▼──────────────────────────────────────┐  │
│  │ Application Load Balancer (ALB)                      │  │
│  │ - HTTPS (TLS 1.3)                                    │  │
│  │ - Path-based routing                                 │  │
│  │ - Multi-AZ (us-east-1a, us-east-1b, us-east-1c)    │  │
│  └───────────────┬──────────────────────────────────────┘  │
│                  │                                            │
│     ┌────────────┼────────────┐                              │
│     │            │            │                              │
│  ┌──▼──┐  ┌──────▼──┐  ┌──────▼──┐                           │
│  │ ECS │  │  ECS   │  │  ECS   │                           │
│  │ AZ1 │  │  AZ2   │  │  AZ3   │  (Auto-scaling: 2-10)    │
│  └──┬──┘  └────┬───┘  └────┬───┘                           │
│     │         │           │                                │
│     └─────────┼───────────┘                                │
│               │                                             │
│  ┌────────────▼────────────────────────────────────────┐  │
│  │ RDS PostgreSQL (Multi-AZ)                           │  │
│  │ - Primary: us-east-1a                               │  │
│  │ - Standby: us-east-1b                               │  │
│  │ - Automated backups (35 days retention)              │  │
│  │ - Multi-AZ failover < 1 minute                       │  │
│  └────────────┬───────────────────────────────────────┘  │
│               │                                             │
│  ┌────────────▼────────────────────────────────────────┐  │
│  │ ElastiCache Redis (Multi-AZ)                         │  │
│  │ - Session caching                                    │  │
│  │ - Rate limit counters                                │  │
│  │ - Model inference cache                              │  │
│  │ - Automatic failover                                 │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Supporting Services                                  │  │
│  │ - S3 (backups, documents, logs)                      │  │
│  │ - CloudWatch (logging, monitoring)                   │  │
│  │ - EventBridge (event-driven workflows)               │  │
│  │ - SNS/SQS (messaging)                                │  │
│  │ - Secrets Manager (credential management)           │  │
│  │ - KMS (key management)                               │  │
│  │ - VPC (network isolation)                            │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Infrastructure as Code (Terraform)

**main.tf:**
```hcl
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  # Backend for state management
  backend "s3" {
    bucket         = "banking-llm-terraform-state"
    key            = "prod/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

provider "aws" {
  region = "us-east-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "banking-llm-vpc"
  }
}

# Public Subnets (3 AZs)
resource "aws_subnet" "public" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "banking-llm-public-${count.index + 1}"
  }
}

# Private Subnets (3 AZs)
resource "aws_subnet" "private" {
  count             = 3
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 10}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  tags = {
    Name = "banking-llm-private-${count.index + 1}"
  }
}

# RDS PostgreSQL
resource "aws_db_instance" "main" {
  identifier     = "banking-llm-prod"
  engine         = "postgres"
  engine_version = "15.5"
  instance_class = "db.r6g.2xlarge"
  
  allocated_storage    = 500
  storage_type         = "io1"
  iops                 = 5000
  storage_encrypted    = true
  kms_key_id           = aws_kms_key.main.arn
  
  multi_az               = true
  publicly_accessible    = false
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 35
  backup_window           = "03:00-04:00"
  maintenance_window      = "mon:04:00-mon:05:00"
  
  skip_final_snapshot = false
  final_snapshot_identifier = "banking-llm-final-snapshot-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  tags = {
    Name = "banking-llm-prod-db"
  }
}

# ElastiCache Redis
resource "aws_elasticache_cluster" "main" {
  cluster_id           = "banking-llm-redis"
  engine               = "redis"
  node_type            = "cache.r6g.xlarge"
  num_cache_nodes      = 3
  parameter_group_name = "default.redis7"
  
  availability_zones = slice(
    data.aws_availability_zones.available.names,
    0, 3
  )
  
  engine_version = "7.2"
  port           = 6379
  
  security_group_ids = [aws_security_group.redis.id]
  subnet_group_name  = aws_elasticache_subnet_group.main.name
  
  automatic_failover_enabled = true
  multi_az_enabled           = true
  
  tags = {
    Name = "banking-llm-prod-redis"
  }
}

# (... more resources ...)
```

### 3.3 Deployment Process

**Production Deployment Steps:**

```bash
# 1. Build and tag Docker image
VERSION=$(git describe --tags --always)
docker build -t banking-llm:${VERSION} .
docker tag banking-llm:${VERSION} ${ECR_REGISTRY}/banking-llm:${VERSION}
docker tag banking-llm:${VERSION} ${ECR_REGISTRY}/banking-llm:latest

# 2. Push to ECR
aws ecr get-login-password | docker login --username AWS --password-stdin ${ECR_REGISTRY}
docker push ${ECR_REGISTRY}/banking-llm:${VERSION}

# 3. Run automated tests
pytest tests/ -v --cov=src

# 4. Update infrastructure (if needed)
terraform plan -out=tfplan
terraform apply tfplan

# 5. Deploy to ECS
aws ecs update-service \
  --cluster banking-llm-prod \
  --service banking-llm-api \
  --force-new-deployment \
  --region us-east-1

# 6. Wait for deployment
aws ecs wait services-stable \
  --cluster banking-llm-prod \
  --services banking-llm-api \
  --region us-east-1

# 7. Smoke tests
curl https://api.banking-llm.com/health/detailed

# 8. Tag release
git tag -a v${VERSION} -m "Release ${VERSION}"
git push origin v${VERSION}
```

---

## 4. Monitoring & Observability

### 4.1 CloudWatch Monitoring

**Key Metrics:**
```
Application:
- Requests/second (target: 1000 req/s)
- Latency: p50, p95, p99 (target: p95 < 500ms)
- Error rate (target: < 0.1%)
- Cache hit rate (target: > 90%)

Infrastructure:
- CPU utilization (target: < 70%)
- Memory utilization (target: < 80%)
- Disk I/O
- Network throughput

Database:
- Connections (target: < 80% of max)
- Query latency (target: p95 < 100ms)
- Replication lag (target: < 1s)
- Transactions/second

Cache:
- Memory utilization (target: < 85%)
- Evictions (target: 0)
- Hit rate (target: > 95%)
```

**CloudWatch Alarms:**
```
Alarm: HighErrorRate
- Metric: ErrorRate
- Threshold: > 1%
- Duration: 2 minutes
- Action: PagerDuty (high priority)

Alarm: HighLatency
- Metric: LatencyP95
- Threshold: > 1000ms
- Duration: 5 minutes
- Action: SNS notification

Alarm: DatabaseConnectionPooled
- Metric: DBConnections
- Threshold: > 80% of max
- Duration: 5 minutes
- Action: Alert database team
```

### 4.2 Distributed Tracing

**X-Ray Configuration:**
```python
from aws_xray_sdk.core import xray_recorder

xray_recorder.configure(
    service='banking-llm',
    sampling=True,
    context_missing='LOG_ERROR'
)

@app.middleware("http")
async def trace_requests(request: Request, call_next):
    # X-Ray automatically traces requests
    response = await call_next(request)
    return response
```

### 4.3 Logging

**CloudWatch Logs:**
```
Log Groups:
- /aws/ecs/banking-llm/api
- /aws/ecs/banking-llm/worker
- /aws/rds/instance/banking-llm-prod/error
- /aws/rds/instance/banking-llm-prod/general

Retention: 90 days (normal), 10 years (compliance events)

Filters:
- ERROR: Any ERROR level
- WARNING: API latency > 1s
- SECURITY: Authentication failures
- COMPLIANCE: SAR/CTR filed
```

---

## 5. Backup & Disaster Recovery

### 5.1 Backup Strategy

**RDS PostgreSQL Backups:**
```
Automated:
- Daily full backups (3:00 AM UTC)
- Continuous WAL archival to S3
- Retention: 35 days
- Recovery Point Objective (RPO): 5 minutes

Manual:
- Weekly snapshots (retained 60 days)
- Monthly snapshots (retained 1 year)
- Final snapshot before major upgrades

Backup Testing:
- Monthly restore test (verify integrity)
- Quarterly full failover test
- Annual DR drill
```

**Application Data Backups (S3):**
```
Documents:
- All customer documents encrypted
- Lifecycle: Delete after 7 years
- Versioning: Enabled
- Cross-region replication: Enabled

Logs:
- Audit logs: Retained 10 years (compliance)
- Application logs: Retained 2 years
- Access logs: Retained 1 year
```

### 5.2 Disaster Recovery Plan

**RTO/RPO Targets:**
```
Recovery Time Objective (RTO): 4 hours
Recovery Point Objective (RPO): 1 hour

Tier 1 (Critical) - RTO: 30 min, RPO: 5 min
- Customer-facing API
- Authentication service
- Payment processing

Tier 2 (Important) - RTO: 2 hours, RPO: 30 min
- Chat/LLM service
- Admin services
- Reporting

Tier 3 (Supportive) - RTO: 4 hours, RPO: 1 hour
- Analytics
- Archive/cold data
- Non-critical services
```

**Failover Procedure:**
```bash
# 1. Detect failure
# CloudWatch alarm triggers
# PagerDuty notification sent

# 2. Initiate failover
aws rds promote-read-replica \
  --db-instance-identifier banking-llm-standby

# 3. Update Route 53 (automatic)
# DNS failover point to secondary region

# 4. Verify services
curl https://api.banking-llm.com/health/detailed

# 5. Notify stakeholders
# Email, Slack, status page update

# 6. Document incident
# Post-incident review (24-48 hours)
```

---

## 6. Security Hardening

### 6.1 Network Security

**Security Groups:**
```
ALB Security Group:
- Inbound: 443 (HTTPS) from 0.0.0.0/0
- Outbound: 8000 (HTTP) to ECS security group

ECS Security Group:
- Inbound: 8000 from ALB
- Outbound: 5432 to RDS, 6379 to Redis, 443 to internet

RDS Security Group:
- Inbound: 5432 only from ECS
- Outbound: None (database only)

Redis Security Group:
- Inbound: 6379 only from ECS
- Outbound: None (cache only)
```

**VPC Flow Logs:**
```
Enabled: Yes
Traffic Type: All (ACCEPT + REJECT)
Destination: CloudWatch Logs
Log Format: Version 5 (with action + flow direction)
Duration: 600 seconds (10 minutes)
```

### 6.2 Secrets Management

**AWS Secrets Manager:**
```
Secrets stored:
- Database credentials
- API keys (external services)
- SSL certificates
- Encryption keys
- OAuth tokens

Rotation:
- Automatic: Every 30 days
- Manual: On demand
- Notifications: Slack, email
```

**Accessing Secrets (Application):**
```python
import boto3
import json

client = boto3.client('secretsmanager')

def get_db_credentials():
    secret = client.get_secret_value(
        SecretId='prod/banking-llm/db'
    )
    return json.loads(secret['SecretString'])

# Usage
credentials = get_db_credentials()
db_url = f"postgresql://{credentials['username']}:{credentials['password']}@{credentials['host']}/banking_llm"
```

---

## 7. Cost Optimization

### 7.1 Cost Breakdown (Monthly)

```
Compute (ECS):
- 4 instances × $0.13/hour × 720 hours = $374
- Network load balancer: $16.20
- Subtotal: $390

Database (RDS):
- r6g.2xlarge instance: $1,192
- Storage (500GB, gp3): $50
- Backup storage: $50
- Subtotal: $1,292

Cache (ElastiCache):
- 3 × cache.r6g.xlarge: $597
- Data transfer: $50
- Subtotal: $647

Networking:
- Data transfer (inter-AZ): $100
- NAT gateway: $32
- Subtotal: $132

Other:
- CloudWatch: $50
- S3 (backups): $50
- KMS: $1/month per key
- Subtotal: $100+

**Total: ~$2,500/month**

Cost Optimization:
- Use Reserved Instances (40% savings) → $1,500/month
- Use Spot Instances for non-critical (50% savings)
- Optimize query patterns (reduce data transfer)
```

---

## 8. Monitoring Dashboard

**Key Metrics Dashboard (Sample):**

```
API Performance:
- Requests/sec: [___] (Target: 1000)
- Latency P95: [___] ms (Target: < 500)
- Error Rate: [___] % (Target: < 0.1%)
- Availability: [___] % (Target: > 99.95%)

Infrastructure:
- CPU: [___] % (Target: < 70%)
- Memory: [___] % (Target: < 80%)
- Disk: [___] % (Target: < 85%)
- Network: [___] Mbps

Database:
- Connections: [___] / [Max] (Target: < 80%)
- Latency P95: [___] ms (Target: < 100)
- Replication Lag: [___] s (Target: < 1)
- Transactions/sec: [___]

Cache:
- Memory: [___] % (Target: < 85%)
- Hit Rate: [___] % (Target: > 95%)
- Evictions: [___] (Target: 0)

Cost:
- Daily: $[___]
- Monthly Projected: $[___]
- YoY: $[___]
```

---

## Next Steps

1. **Immediate (Week 1):**
   - [ ] Setup local development environment
   - [ ] Verify GitHub Codespaces works
   - [ ] Configure pre-commit hooks

2. **Short-term (Week 2-4):**
   - [ ] Deploy to staging (manually)
   - [ ] Setup monitoring and alerting
   - [ ] Create runbooks for common operations

3. **Medium-term (Month 2-3):**
   - [ ] Implement CI/CD pipeline (GitHub Actions)
   - [ ] Setup disaster recovery
   - [ ] Performance testing (load testing)

4. **Long-term (Month 4+):**
   - [ ] Multi-region deployment
   - [ ] Kubernetes migration (optional)
   - [ ] Advanced observability (Datadog, New Relic)
