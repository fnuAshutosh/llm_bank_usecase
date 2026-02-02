"""Database models - SQLAlchemy ORM for Supabase PostgreSQL"""

import enum
from datetime import datetime
from typing import Optional
from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
)
from sqlalchemy import (
    Enum as SQLEnum,
)
from sqlalchemy.dialects.postgresql import INET, JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from .connection import Base


class KYCStatus(str, enum.Enum):
    """KYC verification status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    VERIFIED = "verified"
    REJECTED = "rejected"
    EXPIRED = "expired"


class AccountType(str, enum.Enum):
    """Account types"""
    CHECKING = "checking"
    SAVINGS = "savings"
    CREDIT_CARD = "credit_card"
    LOAN = "loan"
    INVESTMENT = "investment"


class TransactionType(str, enum.Enum):
    """Transaction types"""
    DEBIT = "debit"
    CREDIT = "credit"
    TRANSFER = "transfer"
    PAYMENT = "payment"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


class TransactionStatus(str, enum.Enum):
    """Transaction status"""
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REVERSED = "reversed"


class MessageRole(str, enum.Enum):
    """Message roles"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


# ============================================================================
# Customer & Identity
# ============================================================================

class Customer(Base):
    """Customer table - Core customer information"""
    __tablename__ = "customers"
    
    customer_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Personal Information (encrypted in production)
    email = Column(String(255), unique=True, nullable=False, index=True)
    phone = Column(String(20))
    first_name = Column(String(100), nullable=False)
    last_name = Column(String(100), nullable=False)
    date_of_birth = Column(DateTime)
    
    # Encrypted fields (store as binary in production)
    ssn_encrypted = Column(LargeBinary)  # AES-256 encrypted
    address_encrypted = Column(LargeBinary)
    
    # KYC & Compliance
    kyc_status = Column(SQLEnum(KYCStatus), default=KYCStatus.PENDING, index=True)
    kyc_verified_at = Column(DateTime)
    risk_score = Column(Integer, default=50)  # 0-100 (lower = less risky)
    sanctions_check_passed = Column(Boolean, default=False)
    
    # Account metadata
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login_at = Column(DateTime)
    
    # Relationships
    accounts = relationship("Account", back_populates="customer", cascade="all, delete-orphan")
    conversations = relationship("Conversation", back_populates="customer", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index("idx_customer_email", "email"),
        Index("idx_customer_kyc_status", "kyc_status"),
        Index("idx_customer_created_at", "created_at"),
    )


# ============================================================================
# Accounts & Transactions
# ============================================================================

class Account(Base):
    """Account table - Customer bank accounts"""
    __tablename__ = "accounts"
    
    account_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False)
    
    # Account details
    account_number = Column(String(20), unique=True, nullable=False, index=True)
    account_type = Column(SQLEnum(AccountType), nullable=False)
    balance = Column(Float, default=0.0, nullable=False)
    currency = Column(String(3), default="USD")
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_frozen = Column(Boolean, default=False)
    overdraft_limit = Column(Float, default=0.0)
    
    # Timestamps
    opened_at = Column(DateTime, server_default=func.now())
    closed_at = Column(DateTime)
    
    # Relationships
    customer = relationship("Customer", back_populates="accounts")
    transactions = relationship("Transaction", back_populates="account", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_account_customer_id", "customer_id"),
        Index("idx_account_number", "account_number"),
    )


class Transaction(Base):
    """Transaction table - All financial transactions"""
    __tablename__ = "transactions"
    
    transaction_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    account_id = Column(UUID(as_uuid=True), ForeignKey("accounts.account_id"), nullable=False)
    
    # Transaction details
    transaction_type = Column(SQLEnum(TransactionType), nullable=False)
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    description = Column(Text)
    merchant = Column(String(255))
    
    # Status & fraud detection
    status = Column(SQLEnum(TransactionStatus), default=TransactionStatus.PENDING)
    fraud_score = Column(Float, default=0.0)  # 0-1 (higher = more suspicious)
    is_flagged = Column(Boolean, default=False)
    flagged_reason = Column(Text)
    
    # Balance tracking
    balance_before = Column(Float)
    balance_after = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now(), index=True)
    completed_at = Column(DateTime)
    
    # Relationships
    account = relationship("Account", back_populates="transactions")
    
    __table_args__ = (
        Index("idx_transaction_account_id", "account_id"),
        Index("idx_transaction_created_at", "created_at"),
        Index("idx_transaction_fraud_score", "fraud_score"),
    )


# ============================================================================
# Conversations & Messages (LLM Interactions)
# ============================================================================

class Conversation(Base):
    """Conversation table - LLM chat sessions"""
    __tablename__ = "conversations"
    
    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"), nullable=False)
    
    # Session metadata
    session_id = Column(String(255), index=True)
    title = Column(String(255))
    
    # Conversation stats
    total_messages = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)
    
    # Quality metrics
    sentiment_score = Column(Float)  # -1 to 1
    escalated_to_human = Column(Boolean, default=False)
    customer_satisfaction = Column(Integer)  # 1-5 stars
    
    # Timestamps
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime)
    
    # Relationships
    customer = relationship("Customer", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index("idx_conversation_customer_id", "customer_id"),
        Index("idx_conversation_started_at", "started_at"),
    )


class Message(Base):
    """Message table - Individual chat messages"""
    __tablename__ = "messages"
    
    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.conversation_id"), nullable=False)
    
    # Message content
    role = Column(SQLEnum(MessageRole), nullable=False)
    content = Column(Text, nullable=False)
    content_encrypted = Column(LargeBinary)  # Encrypted version for PII
    
    # Metadata
    model = Column(String(100))  # e.g., "llama2-34b"
    tokens = Column(Integer)
    latency_ms = Column(Integer)
    cost = Column(Float)
    
    # Security & compliance
    pii_detected = Column(Boolean, default=False)
    pii_types = Column(JSONB)  # List of detected PII types
    sanitized_content = Column(Text)  # PII-masked version
    
    # Timestamps
    timestamp = Column(DateTime, server_default=func.now(), index=True)
    
    # Relationships
    conversation = relationship("Conversation", back_populates="messages")
    
    __table_args__ = (
        Index("idx_message_conversation_id", "conversation_id"),
        Index("idx_message_timestamp", "timestamp"),
    )


# ============================================================================
# Audit & Compliance
# ============================================================================

class AuditLog(Base):
    """Audit log table - Immutable audit trail"""
    __tablename__ = "audit_logs"
    
    log_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    
    # Event details
    event_type = Column(String(100), nullable=False, index=True)
    action = Column(String(255), nullable=False)
    resource_type = Column(String(100))
    resource_id = Column(String(255))
    
    # Actor details
    customer_id = Column(UUID(as_uuid=True))
    user_id = Column(UUID(as_uuid=True))
    ip_address = Column(INET)
    user_agent = Column(Text)
    
    # Result
    result = Column(String(50))  # success, failure, error
    error_message = Column(Text)
    
    # Additional context (renamed from metadata to avoid SQLAlchemy conflict)
    event_metadata = Column(JSONB)
    
    # Timestamp (immutable)
    timestamp = Column(DateTime, server_default=func.now(), nullable=False, index=True)
    
    __table_args__ = (
        Index("idx_audit_event_type", "event_type"),
        Index("idx_audit_customer_id", "customer_id"),
        Index("idx_audit_timestamp", "timestamp"),
    )


# ============================================================================
# Fraud Detection
# ============================================================================

class FraudAlert(Base):
    """Fraud alert table - Suspicious activity tracking"""
    __tablename__ = "fraud_alerts"
    
    alert_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    customer_id = Column(UUID(as_uuid=True), ForeignKey("customers.customer_id"))
    transaction_id = Column(UUID(as_uuid=True), ForeignKey("transactions.transaction_id"))
    
    # Alert details
    alert_type = Column(String(100), nullable=False)
    severity = Column(String(20))  # low, medium, high, critical
    fraud_score = Column(Float, nullable=False)
    description = Column(Text)
    
    # Resolution
    status = Column(String(50), default="open")  # open, investigating, resolved, false_positive
    resolved_at = Column(DateTime)
    resolved_by = Column(String(255))
    resolution_notes = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    
    __table_args__ = (
        Index("idx_fraud_alert_customer_id", "customer_id"),
        Index("idx_fraud_alert_status", "status"),
        Index("idx_fraud_alert_created_at", "created_at"),
    )
