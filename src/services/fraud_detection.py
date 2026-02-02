"""Fraud Detection Service - Real-time fraud analysis"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

import numpy as np
from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import (
    Account,
    Customer,
    FraudAlert,
    Transaction,
    TransactionType,
)
from ..observability.metrics import track_fraud_check
from ..observability.tracing import trace_function

logger = logging.getLogger(__name__)


class FraudDetectionService:
    """ML-based fraud detection and analysis"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        
        # Fraud detection thresholds
        self.HIGH_RISK_THRESHOLD = 0.7
        self.MEDIUM_RISK_THRESHOLD = 0.4
        self.UNUSUAL_AMOUNT_MULTIPLIER = 3.0
        self.VELOCITY_LIMIT = 5  # Max transactions in time window
        self.VELOCITY_WINDOW_MINUTES = 10
    
    @trace_function("analyze_transaction")
    async def analyze_transaction(
        self,
        transaction_id: str,
        customer_id: str,
        amount: float,
        merchant: Optional[str] = None,
        location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive fraud analysis for a transaction
        
        Checks:
        - Amount anomaly detection
        - Velocity checks (multiple rapid transactions)
        - Geographic anomalies
        - Merchant risk assessment
        - Customer historical patterns
        - Time-of-day patterns
        """
        logger.info(f"Analyzing transaction {transaction_id} for fraud")
        
        try:
            fraud_signals = []
            fraud_score = 0.0
            
            # 1. Amount Anomaly Detection
            amount_risk = await self._check_amount_anomaly(customer_id, amount)
            if amount_risk > 0:
                fraud_signals.append({
                    "type": "unusual_amount",
                    "score": amount_risk,
                    "details": f"Amount ${amount:.2f} is {amount_risk*100:.0f}% higher than normal"
                })
                fraud_score += amount_risk * 0.3
            
            # 2. Velocity Check
            velocity_risk = await self._check_transaction_velocity(customer_id)
            if velocity_risk > 0:
                fraud_signals.append({
                    "type": "high_velocity",
                    "score": velocity_risk,
                    "details": f"Multiple transactions in short time window"
                })
                fraud_score += velocity_risk * 0.25
            
            # 3. Time-of-day Pattern
            time_risk = await self._check_time_pattern(customer_id)
            if time_risk > 0:
                fraud_signals.append({
                    "type": "unusual_time",
                    "score": time_risk,
                    "details": "Transaction at unusual time"
                })
                fraud_score += time_risk * 0.15
            
            # 4. Merchant Risk Assessment
            if merchant:
                merchant_risk = await self._check_merchant_risk(customer_id, merchant)
                if merchant_risk > 0:
                    fraud_signals.append({
                        "type": "risky_merchant",
                        "score": merchant_risk,
                        "details": f"Merchant '{merchant}' flagged as risky"
                    })
                    fraud_score += merchant_risk * 0.2
            
            # 5. Account Age and Activity
            account_risk = await self._check_account_risk(customer_id)
            if account_risk > 0:
                fraud_signals.append({
                    "type": "account_risk",
                    "score": account_risk,
                    "details": "New account or unusual activity pattern"
                })
                fraud_score += account_risk * 0.1
            
            # Normalize fraud score to 0-1
            fraud_score = min(fraud_score, 1.0)
            
            # Determine severity
            if fraud_score >= self.HIGH_RISK_THRESHOLD:
                severity = "critical"
                action = "block"
            elif fraud_score >= self.MEDIUM_RISK_THRESHOLD:
                severity = "high"
                action = "review"
            elif fraud_score > 0:
                severity = "medium"
                action = "monitor"
            else:
                severity = "low"
                action = "approve"
            
            result = {
                "transaction_id": transaction_id,
                "fraud_score": round(fraud_score, 3),
                "severity": severity,
                "recommended_action": action,
                "signals": fraud_signals,
                "is_flagged": fraud_score >= self.MEDIUM_RISK_THRESHOLD,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Track metrics
            track_fraud_check(severity, "transaction_analysis")
            
            # Create fraud alert if needed
            if result["is_flagged"]:
                await self._create_fraud_alert(
                    customer_id=customer_id,
                    transaction_id=transaction_id,
                    fraud_score=fraud_score,
                    severity=severity,
                    signals=fraud_signals,
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fraud analysis: {e}")
            # Fail open - don't block transaction on analysis error
            return {
                "fraud_score": 0.0,
                "severity": "unknown",
                "recommended_action": "approve",
                "error": str(e),
            }
    
    async def _check_amount_anomaly(self, customer_id: str, amount: float) -> float:
        """Check if transaction amount is anomalous"""
        try:
            # Get customer's average transaction amount (last 90 days)
            cutoff_date = datetime.utcnow() - timedelta(days=90)
            
            stmt = (
                select(func.avg(func.abs(Transaction.amount)))
                .join(Account)
                .where(Account.customer_id == UUID(customer_id))
                .where(Transaction.created_at >= cutoff_date)
            )
            
            result = await self.db.execute(stmt)
            avg_amount = result.scalar() or 100.0  # Default to $100
            
            # Calculate risk based on how much it exceeds average
            if abs(amount) > avg_amount * self.UNUSUAL_AMOUNT_MULTIPLIER:
                risk = min(abs(amount) / (avg_amount * self.UNUSUAL_AMOUNT_MULTIPLIER), 1.0)
                return risk
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking amount anomaly: {e}")
            return 0.0
    
    async def _check_transaction_velocity(self, customer_id: str) -> float:
        """Check for rapid successive transactions"""
        try:
            # Count transactions in last N minutes
            cutoff_time = datetime.utcnow() - timedelta(minutes=self.VELOCITY_WINDOW_MINUTES)
            
            stmt = (
                select(func.count(Transaction.transaction_id))
                .join(Account)
                .where(Account.customer_id == UUID(customer_id))
                .where(Transaction.created_at >= cutoff_time)
            )
            
            result = await self.db.execute(stmt)
            transaction_count = result.scalar() or 0
            
            if transaction_count >= self.VELOCITY_LIMIT:
                risk = min(transaction_count / (self.VELOCITY_LIMIT * 2), 1.0)
                return risk
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking velocity: {e}")
            return 0.0
    
    async def _check_time_pattern(self, customer_id: str) -> float:
        """Check if transaction time is unusual"""
        try:
            current_hour = datetime.utcnow().hour
            
            # Transactions between 2 AM - 6 AM are higher risk
            if 2 <= current_hour < 6:
                return 0.3
            
            # Transactions between midnight - 2 AM
            if 0 <= current_hour < 2:
                return 0.2
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking time pattern: {e}")
            return 0.0
    
    async def _check_merchant_risk(self, customer_id: str, merchant: str) -> float:
        """Check merchant risk based on historical data"""
        try:
            # Check if this is a new merchant for the customer
            stmt = (
                select(func.count(Transaction.transaction_id))
                .join(Account)
                .where(Account.customer_id == UUID(customer_id))
                .where(Transaction.merchant == merchant)
            )
            
            result = await self.db.execute(stmt)
            merchant_count = result.scalar() or 0
            
            # New merchant is slightly risky
            if merchant_count == 0:
                return 0.2
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking merchant risk: {e}")
            return 0.0
    
    async def _check_account_risk(self, customer_id: str) -> float:
        """Check account age and activity patterns"""
        try:
            # Get customer
            stmt = select(Customer).where(Customer.customer_id == UUID(customer_id))
            result = await self.db.execute(stmt)
            customer = result.scalar_one_or_none()
            
            if not customer:
                return 0.5  # Unknown customer is risky
            
            # New account (less than 30 days) is riskier
            account_age = (datetime.utcnow() - customer.created_at).days
            if account_age < 30:
                risk = 0.3 * (1 - account_age / 30)
                return risk
            
            # High risk score customer
            if customer.risk_score and customer.risk_score > 70:
                return 0.4
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error checking account risk: {e}")
            return 0.0
    
    @trace_function("create_fraud_alert")
    async def _create_fraud_alert(
        self,
        customer_id: str,
        transaction_id: str,
        fraud_score: float,
        severity: str,
        signals: List[Dict],
    ) -> FraudAlert:
        """Create a fraud alert in the database"""
        try:
            alert = FraudAlert(
                customer_id=UUID(customer_id),
                transaction_id=UUID(transaction_id),
                alert_type="transaction_fraud",
                severity=severity,
                fraud_score=fraud_score,
                description=f"Fraud detected: {len(signals)} signals triggered",
                status="open",
            )
            
            self.db.add(alert)
            await self.db.commit()
            await self.db.refresh(alert)
            
            logger.warning(
                f"Fraud alert created: {alert.alert_id} - "
                f"Score: {fraud_score:.3f}, Severity: {severity}"
            )
            
            return alert
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating fraud alert: {e}")
            raise
    
    @trace_function("get_customer_fraud_alerts")
    async def get_customer_fraud_alerts(
        self,
        customer_id: str,
        status: Optional[str] = None,
    ) -> List[FraudAlert]:
        """Get fraud alerts for a customer"""
        try:
            stmt = select(FraudAlert).where(FraudAlert.customer_id == UUID(customer_id))
            
            if status:
                stmt = stmt.where(FraudAlert.status == status)
            
            stmt = stmt.order_by(FraudAlert.created_at.desc())
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting fraud alerts: {e}")
            return []
    
    @trace_function("resolve_fraud_alert")
    async def resolve_fraud_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution: str,
        notes: Optional[str] = None,
    ) -> FraudAlert:
        """Resolve a fraud alert"""
        try:
            stmt = select(FraudAlert).where(FraudAlert.alert_id == UUID(alert_id))
            result = await self.db.execute(stmt)
            alert = result.scalar_one_or_none()
            
            if not alert:
                raise ValueError("Alert not found")
            
            alert.status = resolution
            alert.resolved_at = datetime.utcnow()
            alert.resolved_by = resolved_by
            alert.resolution_notes = notes
            
            await self.db.commit()
            await self.db.refresh(alert)
            
            logger.info(f"Fraud alert resolved: {alert_id}")
            return alert
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error resolving fraud alert: {e}")
            raise
    
    @trace_function("get_fraud_statistics")
    async def get_fraud_statistics(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get fraud statistics for a period"""
        try:
            # Total alerts
            stmt = (
                select(func.count(FraudAlert.alert_id))
                .where(and_(
                    FraudAlert.created_at >= start_date,
                    FraudAlert.created_at <= end_date
                ))
            )
            result = await self.db.execute(stmt)
            total_alerts = result.scalar() or 0
            
            # Alerts by severity
            stmt = (
                select(FraudAlert.severity, func.count(FraudAlert.alert_id))
                .where(and_(
                    FraudAlert.created_at >= start_date,
                    FraudAlert.created_at <= end_date
                ))
                .group_by(FraudAlert.severity)
            )
            result = await self.db.execute(stmt)
            severity_counts = {row[0]: row[1] for row in result}
            
            # Average fraud score
            stmt = (
                select(func.avg(FraudAlert.fraud_score))
                .where(and_(
                    FraudAlert.created_at >= start_date,
                    FraudAlert.created_at <= end_date
                ))
            )
            result = await self.db.execute(stmt)
            avg_score = result.scalar() or 0.0
            
            return {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "total_alerts": total_alerts,
                "by_severity": severity_counts,
                "average_fraud_score": round(avg_score, 3),
            }
            
        except Exception as e:
            logger.error(f"Error getting fraud statistics: {e}")
            raise
