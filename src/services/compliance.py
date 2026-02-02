"""Compliance Service - Regulatory compliance and audit management"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import (
    AuditLog,
    Conversation,
    Customer,
    Message,
    Transaction,
)
from ..observability.tracing import trace_function
from ..security.pii_detection import PIIDetector

logger = logging.getLogger(__name__)


class ComplianceService:
    """Handle regulatory compliance, audit trails, and data governance"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pii_detector = PIIDetector()
        
        # Compliance thresholds
        self.LARGE_TRANSACTION_THRESHOLD = 10000.0  # $10,000 - CTR threshold
        self.SUSPICIOUS_ACTIVITY_THRESHOLD = 5000.0  # $5,000 - SAR consideration
        self.DATA_RETENTION_YEARS = 7  # Banking requirement
    
    @trace_function("check_transaction_compliance")
    async def check_transaction_compliance(
        self,
        transaction_id: str,
        amount: float,
        transaction_type: str,
        customer_id: str,
    ) -> Dict[str, Any]:
        """
        Check if transaction meets compliance requirements
        
        Checks:
        - Currency Transaction Report (CTR) requirements
        - Suspicious Activity Report (SAR) triggers
        - Anti-Money Laundering (AML) patterns
        - Transaction limits
        """
        logger.info(f"Checking compliance for transaction {transaction_id}")
        
        try:
            compliance_issues = []
            flags = []
            
            # 1. CTR Check - Transactions over $10,000
            if abs(amount) >= self.LARGE_TRANSACTION_THRESHOLD:
                compliance_issues.append({
                    "type": "CTR_REQUIRED",
                    "severity": "high",
                    "description": f"Transaction amount ${abs(amount):,.2f} exceeds $10,000 threshold",
                    "action_required": "File Currency Transaction Report (CTR) with FinCEN",
                })
                flags.append("ctr_required")
            
            # 2. Pattern Analysis - Multiple transactions near threshold
            structuring_risk = await self._check_structuring(customer_id, amount)
            if structuring_risk["is_suspicious"]:
                compliance_issues.append({
                    "type": "STRUCTURING_SUSPECTED",
                    "severity": "critical",
                    "description": "Pattern suggests intentional structuring to avoid reporting",
                    "action_required": "File Suspicious Activity Report (SAR)",
                    "details": structuring_risk,
                })
                flags.append("sar_required")
            
            # 3. Velocity Check - Rapid transactions
            velocity_issue = await self._check_transaction_velocity_compliance(customer_id)
            if velocity_issue:
                compliance_issues.append(velocity_issue)
                flags.append("velocity_concern")
            
            # 4. Customer Due Diligence
            cdd_status = await self._check_customer_due_diligence(customer_id)
            if not cdd_status["compliant"]:
                compliance_issues.append({
                    "type": "CDD_INCOMPLETE",
                    "severity": "high",
                    "description": "Customer due diligence requirements not met",
                    "action_required": "Complete enhanced due diligence",
                    "details": cdd_status,
                })
                flags.append("cdd_required")
            
            result = {
                "transaction_id": transaction_id,
                "is_compliant": len([i for i in compliance_issues if i["severity"] in ["high", "critical"]]) == 0,
                "compliance_issues": compliance_issues,
                "flags": flags,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Log to audit trail
            await self._create_audit_log(
                event_type="compliance_check",
                action="check_transaction",
                resource_type="transaction",
                resource_id=transaction_id,
                customer_id=customer_id,
                result="flagged" if flags else "compliant",
                metadata=result,
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in compliance check: {e}")
            raise
    
    async def _check_structuring(self, customer_id: str, current_amount: float) -> Dict[str, Any]:
        """Check for structuring patterns (breaking up large transactions)"""
        try:
            # Get transactions in last 24 hours
            cutoff = datetime.utcnow() - timedelta(hours=24)
            
            stmt = (
                select(func.sum(func.abs(Transaction.amount)), func.count(Transaction.transaction_id))
                .join(Account)
                .where(Account.customer_id == UUID(customer_id))
                .where(Transaction.created_at >= cutoff)
                .where(Transaction.amount < self.LARGE_TRANSACTION_THRESHOLD)
                .where(Transaction.amount > 5000)  # Multiple transactions between $5K-$10K
            )
            
            result = await self.db.execute(stmt)
            row = result.first()
            total_amount = row[0] or 0
            transaction_count = row[1] or 0
            
            # Suspicious if multiple transactions sum to >$10K
            is_suspicious = (
                transaction_count >= 2 and
                total_amount + abs(current_amount) >= self.LARGE_TRANSACTION_THRESHOLD
            )
            
            return {
                "is_suspicious": is_suspicious,
                "total_amount": total_amount,
                "transaction_count": transaction_count,
                "timeframe": "24_hours",
            }
            
        except Exception as e:
            logger.error(f"Error checking structuring: {e}")
            return {"is_suspicious": False, "error": str(e)}
    
    async def _check_transaction_velocity_compliance(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Check if transaction velocity violates compliance rules"""
        try:
            # Count transactions in last hour
            cutoff = datetime.utcnow() - timedelta(hours=1)
            
            stmt = (
                select(func.count(Transaction.transaction_id))
                .join(Account)
                .where(Account.customer_id == UUID(customer_id))
                .where(Transaction.created_at >= cutoff)
            )
            
            result = await self.db.execute(stmt)
            count = result.scalar() or 0
            
            # Flag if more than 10 transactions in 1 hour
            if count > 10:
                return {
                    "type": "HIGH_VELOCITY",
                    "severity": "medium",
                    "description": f"{count} transactions in last hour",
                    "action_required": "Monitor for suspicious patterns",
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error checking velocity compliance: {e}")
            return None
    
    async def _check_customer_due_diligence(self, customer_id: str) -> Dict[str, Any]:
        """Check if customer due diligence is complete"""
        try:
            stmt = select(Customer).where(Customer.customer_id == UUID(customer_id))
            result = await self.db.execute(stmt)
            customer = result.scalar_one_or_none()
            
            if not customer:
                return {"compliant": False, "reason": "Customer not found"}
            
            issues = []
            
            # Check KYC verification
            if customer.kyc_status != "verified":
                issues.append("KYC not verified")
            
            # Check sanctions screening
            if not customer.sanctions_check_passed:
                issues.append("Sanctions check failed or not completed")
            
            # Check if high-risk customer has enhanced due diligence
            if customer.risk_score and customer.risk_score > 70:
                issues.append("High-risk customer requires enhanced due diligence")
            
            return {
                "compliant": len(issues) == 0,
                "issues": issues,
                "kyc_status": customer.kyc_status.value if customer.kyc_status else "unknown",
                "risk_score": customer.risk_score,
            }
            
        except Exception as e:
            logger.error(f"Error checking CDD: {e}")
            return {"compliant": False, "reason": str(e)}
    
    @trace_function("check_data_privacy_compliance")
    async def check_data_privacy_compliance(
        self,
        message_content: str,
        customer_id: str,
    ) -> Dict[str, Any]:
        """
        Check if message handling complies with data privacy regulations
        (GDPR, CCPA, etc.)
        """
        try:
            # Detect PII in message
            pii_results = self.pii_detector.detect_pii(message_content)
            
            has_pii = len(pii_results) > 0
            
            compliance_status = {
                "has_pii": has_pii,
                "pii_types": [r["entity_type"] for r in pii_results],
                "requires_encryption": has_pii,
                "requires_audit_log": has_pii,
                "retention_policy": f"{self.DATA_RETENTION_YEARS} years",
                "deletion_required_after": (
                    datetime.utcnow() + timedelta(days=self.DATA_RETENTION_YEARS * 365)
                ).isoformat(),
            }
            
            # Log PII handling
            if has_pii:
                await self._create_audit_log(
                    event_type="pii_handling",
                    action="pii_detected",
                    customer_id=customer_id,
                    result="detected",
                    metadata={
                        "pii_types": compliance_status["pii_types"],
                        "pii_count": len(pii_results),
                    },
                )
            
            return compliance_status
            
        except Exception as e:
            logger.error(f"Error checking data privacy compliance: {e}")
            raise
    
    @trace_function("create_audit_log")
    async def _create_audit_log(
        self,
        event_type: str,
        action: str,
        result: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        customer_id: Optional[str] = None,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> AuditLog:
        """Create immutable audit log entry"""
        try:
            audit_log = AuditLog(
                event_type=event_type,
                action=action,
                resource_type=resource_type,
                resource_id=resource_id,
                customer_id=UUID(customer_id) if customer_id else None,
                user_id=UUID(user_id) if user_id else None,
                ip_address=ip_address,
                result=result,
                metadata=metadata,
            )
            
            self.db.add(audit_log)
            await self.db.commit()
            await self.db.refresh(audit_log)
            
            return audit_log
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating audit log: {e}")
            raise
    
    @trace_function("get_audit_trail")
    async def get_audit_trail(
        self,
        customer_id: Optional[str] = None,
        event_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[AuditLog]:
        """Retrieve audit trail with filters"""
        try:
            stmt = select(AuditLog)
            
            # Apply filters
            if customer_id:
                stmt = stmt.where(AuditLog.customer_id == UUID(customer_id))
            
            if event_type:
                stmt = stmt.where(AuditLog.event_type == event_type)
            
            if start_date:
                stmt = stmt.where(AuditLog.timestamp >= start_date)
            
            if end_date:
                stmt = stmt.where(AuditLog.timestamp <= end_date)
            
            stmt = stmt.order_by(AuditLog.timestamp.desc()).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error retrieving audit trail: {e}")
            return []
    
    @trace_function("generate_compliance_report")
    async def generate_compliance_report(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        try:
            # CTR count
            stmt = (
                select(func.count(Transaction.transaction_id))
                .where(and_(
                    Transaction.created_at >= start_date,
                    Transaction.created_at <= end_date,
                    func.abs(Transaction.amount) >= self.LARGE_TRANSACTION_THRESHOLD
                ))
            )
            result = await self.db.execute(stmt)
            ctr_count = result.scalar() or 0
            
            # Flagged transactions
            stmt = (
                select(func.count(Transaction.transaction_id))
                .where(and_(
                    Transaction.created_at >= start_date,
                    Transaction.created_at <= end_date,
                    Transaction.is_flagged == True
                ))
            )
            result = await self.db.execute(stmt)
            flagged_count = result.scalar() or 0
            
            # KYC completion rate
            stmt = select(
                func.count(Customer.customer_id),
                func.count(Customer.customer_id).filter(Customer.kyc_status == "verified")
            ).where(Customer.created_at >= start_date)
            result = await self.db.execute(stmt)
            row = result.first()
            total_customers = row[0] or 1
            verified_customers = row[1] or 0
            kyc_completion_rate = (verified_customers / total_customers) * 100
            
            # Audit log count
            stmt = (
                select(func.count(AuditLog.log_id))
                .where(and_(
                    AuditLog.timestamp >= start_date,
                    AuditLog.timestamp <= end_date
                ))
            )
            result = await self.db.execute(stmt)
            audit_log_count = result.scalar() or 0
            
            report = {
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "currency_transaction_reports": ctr_count,
                "flagged_transactions": flagged_count,
                "kyc_completion_rate": round(kyc_completion_rate, 2),
                "total_customers_onboarded": total_customers,
                "verified_customers": verified_customers,
                "audit_log_entries": audit_log_count,
                "generated_at": datetime.utcnow().isoformat(),
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            raise
    
    @trace_function("check_data_retention")
    async def check_data_retention(self) -> Dict[str, Any]:
        """Check if any data needs to be deleted per retention policy"""
        try:
            retention_cutoff = datetime.utcnow() - timedelta(days=self.DATA_RETENTION_YEARS * 365)
            
            # Find old conversations
            stmt = (
                select(func.count(Conversation.conversation_id))
                .where(Conversation.ended_at < retention_cutoff)
            )
            result = await self.db.execute(stmt)
            conversations_to_delete = result.scalar() or 0
            
            # Find old audit logs (never delete, but archive)
            stmt = (
                select(func.count(AuditLog.log_id))
                .where(AuditLog.timestamp < retention_cutoff)
            )
            result = await self.db.execute(stmt)
            logs_to_archive = result.scalar() or 0
            
            return {
                "retention_period_years": self.DATA_RETENTION_YEARS,
                "cutoff_date": retention_cutoff.isoformat(),
                "conversations_to_delete": conversations_to_delete,
                "audit_logs_to_archive": logs_to_archive,
                "action_required": conversations_to_delete > 0 or logs_to_archive > 0,
            }
            
        except Exception as e:
            logger.error(f"Error checking data retention: {e}")
            raise

