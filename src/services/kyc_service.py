"""KYC Service - Know Your Customer compliance and verification"""

import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.models import Customer, KYCStatus
from ..observability.tracing import trace_function
from ..security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)


class KYCService:
    """Handle KYC verification and compliance checks"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.audit_logger = AuditLogger(db)
        
        # Sanctions list (simplified - in production, integrate with OFAC/EU lists)
        self.sanctions_list = [
            "sanctioned_entity_1",
            "blocked_person_2",
        ]
    
    @trace_function("verify_customer")
    async def verify_customer(
        self,
        customer_id: str,
        identity_document: Dict[str, Any],
        address_proof: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Comprehensive KYC verification
        
        Steps:
        1. Identity document verification
        2. Address verification
        3. Sanctions screening
        4. Risk assessment
        5. Update KYC status
        """
        logger.info(f"Starting KYC verification for customer {customer_id}")
        
        try:
            customer = await self._get_customer(customer_id)
            if not customer:
                raise ValueError("Customer not found")
            
            verification_results = {
                "customer_id": customer_id,
                "started_at": datetime.utcnow().isoformat(),
                "checks": [],
                "overall_status": "pending",
            }
            
            # 1. Identity Document Verification
            identity_check = await self._verify_identity_document(
                customer,
                identity_document
            )
            verification_results["checks"].append(identity_check)
            
            # 2. Address Verification
            address_check = await self._verify_address(customer, address_proof)
            verification_results["checks"].append(address_check)
            
            # 3. Sanctions Screening
            sanctions_check = await self._sanctions_screening(customer)
            verification_results["checks"].append(sanctions_check)
            
            # 4. Risk Assessment
            risk_assessment = await self._assess_customer_risk(customer)
            verification_results["checks"].append(risk_assessment)
            
            # 5. Determine overall KYC status
            all_passed = all(check["passed"] for check in verification_results["checks"])
            sanctions_failed = not sanctions_check["passed"]
            
            if sanctions_failed:
                new_status = KYCStatus.REJECTED
                verification_results["overall_status"] = "rejected"
                verification_results["reason"] = "Failed sanctions screening"
            elif all_passed:
                new_status = KYCStatus.VERIFIED
                verification_results["overall_status"] = "verified"
            else:
                new_status = KYCStatus.IN_PROGRESS
                verification_results["overall_status"] = "in_progress"
                verification_results["reason"] = "Some checks need manual review"
            
            # Update customer KYC status
            await self._update_kyc_status(customer, new_status)
            
            # Log audit event
            await self.audit_logger.log_event(
                event_type="kyc_verification",
                action="verify_customer",
                customer_id=customer_id,
                result="success",
                metadata=verification_results,
            )
            
            verification_results["completed_at"] = datetime.utcnow().isoformat()
            
            return verification_results
            
        except Exception as e:
            logger.error(f"Error in KYC verification: {e}")
            
            await self.audit_logger.log_event(
                event_type="kyc_verification",
                action="verify_customer",
                customer_id=customer_id,
                result="error",
                metadata={"error": str(e)},
            )
            
            raise
    
    async def _verify_identity_document(
        self,
        customer: Customer,
        document: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify identity document (passport, driver's license, etc.)"""
        
        # In production, integrate with ID verification services like:
        # - Jumio
        # - Onfido
        # - Stripe Identity
        # - AWS Rekognition
        
        doc_type = document.get("type", "")
        doc_number = document.get("number", "")
        expiry_date = document.get("expiry_date")
        
        checks = []
        passed = True
        
        # Check document type is valid
        valid_types = ["passport", "drivers_license", "national_id"]
        if doc_type not in valid_types:
            checks.append("Invalid document type")
            passed = False
        else:
            checks.append("Valid document type")
        
        # Check document number format
        if not doc_number or len(doc_number) < 5:
            checks.append("Invalid document number")
            passed = False
        else:
            checks.append("Valid document number format")
        
        # Check expiry date
        if expiry_date:
            try:
                expiry = datetime.fromisoformat(expiry_date)
                if expiry < datetime.utcnow():
                    checks.append("Document expired")
                    passed = False
                else:
                    checks.append("Document not expired")
            except:
                checks.append("Invalid expiry date format")
                passed = False
        
        # Check name matches
        doc_name = f"{document.get('first_name', '')} {document.get('last_name', '')}"
        customer_name = f"{customer.first_name} {customer.last_name}"
        
        if doc_name.lower().strip() == customer_name.lower().strip():
            checks.append("Name matches customer record")
        else:
            checks.append("Name mismatch - needs review")
            passed = False
        
        return {
            "check_type": "identity_document",
            "passed": passed,
            "details": checks,
            "document_type": doc_type,
        }
    
    async def _verify_address(
        self,
        customer: Customer,
        address_proof: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Verify customer address"""
        
        # In production, integrate with address verification services:
        # - USPS Address Verification
        # - Google Address Validation API
        # - Loqate/Experian
        
        doc_type = address_proof.get("type", "")
        address = address_proof.get("address", "")
        date_issued = address_proof.get("date_issued")
        
        checks = []
        passed = True
        
        # Check document type
        valid_types = ["utility_bill", "bank_statement", "lease_agreement", "government_letter"]
        if doc_type not in valid_types:
            checks.append("Invalid proof type")
            passed = False
        else:
            checks.append("Valid proof type")
        
        # Check date (should be within last 3 months)
        if date_issued:
            try:
                issued = datetime.fromisoformat(date_issued)
                if datetime.utcnow() - issued > timedelta(days=90):
                    checks.append("Document too old (>3 months)")
                    passed = False
                else:
                    checks.append("Document date acceptable")
            except:
                checks.append("Invalid date format")
                passed = False
        
        # Basic address validation
        if not address or len(address) < 10:
            checks.append("Invalid address format")
            passed = False
        else:
            checks.append("Address format acceptable")
        
        return {
            "check_type": "address_verification",
            "passed": passed,
            "details": checks,
        }
    
    async def _sanctions_screening(self, customer: Customer) -> Dict[str, Any]:
        """Screen customer against sanctions lists"""
        
        # In production, integrate with:
        # - OFAC (Office of Foreign Assets Control)
        # - EU Sanctions List
        # - UN Sanctions List
        # - Dow Jones Risk & Compliance
        # - ComplyAdvantage
        
        full_name = f"{customer.first_name} {customer.last_name}".lower()
        
        # Check against simplified sanctions list
        is_sanctioned = any(
            sanctioned.lower() in full_name
            for sanctioned in self.sanctions_list
        )
        
        passed = not is_sanctioned
        
        details = []
        if is_sanctioned:
            details.append("Customer appears on sanctions list")
        else:
            details.append("No sanctions matches found")
        
        # Update customer record
        if customer:
            customer.sanctions_check_passed = passed
            await self.db.commit()
        
        return {
            "check_type": "sanctions_screening",
            "passed": passed,
            "details": details,
        }
    
    async def _assess_customer_risk(self, customer: Customer) -> Dict[str, Any]:
        """Assess customer risk score"""
        
        risk_score = 50  # Start at medium risk
        risk_factors = []
        
        # Factor 1: Account age
        account_age_days = (datetime.utcnow() - customer.created_at).days
        if account_age_days < 30:
            risk_score += 20
            risk_factors.append("New account (<30 days)")
        elif account_age_days > 365:
            risk_score -= 10
            risk_factors.append("Established account (>1 year)")
        
        # Factor 2: Email domain
        email_domain = customer.email.split('@')[1] if '@' in customer.email else ''
        suspicious_domains = ['temp-mail.com', 'throwaway.email', '10minutemail.com']
        if email_domain in suspicious_domains:
            risk_score += 15
            risk_factors.append("Suspicious email domain")
        
        # Factor 3: Phone verification
        if not customer.phone:
            risk_score += 10
            risk_factors.append("No phone number provided")
        
        # Normalize score to 0-100
        risk_score = max(0, min(100, risk_score))
        
        # Update customer risk score
        customer.risk_score = risk_score
        await self.db.commit()
        
        # Determine if passed (risk < 70 is acceptable)
        passed = risk_score < 70
        
        return {
            "check_type": "risk_assessment",
            "passed": passed,
            "details": risk_factors,
            "risk_score": risk_score,
            "risk_level": "high" if risk_score >= 70 else "medium" if risk_score >= 40 else "low",
        }
    
    async def _get_customer(self, customer_id: str) -> Optional[Customer]:
        """Get customer by ID"""
        try:
            stmt = select(Customer).where(Customer.customer_id == UUID(customer_id))
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting customer: {e}")
            return None
    
    async def _update_kyc_status(self, customer: Customer, status: KYCStatus):
        """Update customer KYC status"""
        try:
            customer.kyc_status = status
            if status == KYCStatus.VERIFIED:
                customer.kyc_verified_at = datetime.utcnow()
            await self.db.commit()
            logger.info(f"Updated KYC status for {customer.customer_id} to {status.value}")
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error updating KYC status: {e}")
            raise
    
    @trace_function("get_kyc_status")
    async def get_kyc_status(self, customer_id: str) -> Dict[str, Any]:
        """Get current KYC status for a customer"""
        try:
            customer = await self._get_customer(customer_id)
            if not customer:
                raise ValueError("Customer not found")
            
            return {
                "customer_id": customer_id,
                "kyc_status": customer.kyc_status.value if customer.kyc_status else "pending",
                "verified_at": customer.kyc_verified_at.isoformat() if customer.kyc_verified_at else None,
                "risk_score": customer.risk_score,
                "sanctions_check_passed": customer.sanctions_check_passed,
            }
            
        except Exception as e:
            logger.error(f"Error getting KYC status: {e}")
            raise
    
    @trace_function("refresh_kyc")
    async def refresh_kyc(self, customer_id: str) -> Dict[str, Any]:
        """Refresh KYC verification (required periodically for compliance)"""
        logger.info(f"Refreshing KYC for customer {customer_id}")
        
        try:
            customer = await self._get_customer(customer_id)
            if not customer:
                raise ValueError("Customer not found")
            
            # Re-run sanctions screening
            sanctions_check = await self._sanctions_screening(customer)
            
            # Re-assess risk
            risk_assessment = await self._assess_customer_risk(customer)
            
            # Check if KYC needs to be reverified
            if customer.kyc_verified_at:
                days_since_verification = (datetime.utcnow() - customer.kyc_verified_at).days
                
                # KYC expires after 2 years (730 days)
                if days_since_verification > 730:
                    customer.kyc_status = KYCStatus.EXPIRED
                    await self.db.commit()
                    
                    return {
                        "status": "expired",
                        "message": "KYC verification expired, re-verification required",
                        "days_since_verification": days_since_verification,
                    }
            
            return {
                "status": "refreshed",
                "sanctions_check": sanctions_check,
                "risk_assessment": risk_assessment,
                "message": "KYC information refreshed successfully",
            }
            
        except Exception as e:
            logger.error(f"Error refreshing KYC: {e}")
            raise
