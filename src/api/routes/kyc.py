"""KYC (Know Your Customer) endpoints"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...database.supabase_client import get_supabase_client, supabase_client
from ...observability.tracing import trace_function
from ...security.auth_service import get_current_user
from ...services.kyc_service import KYCService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/kyc", tags=["kyc"])

def get_kyc_service():
    """Dependency to get KYC service instance"""
    return KYCService(db=supabase_client)


# Request/Response Models
class KYCVerificationRequest(BaseModel):
    document_type: str  # passport, drivers_license, national_id
    document_number: str
    document_issuing_country: str
    document_expiry_date: str
    address_line1: str
    address_line2: Optional[str] = None
    city: str
    state: str
    postal_code: str
    country: str
    proof_of_address_type: str  # utility_bill, bank_statement, lease_agreement


class KYCStatusResponse(BaseModel):
    customer_id: str
    kyc_status: str
    risk_score: int
    verified_at: Optional[str]
    kyc_expiry_date: Optional[str]


class SanctionsCheckResponse(BaseModel):
    is_sanctioned: bool
    match_score: float
    lists_checked: list
    details: Optional[str]


@router.post("/verify", response_model=KYCStatusResponse)
@trace_function("kyc_verify")
async def verify_kyc(
    verification_data: KYCVerificationRequest,
    customer_id: str = Depends(get_current_user),
    kyc_service: KYCService = Depends(get_kyc_service)
):
    """
    Submit KYC verification documents
    
    Verifies identity document, address, sanctions, and calculates risk score
    """
    try:
        supabase = get_supabase_client()
        
        # Get customer
        customer = await supabase.get_customer_by_id(customer_id)
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Build verification data
        identity_document = {
            "type": verification_data.document_type,
            "number": verification_data.document_number,
            "issuing_country": verification_data.document_issuing_country,
            "expiry_date": verification_data.document_expiry_date
        }
        
        address = {
            "line1": verification_data.address_line1,
            "line2": verification_data.address_line2,
            "city": verification_data.city,
            "state": verification_data.state,
            "postal_code": verification_data.postal_code,
            "country": verification_data.country
        }
        
        proof_of_address = {
            "type": verification_data.proof_of_address_type
        }
        
        # Run KYC verification
        result = await kyc_service.verify_customer(
            customer_id=customer_id,
            identity_document=identity_document,
            address=address,
            proof_of_address=proof_of_address
        )
        
        logger.info(f"KYC verification completed for customer {customer_id}: {result['kyc_status']}")
        
        return KYCStatusResponse(
            customer_id=customer_id,
            kyc_status=result["kyc_status"],
            risk_score=result["risk_score"],
            verified_at=result.get("verified_at"),
            kyc_expiry_date=result.get("kyc_expiry_date")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KYC verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KYC verification failed"
        )


@router.get("/status", response_model=KYCStatusResponse)
@trace_function("kyc_status")
async def get_kyc_status(
    customer_id: str = Depends(get_current_user),
    kyc_service: KYCService = Depends(get_kyc_service)
):
    """
    Get current KYC status
    """
    try:
        supabase = get_supabase_client()
        
        customer = await supabase.get_customer_by_id(customer_id)
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        return KYCStatusResponse(
            customer_id=customer["customer_id"],
            kyc_status=customer["kyc_status"],
            risk_score=customer.get("risk_score", 50),
            verified_at=customer.get("verified_at"),
            kyc_expiry_date=customer.get("kyc_expiry_date")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get KYC status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve KYC status"
        )


@router.post("/sanctions-check", response_model=SanctionsCheckResponse)
@trace_function("sanctions_check")
async def check_sanctions(
    customer_id: str = Depends(get_current_user),
    kyc_service: KYCService = Depends(get_kyc_service)
):
    """
    Run sanctions screening check
    
    Checks against OFAC, UN, EU sanctions lists
    """
    try:
        supabase = get_supabase_client()
        
        customer = await supabase.get_customer_by_id(customer_id)
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Run sanctions screening
        result = await kyc_service.sanctions_screening(
            first_name=customer["first_name"],
            last_name=customer["last_name"],
            date_of_birth=customer.get("date_of_birth")
        )
        
        return SanctionsCheckResponse(
            is_sanctioned=result["is_sanctioned"],
            match_score=result["match_score"],
            lists_checked=result["lists_checked"],
            details=result.get("details")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sanctions check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Sanctions screening failed"
        )


@router.post("/refresh")
@trace_function("kyc_refresh")
async def refresh_kyc(
    customer_id: str = Depends(get_current_user),
    kyc_service: KYCService = Depends(get_kyc_service)
):
    """
    Refresh KYC verification (required every 2 years)
    """
    try:
        supabase = get_supabase_client()
        
        customer = await supabase.get_customer_by_id(customer_id)
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Check if refresh is needed
        result = await kyc_service.refresh_kyc(customer_id)
        
        if result["status"] == "not_needed":
            return {
                "message": "KYC verification is still valid",
                "kyc_expiry_date": result["kyc_expiry_date"]
            }
        
        return {
            "message": "KYC refresh initiated",
            "status": result["kyc_status"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KYC refresh failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="KYC refresh failed"
        )


@router.get("/risk-assessment")
@trace_function("kyc_risk_assessment")
async def get_risk_assessment(
    customer_id: str = Depends(get_current_user),
    kyc_service: KYCService = Depends(get_kyc_service)
):
    """
    Get detailed risk assessment
    """
    try:
        supabase = get_supabase_client()
        
        customer = await supabase.get_customer_by_id(customer_id)
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Get accounts and transactions for risk assessment
        accounts = await supabase.get_customer_accounts(customer_id)
        
        # Calculate risk factors
        risk_factors = {
            "kyc_status": customer["kyc_status"],
            "risk_score": customer.get("risk_score", 50),
            "account_count": len(accounts),
            "kyc_verified": customer["kyc_status"] == "verified",
            "kyc_expired": customer.get("kyc_expiry_date") is not None
        }
        
        # Risk level based on score
        risk_score = customer.get("risk_score", 50)
        if risk_score < 30:
            risk_level = "low"
        elif risk_score < 60:
            risk_level = "medium"
        else:
            risk_level = "high"
        
        return {
            "customer_id": customer_id,
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_factors": risk_factors
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Risk assessment failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Risk assessment failed"
        )
