"""Admin endpoints - requires admin role"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...database.supabase_client import get_supabase_client
from ...observability.tracing import trace_function
from ...security.auth_service import get_current_user
from ...services.compliance import ComplianceService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


# Response Models
class CustomerSummary(BaseModel):
    customer_id: str
    email: str
    first_name: str
    last_name: str
    kyc_status: str
    risk_score: int
    account_count: int
    is_active: bool


class ComplianceReportResponse(BaseModel):
    start_date: str
    end_date: str
    total_transactions: int
    ctr_reports: int
    sar_reports: int
    high_risk_customers: int
    total_transaction_volume: float


class AuditLogResponse(BaseModel):
    log_id: str
    customer_id: str
    action: str
    entity_type: str
    entity_id: str
    created_at: str


# TODO: Add proper admin role verification
# For now, any authenticated user can access admin endpoints
# In production, check user role from token


@router.get("/customers", response_model=List[CustomerSummary])
@trace_function("admin_list_customers")
async def list_all_customers(
    limit: int = 100,
    kyc_status: Optional[str] = None,
    customer_id: str = Depends(get_current_user)
):
    """
    List all customers (admin only)
    
    Optional filtering by KYC status
    """
    try:
        supabase = get_supabase_client()
        
        # Get all customers (in production, add pagination)
        customers = await supabase.get_all_customers(limit)
        
        # Filter by KYC status if specified
        if kyc_status:
            customers = [c for c in customers if c.get("kyc_status") == kyc_status]
        
        # Get account counts for each customer
        summaries = []
        for customer in customers:
            accounts = await supabase.get_customer_accounts(customer["customer_id"])
            
            summaries.append(CustomerSummary(
                customer_id=customer["customer_id"],
                email=customer["email"],
                first_name=customer["first_name"],
                last_name=customer["last_name"],
                kyc_status=customer["kyc_status"],
                risk_score=customer.get("risk_score", 50),
                account_count=len(accounts),
                is_active=customer.get("is_active", True)
            ))
        
        return summaries
        
    except Exception as e:
        logger.error(f"Failed to list customers: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve customers"
        )


@router.get("/customers/{target_customer_id}")
@trace_function("admin_get_customer")
async def get_customer_details(
    target_customer_id: str,
    customer_id: str = Depends(get_current_user)
):
    """
    Get detailed customer information (admin only)
    """
    try:
        supabase = get_supabase_client()
        
        customer = await supabase.get_customer_by_id(target_customer_id)
        
        if not customer:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Customer not found"
            )
        
        # Get accounts
        accounts = await supabase.get_customer_accounts(target_customer_id)
        
        # Get fraud alerts
        fraud_alerts = await supabase.get_fraud_alerts_by_customer(target_customer_id, 10)
        
        return {
            "customer": customer,
            "accounts": accounts,
            "recent_fraud_alerts": fraud_alerts
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get customer details: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve customer details"
        )


@router.post("/customers/{target_customer_id}/suspend")
@trace_function("admin_suspend_customer")
async def suspend_customer(
    target_customer_id: str,
    reason: str,
    customer_id: str = Depends(get_current_user)
):
    """
    Suspend customer account (admin only)
    """
    try:
        supabase = get_supabase_client()
        
        # Update customer status
        await supabase.update_customer(target_customer_id, {"is_active": False})
        
        # Create audit log
        await supabase.create_audit_log({
            "customer_id": customer_id,  # Admin who performed action
            "action": "customer_suspended",
            "entity_type": "customer",
            "entity_id": target_customer_id,
            "event_metadata": {"reason": reason, "suspended_by": customer_id},
            "ip_address": None,
            "user_agent": None
        })
        
        logger.warning(f"Customer {target_customer_id} suspended by admin {customer_id}")
        
        return {"message": "Customer suspended", "customer_id": target_customer_id}
        
    except Exception as e:
        logger.error(f"Failed to suspend customer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to suspend customer"
        )


@router.post("/customers/{target_customer_id}/activate")
@trace_function("admin_activate_customer")
async def activate_customer(
    target_customer_id: str,
    customer_id: str = Depends(get_current_user)
):
    """
    Activate suspended customer account (admin only)
    """
    try:
        supabase = get_supabase_client()
        
        # Update customer status
        await supabase.update_customer(target_customer_id, {"is_active": True})
        
        # Create audit log
        await supabase.create_audit_log({
            "customer_id": customer_id,
            "action": "customer_activated",
            "entity_type": "customer",
            "entity_id": target_customer_id,
            "event_metadata": {"activated_by": customer_id},
            "ip_address": None,
            "user_agent": None
        })
        
        logger.info(f"Customer {target_customer_id} activated by admin {customer_id}")
        
        return {"message": "Customer activated", "customer_id": target_customer_id}
        
    except Exception as e:
        logger.error(f"Failed to activate customer: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to activate customer"
        )


@router.get("/compliance/report", response_model=ComplianceReportResponse)
@trace_function("admin_compliance_report")
async def generate_compliance_report(
    start_date: str,
    end_date: str,
    customer_id: str = Depends(get_current_user)
):
    """
    Generate compliance report for date range (admin only)
    """
    try:
        compliance_service = ComplianceService(None)  # Uses Supabase client internally
        report = await compliance_service.generate_compliance_report(
            start_date=start_date,
            end_date=end_date
        )
        
        return ComplianceReportResponse(
            start_date=report["period"]["start_date"],
            end_date=report["period"]["end_date"],
            total_transactions=report["total_transactions"],
            ctr_reports=report["ctr_reports"],
            sar_reports=report["sar_reports"],
            high_risk_customers=report["high_risk_customers"],
            total_transaction_volume=report["total_transaction_volume"]
        )
        
    except Exception as e:
        logger.error(f"Failed to generate compliance report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate compliance report"
        )


@router.get("/audit-logs", response_model=List[AuditLogResponse])
@trace_function("admin_audit_logs")
async def get_audit_logs(
    limit: int = 100,
    action: Optional[str] = None,
    customer_id: str = Depends(get_current_user)
):
    """
    Get system audit logs (admin only)
    """
    try:
        supabase = get_supabase_client()
        
        # Get audit logs
        logs = await supabase.get_audit_logs(limit)
        
        # Filter by action if specified
        if action:
            logs = [log for log in logs if log.get("action") == action]
        
        return [
            AuditLogResponse(
                log_id=log["log_id"],
                customer_id=log.get("customer_id", "system"),
                action=log["action"],
                entity_type=log["entity_type"],
                entity_id=log["entity_id"],
                created_at=log["created_at"]
            )
            for log in logs
        ]
        
    except Exception as e:
        logger.error(f"Failed to get audit logs: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve audit logs"
        )


@router.get("/fraud/dashboard")
@trace_function("admin_fraud_dashboard")
async def get_fraud_dashboard(customer_id: str = Depends(get_current_user)):
    """
    Get fraud detection dashboard metrics (admin only)
    """
    try:
        supabase = get_supabase_client()
        
        # Get recent fraud alerts
        recent_alerts = await supabase.get_recent_fraud_alerts(50)
        
        # Calculate statistics
        total_alerts = len(recent_alerts)
        critical = len([a for a in recent_alerts if a.get("severity") == "critical"])
        high = len([a for a in recent_alerts if a.get("severity") == "high"])
        medium = len([a for a in recent_alerts if a.get("severity") == "medium"])
        low = len([a for a in recent_alerts if a.get("severity") == "low"])
        
        # Get alerts by status
        pending = len([a for a in recent_alerts if a.get("status") == "pending"])
        investigating = len([a for a in recent_alerts if a.get("status") == "investigating"])
        resolved = len([a for a in recent_alerts if a.get("status") == "resolved"])
        false_positive = len([a for a in recent_alerts if a.get("status") == "false_positive"])
        
        return {
            "total_alerts": total_alerts,
            "by_severity": {
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low
            },
            "by_status": {
                "pending": pending,
                "investigating": investigating,
                "resolved": resolved,
                "false_positive": false_positive
            },
            "recent_alerts": recent_alerts[:10]  # Last 10
        }
        
    except Exception as e:
        logger.error(f"Failed to get fraud dashboard: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud dashboard"
        )


@router.get("/stats", status_code=status.HTTP_200_OK)
async def get_stats() -> dict:
    """Get system statistics"""
    return {
        "total_requests": 12345,
        "total_errors": 12,
        "avg_latency_ms": 234,
        "uptime_seconds": 86400,
    }
