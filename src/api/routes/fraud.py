"""Fraud detection endpoints"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...database.supabase_client import get_supabase_client, supabase_client
from ...observability.tracing import trace_function
from ...security.auth_service import get_current_user
from ...services.fraud_detection import FraudDetectionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])

def get_fraud_detection_service():
    """Dependency to get fraud detection service instance"""
    return FraudDetectionService(db=supabase_client)


# Response Models
class FraudAlertResponse(BaseModel):
    alert_id: str
    customer_id: str
    account_id: Optional[str]
    transaction_id: Optional[str]
    alert_type: str
    severity: str
    status: str
    description: str
    created_at: str


class FraudStatisticsResponse(BaseModel):
    total_alerts: int
    critical_alerts: int
    high_alerts: int
    medium_alerts: int
    low_alerts: int
    total_fraud_amount: float
    alerts_by_type: dict


@router.get("/alerts", response_model=List[FraudAlertResponse])
@trace_function("get_fraud_alerts")
async def get_fraud_alerts(
    limit: int = 50,
    severity: Optional[str] = None,
    customer_id: str = Depends(get_current_user),
    fraud_detection: FraudDetectionService = Depends(get_fraud_detection_service)
):
    """
    Get fraud alerts for current user
    
    Optional filtering by severity: critical, high, medium, low
    """
    try:
        supabase = get_supabase_client()
        
        # Get alerts for customer
        alerts = await supabase.get_fraud_alerts_by_customer(customer_id, limit)
        
        # Filter by severity if specified
        if severity:
            valid_severities = ["critical", "high", "medium", "low"]
            if severity not in valid_severities:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid severity. Must be one of: {', '.join(valid_severities)}"
                )
            alerts = [a for a in alerts if a.get("severity") == severity]
        
        return [
            FraudAlertResponse(
                alert_id=alert["alert_id"],
                customer_id=alert["customer_id"],
                account_id=alert.get("account_id"),
                transaction_id=alert.get("transaction_id"),
                alert_type=alert["alert_type"],
                severity=alert["severity"],
                status=alert["status"],
                description=alert.get("description", ""),
                created_at=alert["created_at"]
            )
            for alert in alerts
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fraud alerts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud alerts"
        )


@router.get("/alerts/{alert_id}", response_model=FraudAlertResponse)
@trace_function("get_fraud_alert")
async def get_fraud_alert(
    alert_id: str,
    customer_id: str = Depends(get_current_user),
    fraud_detection: FraudDetectionService = Depends(get_fraud_detection_service)
):
    """
    Get specific fraud alert details
    """
    try:
        supabase = get_supabase_client()
        
        alert = await supabase.get_fraud_alert_by_id(alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fraud alert not found"
            )
        
        # Verify ownership
        if alert["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this alert"
            )
        
        return FraudAlertResponse(
            alert_id=alert["alert_id"],
            customer_id=alert["customer_id"],
            account_id=alert.get("account_id"),
            transaction_id=alert.get("transaction_id"),
            alert_type=alert["alert_type"],
            severity=alert["severity"],
            status=alert["status"],
            description=alert.get("description", ""),
            created_at=alert["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get fraud alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud alert"
        )


@router.patch("/alerts/{alert_id}/acknowledge")
@trace_function("acknowledge_fraud_alert")
async def acknowledge_fraud_alert(
    alert_id: str,
    customer_id: str = Depends(get_current_user),
    fraud_detection: FraudDetectionService = Depends(get_fraud_detection_service)
):
    """
    Acknowledge a fraud alert
    """
    try:
        supabase = get_supabase_client()
        
        alert = await supabase.get_fraud_alert_by_id(alert_id)
        
        if not alert:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Fraud alert not found"
            )
        
        # Verify ownership
        if alert["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this alert"
            )
        
        # Update status
        await supabase.update_fraud_alert(alert_id, {"status": "acknowledged"})
        
        logger.info(f"Fraud alert {alert_id} acknowledged by customer {customer_id}")
        
        return {"message": "Alert acknowledged", "status": "acknowledged"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to acknowledge alert"
        )


@router.get("/statistics", response_model=FraudStatisticsResponse)
@trace_function("get_fraud_statistics")
async def get_fraud_statistics(
    customer_id: str = Depends(get_current_user),
    fraud_detection: FraudDetectionService = Depends(get_fraud_detection_service)
):
    """
    Get fraud detection statistics for user
    """
    try:
        stats = await fraud_detection_service.get_fraud_statistics(customer_id)
        
        return FraudStatisticsResponse(
            total_alerts=stats["total_alerts"],
            critical_alerts=stats["critical_alerts"],
            high_alerts=stats["high_alerts"],
            medium_alerts=stats["medium_alerts"],
            low_alerts=stats["low_alerts"],
            total_fraud_amount=stats["total_fraud_amount"],
            alerts_by_type=stats["alerts_by_type"]
        )
        
    except Exception as e:
        logger.error(f"Failed to get fraud statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve fraud statistics"
        )


@router.post("/report")
@trace_function("report_fraud")
async def report_fraud(
    description: str,
    transaction_id: Optional[str] = None,
    account_id: Optional[str] = None,
    customer_id: str = Depends(get_current_user),
    fraud_detection: FraudDetectionService = Depends(get_fraud_detection_service)
):
    """
    Report suspected fraud
    """
    try:
        supabase = get_supabase_client()
        
        # Verify account ownership if provided
        if account_id:
            account = await supabase.get_account_by_id(account_id)
            if not account or account["customer_id"] != customer_id:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to access this account"
                )
        
        # Create fraud alert
        alert = await fraud_detection_service.create_fraud_alert(
            customer_id=customer_id,
            alert_type="user_reported",
            severity="high",
            description=description,
            account_id=account_id,
            transaction_id=transaction_id,
            fraud_indicators={"user_report": True}
        )
        
        logger.info(f"Fraud reported by customer {customer_id}")
        
        return {
            "message": "Fraud report submitted successfully",
            "alert_id": alert["alert_id"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to report fraud: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit fraud report"
        )
