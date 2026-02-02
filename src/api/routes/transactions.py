"""Transaction endpoints"""

import logging
from decimal import Decimal
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...database.supabase_client import get_supabase_client, supabase_client
from ...observability.tracing import trace_function
from ...security.auth_service import get_current_user
from ...services.banking_service import BankingService
from ...services.fraud_detection import FraudDetectionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/transactions", tags=["transactions"])

def get_banking_service():
    """Dependency to get banking service instance"""
    return BankingService(db=supabase_client)

def get_fraud_detection_service():
    """Dependency to get fraud detection service instance"""
    return FraudDetectionService(db=supabase_client)


# Request/Response Models
class TransactionCreate(BaseModel):
    from_account_id: str
    to_account_id: Optional[str] = None
    transaction_type: str  # deposit, withdrawal, transfer, payment
    amount: Decimal
    description: Optional[str] = None
    merchant_name: Optional[str] = None
    merchant_category: Optional[str] = None


class TransactionResponse(BaseModel):
    transaction_id: str
    from_account_id: str
    to_account_id: Optional[str]
    transaction_type: str
    amount: Decimal
    description: Optional[str]
    status: str
    fraud_score: float
    created_at: str
@router.post("/", response_model=TransactionResponse, status_code=status.HTTP_201_CREATED)
@trace_function("create_transaction")
async def create_transaction(
    transaction_data: TransactionCreate,
    customer_id: str = Depends(get_current_user)
):
    """
    Create a new transaction
    
    Supports: deposit, withdrawal, transfer, payment
    Includes basic fraud scoring
    """
    try:
        supabase = get_supabase_client()
        
        # Verify account ownership
        from_account = await supabase.get_account_by_id(transaction_data.from_account_id)
        if not from_account or from_account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this account"
            )
        
        # Validate transaction type
        valid_types = ["deposit", "withdrawal", "transfer", "payment"]
        if transaction_data.transaction_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid transaction type. Must be one of: {', '.join(valid_types)}"
            )

        amount = float(transaction_data.amount)
        from_balance = float(from_account.get("balance", 0))

        # Handle balance updates
        if transaction_data.transaction_type == "deposit":
            new_from_balance = from_balance + amount
        elif transaction_data.transaction_type in ["withdrawal", "payment"]:
            if from_balance < amount:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Insufficient funds"
                )
            new_from_balance = from_balance - amount
        elif transaction_data.transaction_type == "transfer":
            if not transaction_data.to_account_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="to_account_id is required for transfer"
                )
            if from_balance < amount:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Insufficient funds"
                )
            to_account = await supabase.get_account_by_id(transaction_data.to_account_id)
            if not to_account:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Destination account not found"
                )
            new_from_balance = from_balance - amount
            to_balance = float(to_account.get("balance", 0))
            await supabase.update_account_balance(
                transaction_data.to_account_id,
                to_balance + amount
            )
        else:
            new_from_balance = from_balance

        await supabase.update_account_balance(
            transaction_data.from_account_id,
            new_from_balance
        )

        # Simple fraud score heuristic
        fraud_score = 0.1 if amount >= 1000 else 0.02

        # Create transaction record
        transaction = await supabase.create_transaction({
            "account_id": transaction_data.from_account_id,
            "from_account_id": transaction_data.from_account_id,
            "to_account_id": transaction_data.to_account_id,
            "transaction_type": transaction_data.transaction_type,
            "amount": amount,
            "currency": "USD",
            "status": "completed",
            "description": transaction_data.description,
            "merchant_name": transaction_data.merchant_name,
            "merchant_category": transaction_data.merchant_category,
            "fraud_score": fraud_score
        })
        
        logger.info(f"Transaction created: {transaction['transaction_id']} - {transaction_data.transaction_type} ${transaction_data.amount}")
        
        return TransactionResponse(
            transaction_id=transaction["transaction_id"],
            from_account_id=transaction["from_account_id"],
            to_account_id=transaction.get("to_account_id"),
            transaction_type=transaction["transaction_type"],
            amount=Decimal(str(transaction["amount"])),
            description=transaction.get("description"),
            status=transaction["status"],
            fraud_score=transaction.get("fraud_score", 0.0),
            created_at=transaction["created_at"]
        )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Transaction creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process transaction"
        )


@router.get("/{transaction_id}", response_model=TransactionResponse)
@trace_function("get_transaction")
async def get_transaction(
    transaction_id: str,
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    Get transaction details
    """
    try:
        supabase = get_supabase_client()
        
        transaction = await supabase.get_transaction_by_id(transaction_id)
        
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        # Verify ownership of either account
        from_account = await supabase.get_account_by_id(transaction["from_account_id"])
        
        is_authorized = False
        if from_account and from_account["customer_id"] == customer_id:
            is_authorized = True
        
        if transaction.get("to_account_id"):
            to_account = await supabase.get_account_by_id(transaction["to_account_id"])
            if to_account and to_account["customer_id"] == customer_id:
                is_authorized = True
        
        if not is_authorized:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this transaction"
            )
        
        return TransactionResponse(
            transaction_id=transaction["transaction_id"],
            from_account_id=transaction["from_account_id"],
            to_account_id=transaction.get("to_account_id"),
            transaction_type=transaction["transaction_type"],
            amount=Decimal(str(transaction["amount"])),
            description=transaction.get("description"),
            status=transaction["status"],
            fraud_score=transaction.get("fraud_score", 0.0),
            created_at=transaction["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transaction"
        )


@router.get("/{transaction_id}/fraud-analysis")
@trace_function("analyze_transaction_fraud")
async def analyze_transaction_fraud(
    transaction_id: str,
    customer_id: str = Depends(get_current_user)
):
    """
    Get detailed fraud analysis for transaction
    """
    try:
        supabase = get_supabase_client()
        
        # Get transaction
        transaction = await supabase.get_transaction_by_id(transaction_id)
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        # Verify ownership
        from_account = await supabase.get_account_by_id(transaction["from_account_id"])
        if not from_account or from_account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to view this transaction"
            )
        
        # Simple fraud analysis heuristic
        amount = float(transaction["amount"])
        risk_score = min(amount / 10000, 1.0)
        if risk_score >= 0.7:
            risk_level = "high"
        elif risk_score >= 0.4:
            risk_level = "medium"
        else:
            risk_level = "low"

        fraud_signals = []
        if amount >= 10000:
            fraud_signals.append({
                "signal_name": "large_amount",
                "description": f"High value transaction: ${amount:.2f}"
            })

        return {
            "transaction_id": transaction_id,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "fraud_signals": fraud_signals,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Fraud analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to analyze transaction"
        )


@router.post("/{transaction_id}/dispute")
@trace_function("dispute_transaction")
async def dispute_transaction(
    transaction_id: str,
    reason: str,
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    Dispute a transaction
    """
    try:
        supabase = get_supabase_client()
        
        # Get transaction
        transaction = await supabase.get_transaction_by_id(transaction_id)
        if not transaction:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction not found"
            )
        
        # Verify ownership
        from_account = await supabase.get_account_by_id(transaction["from_account_id"])
        if not from_account or from_account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to dispute this transaction"
            )
        
        # Update transaction status
        await supabase.update_transaction(transaction_id, {"status": "disputed"})
        
        # Create audit log
        await supabase.create_audit_log({
            "customer_id": customer_id,
            "action": "transaction_disputed",
            "entity_type": "transaction",
            "entity_id": transaction_id,
            "event_metadata": {"reason": reason},
            "ip_address": None,
            "user_agent": None
        })
        
        logger.info(f"Transaction {transaction_id} disputed by customer {customer_id}")
        
        return {"message": "Transaction disputed successfully", "status": "disputed"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to dispute transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to dispute transaction"
        )
