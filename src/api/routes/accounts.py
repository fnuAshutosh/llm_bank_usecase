"""Account management endpoints"""

import logging
from decimal import Decimal
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from ...database.supabase_client import get_supabase_client, supabase_client
from ...observability.tracing import trace_function
from ...security.auth_service import get_current_user
from ...services.banking_service import BankingService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/accounts", tags=["accounts"])

def get_banking_service():
    """Dependency to get banking service instance"""
    return BankingService(db=supabase_client)


# Request/Response Models
class AccountCreate(BaseModel):
    account_type: str  # checking, savings, credit, investment
    initial_balance: Optional[Decimal] = Decimal("0.00")
    currency: str = "USD"


class AccountResponse(BaseModel):
    account_id: str
    account_number: str
    account_type: str
    balance: Decimal
    currency: str
    status: str
    created_at: str


class TransactionResponse(BaseModel):
    transaction_id: str
    transaction_type: str
    amount: Decimal
    description: Optional[str]
    status: str
    fraud_score: float
    created_at: str


@router.post("/", response_model=AccountResponse, status_code=status.HTTP_201_CREATED)
@trace_function("create_account")
async def create_account(
    account_data: AccountCreate,
    customer_id: str = Depends(get_current_user)
):
    """
    Create a new bank account
    
    Supports: checking, savings, credit, investment
    """
    try:
        # Validate account type
        valid_types = ["checking", "savings", "credit", "investment"]
        if account_data.account_type not in valid_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid account type. Must be one of: {', '.join(valid_types)}"
            )
        
        import random

        supabase = get_supabase_client()

        # Generate simple account number (10 digits)
        account_number = ''.join([str(random.randint(0, 9)) for _ in range(10)])

        # Create account via Supabase
        account = await supabase.create_account({
            "customer_id": customer_id,
            "account_number": account_number,
            "account_type": account_data.account_type,
            "balance": float(account_data.initial_balance),
            "currency": account_data.currency,
            "status": "active",
            "is_active": True
        })
        
        logger.info(f"Account created: {account['account_id']} for customer {customer_id}")
        
        return AccountResponse(
            account_id=account["account_id"],
            account_number=account["account_number"],
            account_type=account["account_type"],
            balance=Decimal(str(account["balance"])),
            currency=account["currency"],
            status=account["status"],
            created_at=account["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Account creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account"
        )


@router.get("/", response_model=List[AccountResponse])
@trace_function("list_accounts")
async def list_accounts(
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    List all accounts for current user
    """
    try:
        supabase = get_supabase_client()
        
        accounts = await supabase.get_customer_accounts(customer_id)
        
        return [
            AccountResponse(
                account_id=acc["account_id"],
                account_number=acc["account_number"],
                account_type=acc["account_type"],
                balance=Decimal(str(acc["balance"])),
                currency=acc["currency"],
                status=acc["status"],
                created_at=acc["created_at"]
            )
            for acc in accounts
        ]
        
    except Exception as e:
        logger.error(f"Failed to list accounts: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve accounts"
        )


@router.get("/{account_id}", response_model=AccountResponse)
@trace_function("get_account")
async def get_account(
    account_id: str,
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    Get specific account details
    """
    try:
        supabase = get_supabase_client()
        
        account = await supabase.get_account_by_id(account_id)
        
        if not account:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Account not found"
            )
        
        # Verify ownership
        if account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this account"
            )
        
        return AccountResponse(
            account_id=account["account_id"],
            account_number=account["account_number"],
            account_type=account["account_type"],
            balance=Decimal(str(account["balance"])),
            currency=account["currency"],
            status=account["status"],
            created_at=account["created_at"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve account"
        )


@router.get("/{account_id}/transactions", response_model=List[TransactionResponse])
@trace_function("get_account_transactions")
async def get_account_transactions(
    account_id: str,
    limit: int = 50,
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    Get transaction history for an account
    """
    try:
        supabase = get_supabase_client()
        
        # Verify account ownership
        account = await supabase.get_account_by_id(account_id)
        if not account or account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this account"
            )
        
        transactions = await supabase.get_account_transactions(account_id, limit)
        
        return [
            TransactionResponse(
                transaction_id=txn["transaction_id"],
                transaction_type=txn["transaction_type"],
                amount=Decimal(str(txn["amount"])),
                description=txn.get("description"),
                status=txn["status"],
                fraud_score=txn.get("fraud_score", 0.0),
                created_at=txn["created_at"]
            )
            for txn in transactions
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get transactions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve transactions"
        )


@router.get("/{account_id}/statement")
@trace_function("get_account_statement")
async def get_account_statement(
    account_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    Generate account statement for date range
    """
    try:
        # Verify account ownership
        supabase = get_supabase_client()
        account = await supabase.get_account_by_id(account_id)
        
        if not account or account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this account"
            )
        
        # Get statement
        statement = await banking_service.get_account_statement(
            account_id=account_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return statement
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to generate statement: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate statement"
        )


@router.patch("/{account_id}/status")
@trace_function("update_account_status")
async def update_account_status(
    account_id: str,
    new_status: str,
    customer_id: str = Depends(get_current_user),
    banking_service: BankingService = Depends(get_banking_service)
):
    """
    Update account status (active, frozen, closed)
    """
    try:
        valid_statuses = ["active", "frozen", "closed"]
        if new_status not in valid_statuses:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status. Must be one of: {', '.join(valid_statuses)}"
            )
        
        supabase = get_supabase_client()
        
        # Verify ownership
        account = await supabase.get_account_by_id(account_id)
        if not account or account["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to modify this account"
            )
        
        # Update status
        updated = await supabase.update_account(account_id, {"status": new_status})
        
        logger.info(f"Account {account_id} status changed to {new_status}")
        
        return {"message": "Account status updated", "status": new_status}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update account status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update account status"
        )
