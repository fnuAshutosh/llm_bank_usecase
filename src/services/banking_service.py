"""Banking service - Core banking operations with real database queries"""

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from ..database.models import Account, AccountType, Customer, Transaction, TransactionStatus
from ..observability.metrics import track_database_query
from ..observability.tracing import trace_function

logger = logging.getLogger(__name__)


class BankingService:
    """Handle core banking operations with real database integration"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    @trace_function("get_customer_context")
    async def get_customer_context(
        self,
        customer_id: str,
        query: str,
    ) -> Dict[str, Any]:
        """
        Get comprehensive customer context for LLM
        
        Retrieves:
        - Customer profile
        - Account balances
        - Recent transactions
        - Product holdings
        - KYC status
        - Risk score
        """
        logger.info(f"Retrieving context for customer {customer_id}")
        
        start_time = time.time()
        
        try:
            # Get customer with accounts
            customer = await self.get_customer_by_id(customer_id)
            if not customer:
                return {"error": "Customer not found"}
            
            # Get all accounts
            accounts = await self.get_customer_accounts(customer_id)
            
            # Get recent transactions (last 30 days)
            transactions = await self.get_recent_transactions(
                customer_id,
                days=30,
                limit=10
            )
            
            # Build context
            context = {
                "customer_id": str(customer.customer_id),
                "customer_name": f"{customer.first_name} {customer.last_name}",
                "email": customer.email,
                "kyc_status": customer.kyc_status.value if customer.kyc_status else "unknown",
                "risk_score": customer.risk_score,
                "is_active": customer.is_active,
                "accounts": [
                    {
                        "account_id": str(acc.account_id),
                        "account_number": acc.account_number,
                        "account_type": acc.account_type.value,
                        "balance": acc.balance,
                        "currency": acc.currency,
                        "is_active": acc.is_active,
                        "is_frozen": acc.is_frozen,
                    }
                    for acc in accounts
                ],
                "total_balance": sum(acc.balance for acc in accounts),
                "recent_transactions": [
                    {
                        "transaction_id": str(txn.transaction_id),
                        "date": txn.created_at.isoformat(),
                        "amount": txn.amount,
                        "type": txn.transaction_type.value,
                        "description": txn.description,
                        "merchant": txn.merchant,
                        "status": txn.status.value,
                    }
                    for txn in transactions
                ],
                "products": list(set(acc.account_type.value for acc in accounts)),
            }
            
            duration = time.time() - start_time
            track_database_query("select", "customers", duration, success=True)
            
            return context
            
        except Exception as e:
            duration = time.time() - start_time
            track_database_query("select", "customers", duration, success=False)
            logger.error(f"Error retrieving customer context: {e}")
            raise
    
    @trace_function("get_customer_by_id")
    async def get_customer_by_id(self, customer_id: str) -> Optional[Customer]:
        """Get customer by ID"""
        try:
            stmt = select(Customer).where(Customer.customer_id == UUID(customer_id))
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting customer: {e}")
            return None
    
    @trace_function("get_customer_by_email")
    async def get_customer_by_email(self, email: str) -> Optional[Customer]:
        """Get customer by email"""
        try:
            stmt = select(Customer).where(Customer.email == email)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting customer by email: {e}")
            return None
    
    @trace_function("get_customer_accounts")
    async def get_customer_accounts(self, customer_id: str) -> List[Account]:
        """Get all accounts for a customer"""
        try:
            stmt = (
                select(Account)
                .where(Account.customer_id == UUID(customer_id))
                .where(Account.is_active == True)
            )
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"Error getting customer accounts: {e}")
            return []
    
    @trace_function("get_account_balance")
    async def get_account_balance(self, account_id: str) -> Optional[float]:
        """Get account balance"""
        try:
            stmt = select(Account.balance).where(Account.account_id == UUID(account_id))
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"Error getting account balance: {e}")
            return None
    
    @trace_function("get_recent_transactions")
    async def get_recent_transactions(
        self,
        customer_id: str,
        days: int = 30,
        limit: int = 50
    ) -> List[Transaction]:
        """Get recent transactions for a customer"""
        try:
            # Get customer's account IDs
            accounts = await self.get_customer_accounts(customer_id)
            account_ids = [acc.account_id for acc in accounts]
            
            if not account_ids:
                return []
            
            # Get transactions from the last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = (
                select(Transaction)
                .where(Transaction.account_id.in_(account_ids))
                .where(Transaction.created_at >= cutoff_date)
                .order_by(Transaction.created_at.desc())
                .limit(limit)
            )
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
            
        except Exception as e:
            logger.error(f"Error getting recent transactions: {e}")
            return []
    
    @trace_function("create_customer")
    async def create_customer(
        self,
        email: str,
        first_name: str,
        last_name: str,
        phone: Optional[str] = None,
        date_of_birth: Optional[datetime] = None,
    ) -> Customer:
        """Create a new customer"""
        try:
            customer = Customer(
                email=email,
                first_name=first_name,
                last_name=last_name,
                phone=phone,
                date_of_birth=date_of_birth,
            )
            
            self.db.add(customer)
            await self.db.commit()
            await self.db.refresh(customer)
            
            logger.info(f"Created customer: {customer.customer_id}")
            return customer
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating customer: {e}")
            raise
    
    @trace_function("create_account")
    async def create_account(
        self,
        customer_id: str,
        account_type: AccountType,
        initial_balance: float = 0.0,
        currency: str = "USD",
    ) -> Account:
        """Create a new account for a customer"""
        try:
            # Generate unique account number
            account_number = await self._generate_account_number()
            
            account = Account(
                customer_id=UUID(customer_id),
                account_number=account_number,
                account_type=account_type,
                balance=initial_balance,
                currency=currency,
            )
            
            self.db.add(account)
            await self.db.commit()
            await self.db.refresh(account)
            
            logger.info(f"Created account: {account.account_id} for customer {customer_id}")
            return account
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error creating account: {e}")
            raise
    
    async def _generate_account_number(self) -> str:
        """Generate unique account number"""
        import random
        
        while True:
            # Generate 10-digit account number
            account_number = ''.join([str(random.randint(0, 9)) for _ in range(10)])
            
            # Check if exists
            stmt = select(Account).where(Account.account_number == account_number)
            result = await self.db.execute(stmt)
            
            if result.scalar_one_or_none() is None:
                return account_number
    
    @trace_function("process_transaction")
    async def process_transaction(
        self,
        account_id: str,
        amount: float,
        transaction_type: str,
        description: str,
        merchant: Optional[str] = None,
    ) -> Transaction:
        """Process a transaction"""
        try:
            # Get account
            stmt = select(Account).where(Account.account_id == UUID(account_id))
            result = await self.db.execute(stmt)
            account = result.scalar_one_or_none()
            
            if not account:
                raise ValueError("Account not found")
            
            if not account.is_active:
                raise ValueError("Account is not active")
            
            if account.is_frozen:
                raise ValueError("Account is frozen")
            
            # Check balance for debit transactions
            if amount < 0 and account.balance + amount < -account.overdraft_limit:
                raise ValueError("Insufficient funds")
            
            # Create transaction
            balance_before = account.balance
            balance_after = balance_before + amount
            
            transaction = Transaction(
                account_id=account.account_id,
                transaction_type=transaction_type,
                amount=amount,
                description=description,
                merchant=merchant,
                status=TransactionStatus.COMPLETED,
                balance_before=balance_before,
                balance_after=balance_after,
                completed_at=datetime.utcnow(),
            )
            
            # Update account balance
            account.balance = balance_after
            
            self.db.add(transaction)
            await self.db.commit()
            await self.db.refresh(transaction)
            
            logger.info(f"Processed transaction: {transaction.transaction_id}")
            return transaction
            
        except Exception as e:
            await self.db.rollback()
            logger.error(f"Error processing transaction: {e}")
            raise
    
    @trace_function("get_account_statement")
    async def get_account_statement(
        self,
        account_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> Dict[str, Any]:
        """Get account statement for a period"""
        try:
            # Get account
            stmt = select(Account).where(Account.account_id == UUID(account_id))
            result = await self.db.execute(stmt)
            account = result.scalar_one_or_none()
            
            if not account:
                raise ValueError("Account not found")
            
            # Get transactions in period
            stmt = (
                select(Transaction)
                .where(Transaction.account_id == UUID(account_id))
                .where(and_(
                    Transaction.created_at >= start_date,
                    Transaction.created_at <= end_date
                ))
                .order_by(Transaction.created_at.asc())
            )
            
            result = await self.db.execute(stmt)
            transactions = result.scalars().all()
            
            # Calculate statistics
            total_credits = sum(t.amount for t in transactions if t.amount > 0)
            total_debits = sum(abs(t.amount) for t in transactions if t.amount < 0)
            
            statement = {
                "account_number": account.account_number,
                "account_type": account.account_type.value,
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
                "opening_balance": transactions[0].balance_before if transactions else account.balance,
                "closing_balance": account.balance,
                "total_credits": total_credits,
                "total_debits": total_debits,
                "transaction_count": len(transactions),
                "transactions": [
                    {
                        "date": t.created_at.isoformat(),
                        "description": t.description,
                        "merchant": t.merchant,
                        "amount": t.amount,
                        "balance": t.balance_after,
                    }
                    for t in transactions
                ],
            }
            
            return statement
            
        except Exception as e:
            logger.error(f"Error generating account statement: {e}")
            raise
        return 5432.18
    
    async def get_transaction_history(
        self,
        customer_id: str,
        days: int = 30,
    ) -> list:
        """Get transaction history"""
        # TODO: Query database
        return []
    
    async def initiate_fraud_investigation(
        self,
        customer_id: str,
        transaction_id: str,
    ) -> Dict[str, Any]:
        """Initiate fraud investigation"""
        logger.info(f"Initiating fraud investigation for transaction {transaction_id}")
        # TODO: Implement fraud investigation workflow
        return {"status": "investigating", "case_id": "CASE001"}

