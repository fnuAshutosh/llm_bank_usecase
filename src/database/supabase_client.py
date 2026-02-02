"""Supabase REST API client for Codespaces compatibility"""

import logging
from typing import Any, Dict, List, Optional

from supabase import Client, create_client

from ..utils.config import settings

logger = logging.getLogger(__name__)


class SupabaseClient:
    """
    Supabase REST API client wrapper
    
    Works in GitHub Codespaces where direct PostgreSQL connection is blocked.
    Uses Supabase's PostgREST API instead of direct database connection.
    """
    
    def __init__(self):
        self.client: Optional[Client] = None
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase client"""
        try:
            if not settings.SUPABASE_URL or not settings.SUPABASE_KEY:
                logger.warning("Supabase credentials not configured")
                return
            
            self.client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
            
            logger.info(f"Supabase client initialized: {settings.SUPABASE_URL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase client: {e}")
            raise
    
    def get_client(self) -> Client:
        """Get Supabase client instance"""
        if not self.client:
            self._initialize()
        return self.client
    
    # ========================================================================
    # Customer Operations
    # ========================================================================
    
    async def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer by ID"""
        try:
            response = self.client.table('customers').select('*').eq('customer_id', customer_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting customer: {e}")
            return None
    
    async def get_customer_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get customer by email"""
        try:
            response = self.client.table('customers').select('*').eq('email', email).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting customer by email: {e}")
            return None
    
    async def create_customer(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new customer"""
        try:
            response = self.client.table('customers').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating customer: {e}")
            raise

    async def get_all_customers(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get all customers (admin)"""
        try:
            response = self.client.table('customers').select('*').limit(limit).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting customers: {e}")
            return []
    
    async def update_customer(self, customer_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Update customer"""
        try:
            response = self.client.table('customers').update(data).eq('customer_id', customer_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error updating customer: {e}")
            raise
    
    # ========================================================================
    # Account Operations
    # ========================================================================
    
    async def get_customer_accounts(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all accounts for a customer"""
        try:
            response = self.client.table('accounts').select('*').eq('customer_id', customer_id).eq('is_active', True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting accounts: {e}")
            return []
    
    async def get_account_by_id(self, account_id: str) -> Optional[Dict[str, Any]]:
        """Get account by ID"""
        try:
            response = self.client.table('accounts').select('*').eq('account_id', account_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    async def create_account(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new account"""
        try:
            response = self.client.table('accounts').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating account: {e}")
            raise
    
    async def update_account_balance(self, account_id: str, new_balance: float) -> Dict[str, Any]:
        """Update account balance"""
        try:
            response = self.client.table('accounts').update({'balance': new_balance}).eq('account_id', account_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error updating balance: {e}")
            raise
    
    # ========================================================================
    # Transaction Operations
    # ========================================================================
    
    async def get_transactions(
        self,
        account_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get transactions for an account"""
        try:
            response = (
                self.client.table('transactions')
                .select('*')
                .eq('account_id', account_id)
                .order('created_at', desc=True)
                .limit(limit)
                .offset(offset)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error getting transactions: {e}")
            return []

    async def get_transaction_by_id(self, transaction_id: str) -> Optional[Dict[str, Any]]:
        """Get transaction by ID"""
        try:
            response = self.client.table('transactions').select('*').eq('transaction_id', transaction_id).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error getting transaction: {e}")
            return None
    
    async def create_transaction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new transaction"""
        try:
            response = self.client.table('transactions').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating transaction: {e}")
            raise
    
    # ========================================================================
    # Conversation & Message Operations
    # ========================================================================
    
    async def create_conversation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new conversation"""
        try:
            response = self.client.table('conversations').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating conversation: {e}")
            raise
    
    async def create_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new message"""
        try:
            response = self.client.table('messages').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating message: {e}")
            raise
    
    async def get_conversation_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        """Get all messages in a conversation"""
        try:
            response = (
                self.client.table('messages')
                .select('*')
                .eq('conversation_id', conversation_id)
                .order('timestamp', desc=False)
                .execute()
            )
            return response.data
        except Exception as e:
            logger.error(f"Error getting messages: {e}")
            return []
    
    # ========================================================================
    # Audit Log Operations
    # ========================================================================
    
    async def create_audit_log(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create audit log entry"""
        try:
            response = self.client.table('audit_logs').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating audit log: {e}")
            raise
    
    # ========================================================================
    # Fraud Alert Operations
    # ========================================================================
    
    async def create_fraud_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create fraud alert"""
        try:
            response = self.client.table('fraud_alerts').insert(data).execute()
            return response.data[0] if response.data else None
        except Exception as e:
            logger.error(f"Error creating fraud alert: {e}")
            raise
    
    async def get_fraud_alerts(
        self,
        customer_id: str,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get fraud alerts for customer"""
        try:
            query = self.client.table('fraud_alerts').select('*').eq('customer_id', customer_id)
            
            if status:
                query = query.eq('status', status)
            
            response = query.order('created_at', desc=True).execute()
            return response.data
        except Exception as e:
            logger.error(f"Error getting fraud alerts: {e}")
            return []


# Create module-level instance
supabase_client = SupabaseClient()

# Create instance only when needed, not at module level
def get_supabase_client() -> SupabaseClient:
    """Get Supabase client instance"""
    return supabase_client
