"""Enhanced Chat Service with Vector Database Integration"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.services.banking_service import BankingService
from src.services.vector_service import get_vector_service
from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class EnhancedChatService:
    """
    Chat service combining:
    - Banking operations
    - Vector DB semantic search
    - Chat history management
    """
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.banking_service = BankingService(db)
        self.settings = get_settings()
    
    async def get_context_with_similar_queries(
        self,
        customer_id: str,
        query: str,
        top_k_similar: int = 3
    ) -> Dict[str, Any]:
        """
        Get banking context + find similar past queries for better responses
        
        Process:
        1. Get current banking context (account balance, etc)
        2. Search vector DB for semantically similar past queries
        3. Return combined context for LLM
        """
        try:
            # Get banking context
            banking_context = await self.banking_service.get_customer_context(
                customer_id=customer_id,
                query=query
            )
            
            # Get vector service
            vector_service = await get_vector_service()
            
            # Search for similar messages
            similar_queries = await vector_service.semantic_search(
                query=query,
                top_k=top_k_similar,
                user_id=customer_id
            )
            
            # Prepare similar context for LLM
            similar_context = []
            for similar in similar_queries:
                similar_context.append({
                    "past_question": similar["user_message"],
                    "past_answer": similar["assistant_response"],
                    "similarity_score": round(similar["score"], 3),
                    "intent": similar["intent"]
                })
            
            # Combine contexts
            combined_context = {
                **banking_context,
                "similar_past_queries": similar_context,
                "message_count": len(similar_queries)
            }
            
            logger.info(f"✓ Context with {len(similar_context)} similar queries retrieved")
            return combined_context
            
        except Exception as e:
            logger.error(f"Failed to get context with similar queries: {e}")
            # Fallback to banking context only
            return await self.banking_service.get_customer_context(
                customer_id=customer_id,
                query=query
            )
    
    async def store_chat_message(
        self,
        user_id: str,
        session_id: str,
        user_message: str,
        assistant_response: str,
        intent: str,
        confidence: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store chat message in both database and vector DB
        
        Args:
            user_id: Customer ID
            session_id: Chat session ID
            user_message: Customer's message
            assistant_response: LLM's response
            intent: Classified banking intent
            confidence: Intent confidence score
            metadata: Additional data
        
        Returns:
            message_id
        """
        message_id = str(uuid4())
        
        try:
            # Store in Pinecone for semantic search
            vector_service = await get_vector_service()
            
            vector_metadata = {
                "confidence": confidence,
                "source": "chat",
            }
            if metadata:
                vector_metadata.update(metadata)
            
            success = await vector_service.store_message(
                message_id=message_id,
                user_message=user_message,
                assistant_response=assistant_response,
                user_id=user_id,
                session_id=session_id,
                intent=intent,
                metadata=vector_metadata
            )
            
            if success:
                logger.info(f"✓ Message {message_id} stored in vector DB")
            else:
                logger.warning(f"✗ Failed to store in vector DB, but continuing")
            
            # TODO: Also store in PostgreSQL for full audit trail
            # await self._store_in_database(message_id, ...)
            
            return message_id
            
        except Exception as e:
            logger.error(f"Failed to store chat message: {e}")
            raise
    
    async def get_customer_chat_insights(
        self,
        customer_id: str
    ) -> Dict[str, Any]:
        """
        Get insights about customer's chat patterns
        
        Returns:
        - Most common intents
        - Typical query types
        - Past similar issues
        - Suggested assistance
        """
        try:
            vector_service = await get_vector_service()
            
            # Get all customer messages from vector DB
            # This would require custom implementation in vector DB
            
            insights = {
                "customer_id": customer_id,
                "total_messages": 0,  # Would be calculated from vector DB
                "common_intents": [],
                "insights": "Customer chat pattern analysis"
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get chat insights: {e}")
            return {}
    
    async def find_similar_customer_issues(
        self,
        query: str,
        intent: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar issues other customers faced
        
        Useful for:
        - Agent training
        - Knowledge base building
        - Pattern detection
        """
        try:
            vector_service = await get_vector_service()
            
            similar_issues = await vector_service.find_similar_intents(
                query=query,
                intent=intent,
                top_k=top_k
            )
            
            return similar_issues
            
        except Exception as e:
            logger.error(f"Failed to find similar issues: {e}")
            return []
    
    async def sync_historical_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Bulk sync historical messages to Pinecone
        
        Used for:
        - Initial data load
        - Migration
        - Backfill
        
        Args:
            messages: List of message dicts
        
        Returns:
            {'successful': count, 'failed': count}
        """
        try:
            vector_service = await get_vector_service()
            
            successful, failed = await vector_service.batch_store_messages(messages)
            
            logger.info(f"✓ Synced {successful} messages, {failed} failed")
            
            return {
                "successful": successful,
                "failed": failed,
                "total": successful + failed
            }
            
        except Exception as e:
            logger.error(f"Failed to sync historical messages: {e}")
            return {"successful": 0, "failed": len(messages)}
