"""Vector Database Service - Pinecone Integration for Semantic Search"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from src.utils.config import get_settings

logger = logging.getLogger(__name__)


class VectorService:
    """Handle semantic search and embeddings with Pinecone"""
    
    def __init__(self):
        """Initialize Pinecone and embedding model"""
        self.settings = get_settings()
        
        # Initialize Pinecone
        try:
            self.pc = Pinecone(api_key=self.settings.PINECONE_API_KEY)
            self.index_name = self.settings.PINECONE_INDEX_NAME
            
            # Get or create index
            self._ensure_index_exists()
            self.index = self.pc.Index(self.index_name)
            
            logger.info(f"✓ Pinecone connected to index: {self.index_name}")
        except Exception as e:
            logger.error(f"✗ Pinecone initialization failed: {e}")
            raise
        
        # Load embedding model (cached after first load)
        try:
            self.model = SentenceTransformer(self.settings.EMBEDDING_MODEL)
            logger.info(f"✓ Embedding model loaded: {self.settings.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"✗ Failed to load embedding model: {e}")
            raise
    
    def _ensure_index_exists(self):
        """Create Pinecone index if it doesn't exist"""
        try:
            # List existing indexes
            indexes = self.pc.list_indexes()
            index_names = [idx.name for idx in indexes]
            
            if self.index_name not in index_names:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.settings.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.settings.PINECONE_ENVIRONMENT.split("-")[0:2]
                    )
                )
                logger.info(f"✓ Index created: {self.index_name}")
            else:
                logger.info(f"✓ Index already exists: {self.index_name}")
        except Exception as e:
            logger.warning(f"Could not create index (might already exist): {e}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """Convert text to embedding vector"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed text: {e}")
            raise
    
    async def store_message(
        self,
        message_id: str,
        user_message: str,
        assistant_response: str,
        user_id: str,
        session_id: str,
        intent: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Store chat message with embedding in Pinecone
        
        Args:
            message_id: Unique message identifier
            user_message: Customer's question/request
            assistant_response: LLM's response
            user_id: Customer ID
            session_id: Chat session ID
            intent: Classified banking intent
            metadata: Additional metadata
        
        Returns:
            bool: Success status
        """
        try:
            # Combine message and response for better semantic search
            combined_text = f"{user_message} {assistant_response}"
            
            # Generate embedding
            embedding = self.embed_text(combined_text).tolist()
            
            # Prepare metadata
            vector_metadata = {
                "user_id": user_id,
                "session_id": session_id,
                "intent": intent,
                "user_message": user_message[:500],  # Truncate for metadata
                "assistant_response": assistant_response[:500],
                "timestamp": datetime.utcnow().isoformat(),
                "message_type": "chat_exchange"
            }
            
            # Add custom metadata
            if metadata:
                vector_metadata.update(metadata)
            
            # Upsert to Pinecone
            self.index.upsert(
                vectors=[
                    (message_id, embedding, vector_metadata)
                ]
            )
            
            logger.info(f"✓ Message {message_id} stored in Pinecone")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            return False
    
    async def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        user_id: Optional[str] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for semantically similar messages
        
        Args:
            query: User query/message
            top_k: Number of results to return
            user_id: Filter by user (optional)
            filter_metadata: Additional metadata filters
        
        Returns:
            List of similar messages with scores
        """
        try:
            # Embed the query
            query_embedding = self.embed_text(query).tolist()
            
            # Prepare filter
            filter_dict = None
            if user_id or filter_metadata:
                filter_dict = {}
                if user_id:
                    filter_dict["user_id"] = {"$eq": user_id}
                if filter_metadata:
                    filter_dict.update(filter_metadata)
            
            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Parse results
            similar_messages = []
            for match in results.get("matches", []):
                similar_messages.append({
                    "message_id": match.get("id"),
                    "score": match.get("score"),  # Similarity 0-1
                    "user_message": match.get("metadata", {}).get("user_message"),
                    "assistant_response": match.get("metadata", {}).get("assistant_response"),
                    "intent": match.get("metadata", {}).get("intent"),
                    "timestamp": match.get("metadata", {}).get("timestamp"),
                    "metadata": match.get("metadata", {})
                })
            
            logger.info(f"✓ Found {len(similar_messages)} similar messages")
            return similar_messages
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def delete_message(self, message_id: str) -> bool:
        """Delete message embedding from Pinecone (for GDPR compliance)"""
        try:
            self.index.delete(ids=[message_id])
            logger.info(f"✓ Message {message_id} deleted from Pinecone")
            return True
        except Exception as e:
            logger.error(f"Failed to delete message: {e}")
            return False
    
    async def find_similar_intents(
        self,
        query: str,
        intent: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Find similar messages for a specific banking intent
        
        Useful for:
        - Showing customer past similar requests
        - Training data selection
        - Pattern analysis
        """
        try:
            # Search with intent filter
            results = await self.semantic_search(
                query=query,
                top_k=top_k,
                filter_metadata={"intent": {"$eq": intent}}
            )
            return results
        except Exception as e:
            logger.error(f"Failed to find similar intents: {e}")
            return []
    
    async def get_intent_distribution(self) -> Dict[str, int]:
        """Get distribution of intents in vector DB"""
        try:
            stats = self.index.describe_index_stats()
            # Note: Pinecone doesn't natively support this, would need custom implementation
            logger.info(f"Index stats: {stats}")
            return {}
        except Exception as e:
            logger.error(f"Failed to get intent distribution: {e}")
            return {}
    
    async def batch_store_messages(
        self,
        messages: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Batch store multiple messages for efficiency
        
        Args:
            messages: List of message dicts with required fields
        
        Returns:
            Tuple of (successful, failed) counts
        """
        successful = 0
        failed = 0
        
        try:
            vectors_to_upsert = []
            
            for msg in messages:
                try:
                    # Embed
                    combined = f"{msg['user_message']} {msg['assistant_response']}"
                    embedding = self.embed_text(combined).tolist()
                    
                    # Prepare metadata
                    metadata = {
                        "user_id": msg["user_id"],
                        "session_id": msg["session_id"],
                        "intent": msg.get("intent", "unknown"),
                        "user_message": msg["user_message"][:500],
                        "assistant_response": msg["assistant_response"][:500],
                        "timestamp": msg.get("timestamp", datetime.utcnow().isoformat()),
                    }
                    
                    vectors_to_upsert.append(
                        (msg["message_id"], embedding, metadata)
                    )
                    successful += 1
                except Exception as e:
                    logger.warning(f"Failed to prepare message: {e}")
                    failed += 1
            
            # Batch upsert
            if vectors_to_upsert:
                self.index.upsert(vectors=vectors_to_upsert)
                logger.info(f"✓ Batch stored {successful} messages")
            
            return successful, failed
            
        except Exception as e:
            logger.error(f"Batch store failed: {e}")
            return 0, len(messages)


# Singleton instance
_vector_service: Optional[VectorService] = None


async def get_vector_service() -> VectorService:
    """Get or create VectorService singleton"""
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
