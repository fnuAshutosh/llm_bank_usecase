"""Vector search routes - Semantic search API endpoints"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.services.vector_service import get_vector_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v2/search", tags=["semantic-search"])


# Request/Response Models
class SemanticSearchRequest(BaseModel):
    """Semantic search query"""
    query: str = Field(..., description="Search query")
    top_k: int = Field(5, ge=1, le=20, description="Number of results")
    user_id: Optional[str] = Field(None, description="Filter by user ID")


class SimilarMessage(BaseModel):
    """Similar message result"""
    message_id: str
    score: float = Field(..., description="Similarity score 0-1")
    user_message: str
    assistant_response: str
    intent: str
    timestamp: str


class SemanticSearchResponse(BaseModel):
    """Search results"""
    query: str
    results: List[SimilarMessage]
    count: int


class SimilarIntentRequest(BaseModel):
    """Find similar intents request"""
    query: str
    intent: str
    top_k: int = Field(3, ge=1, le=20)


class IntentSearchResponse(BaseModel):
    """Intent search results"""
    intent: str
    results: List[SimilarMessage]
    count: int


@router.post(
    "/similar",
    response_model=SemanticSearchResponse,
    summary="Semantic Search",
    description="Find semantically similar past chat messages"
)
async def semantic_search(
    request: SemanticSearchRequest,
    vector_service = Depends(get_vector_service)
) -> SemanticSearchResponse:
    """
    Search for semantically similar messages
    
    Returns messages similar to the query based on meaning, not keywords.
    
    **Example:**
    ```
    Query: "How do I transfer money?"
    Results: [
        "Can I send funds to another account?",
        "Transfer to external bank",
        "Send money between accounts"
    ]
    ```
    """
    try:
        results = await vector_service.semantic_search(
            query=request.query,
            top_k=request.top_k,
            user_id=request.user_id
        )
        
        # Convert to response model
        similar_messages = [
            SimilarMessage(
                message_id=r["message_id"],
                score=r["score"],
                user_message=r["user_message"],
                assistant_response=r["assistant_response"],
                intent=r["intent"],
                timestamp=r["timestamp"]
            )
            for r in results
        ]
        
        return SemanticSearchResponse(
            query=request.query,
            results=similar_messages,
            count=len(similar_messages)
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post(
    "/by-intent",
    response_model=IntentSearchResponse,
    summary="Find Similar Intents",
    description="Find messages with specific banking intent"
)
async def find_similar_intents(
    request: SimilarIntentRequest,
    vector_service = Depends(get_vector_service)
) -> IntentSearchResponse:
    """
    Find similar messages for a specific banking intent
    
    Useful for showing customers examples of similar requests
    and for agent training.
    
    **Banking Intents:**
    - transfer_funds
    - check_balance
    - pay_bill
    - apply_loan
    - report_fraud
    - etc.
    """
    try:
        results = await vector_service.find_similar_intents(
            query=request.query,
            intent=request.intent,
            top_k=request.top_k
        )
        
        similar_messages = [
            SimilarMessage(
                message_id=r["message_id"],
                score=r["score"],
                user_message=r["user_message"],
                assistant_response=r["assistant_response"],
                intent=r["intent"],
                timestamp=r["timestamp"]
            )
            for r in results
        ]
        
        return IntentSearchResponse(
            intent=request.intent,
            results=similar_messages,
            count=len(similar_messages)
        )
        
    except Exception as e:
        logger.error(f"Intent search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Vector DB Health",
    description="Check if vector database is accessible"
)
async def vector_db_health(
    vector_service = Depends(get_vector_service)
) -> dict:
    """
    Check vector database health
    
    Returns:
    - status: "healthy" or "error"
    - index_name: Pinecone index name
    - model: Embedding model in use
    """
    try:
        # Try a simple operation
        index_stats = vector_service.index.describe_index_stats()
        
        return {
            "status": "healthy",
            "service": "pinecone",
            "index_name": vector_service.index_name,
            "embedding_model": vector_service.settings.EMBEDDING_MODEL,
            "dimension": vector_service.settings.EMBEDDING_DIMENSION,
            "vector_count": index_stats.get("total_vector_count", 0)
        }
        
    except Exception as e:
        logger.error(f"Vector DB health check failed: {e}")
        return {
            "status": "error",
            "service": "pinecone",
            "error": str(e)
        }


@router.post(
    "/test",
    summary="Test Vector DB",
    description="Test embedding and search with sample data"
)
async def test_vector_db(
    vector_service = Depends(get_vector_service)
) -> dict:
    """
    Test vector database functionality
    
    Embeds a sample banking query and searches for it.
    Useful for debugging and validation.
    """
    try:
        # Sample banking query
        test_query = "How do I transfer money to another bank account?"
        
        # Test embedding
        embedding = vector_service.embed_text(test_query)
        
        # Test search
        results = await vector_service.semantic_search(
            query=test_query,
            top_k=1
        )
        
        return {
            "status": "success",
            "test_query": test_query,
            "embedding_dim": len(embedding),
            "search_results": len(results),
            "sample_result": results[0] if results else None
        }
        
    except Exception as e:
        logger.error(f"Vector DB test failed: {e}")
        return {
            "status": "error",
            "error": str(e)
        }
