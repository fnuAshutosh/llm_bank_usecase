"""Chat API endpoints"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
import logging

from src.models.inference import get_inference_service, InferenceService, InferenceRequest
from src.services.banking_service import BankingService
from src.security.pii_detection import PIIDetector
from src.security.audit_logger import AuditLogger

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str = Field(..., min_length=1, max_length=4096, description="User message")
    customer_id: str = Field(..., description="Customer identifier")
    session_id: Optional[str] = Field(None, description="Session identifier for context")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is my account balance?",
                "customer_id": "CUST001",
                "session_id": "sess_12345",
            }
        }


class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str = Field(..., description="Assistant response")
    conversation_id: str = Field(..., description="Conversation identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: int = Field(..., description="Response latency in milliseconds")
    model: str = Field(..., description="Model used for inference")
    tokens_generated: int = Field(..., description="Number of tokens generated")
    pii_detected: bool = Field(default=False, description="Whether PII was detected")
    escalated: bool = Field(default=False, description="Whether escalated to human")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "Your current account balance is $5,432.18",
                "conversation_id": "conv_12345",
                "session_id": "sess_12345",
                "timestamp": "2026-02-01T10:30:45.123Z",
                "latency_ms": 234,
                "model": "llama2-7b",
                "tokens_generated": 15,
                "pii_detected": False,
                "escalated": False,
            }
        }


@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(
    request: ChatRequest,
    inference_service: InferenceService = Depends(get_inference_service),
) -> ChatResponse:
    """
    Handle customer chat queries
    
    Process customer messages with PII detection, banking context retrieval,
    and model inference. Automatically escalates to human agents when needed.
    
    **Use Cases**:
    - Account balance inquiries
    - Transaction history
    - Fraud detection
    - Loan applications
    - Card management
    
    **Security**:
    - PII automatically detected and masked in logs
    - All requests logged for compliance
    - Rate limited per customer
    """
    start_time = datetime.utcnow()
    
    try:
        logger.info(
            f"Chat request received",
            extra={
                "customer_id": request.customer_id,
                "session_id": request.session_id,
                "message_length": len(request.message),
            }
        )
        
        # 1. PII Detection
        pii_detector = PIIDetector()
        pii_result = await pii_detector.detect_and_mask(request.message)
        
        if pii_result.pii_detected:
            logger.warning(
                f"PII detected in customer message",
                extra={
                    "customer_id": request.customer_id,
                    "pii_types": pii_result.pii_types,
                }
            )
        
        # 2. Get banking context (account info, transaction history, etc.)
        banking_service = BankingService()
        context = await banking_service.get_customer_context(
            customer_id=request.customer_id,
            query=request.message
        )
        
        # 3. Model inference
        response = await inference_service.generate_response(
            message=pii_result.masked_text,
            context=context,
            customer_id=request.customer_id,
        )
        
        # 4. Calculate latency
        end_time = datetime.utcnow()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # 5. Audit logging
        audit_logger = AuditLogger()
        await audit_logger.log_interaction(
            customer_id=request.customer_id,
            message=pii_result.masked_text,  # Never log original with PII
            response=response.text,
            latency_ms=latency_ms,
            model=response.model,
            pii_detected=pii_result.pii_detected,
        )
        
        # 6. Build response
        chat_response = ChatResponse(
            response=response.text,
            conversation_id=request.conversation_id or response.conversation_id,
            session_id=request.session_id or response.session_id,
            timestamp=end_time,
            latency_ms=latency_ms,
            model=response.model,
            tokens_generated=response.tokens_generated,
            pii_detected=pii_result.pii_detected,
            escalated=response.escalated,
        )
        
        logger.info(
            f"Chat response generated successfully",
            extra={
                "customer_id": request.customer_id,
                "latency_ms": latency_ms,
                "tokens_generated": response.tokens_generated,
            }
        )
        
        return chat_response
        
    except Exception as e:
        logger.error(
            f"Error processing chat request: {e}",
            extra={"customer_id": request.customer_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request. Please try again.",
        )


@router.get("/conversations/{customer_id}", status_code=status.HTTP_200_OK)
        from src.models.inference import get_inference_service, InferenceService, InferenceRequest
    customer_id: str,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    """
    Get conversation history for a customer
    
    **Parameters**:
    - customer_id: Customer identifier
    - limit: Maximum number of conversations to return (default: 10)
    - offset: Number of conversations to skip (default: 0)
    """
    try:
        # TODO: Implement conversation retrieval from database
        return {
            "customer_id": customer_id,
            "conversations": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
        }
    except Exception as e:
        logger.error(f"Error retrieving conversations: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations",
        )


@router.post("/escalate", status_code=status.HTTP_200_OK)
async def escalate_to_human(
    conversation_id: str,
    reason: str,
    customer_id: str,
) -> dict:
    """
    Escalate conversation to human agent
    
    **Use Cases**:
    - Complex queries requiring human judgment
    - Sensitive financial transactions
    - Customer explicitly requests human agent
    - Model confidence below threshold
    """
    try:
        logger.info(
            f"Escalation requested",
            extra={
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "reason": reason,
            }
        )
        
        # TODO: Implement escalation logic (create ticket, notify agents, etc.)
        
        return {
            "status": "escalated",
            "conversation_id": conversation_id,
            "ticket_id": "TKT001",  # Placeholder
            "estimated_wait_time_minutes": 5,
            "message": "Your query has been escalated to a human agent. You will receive a response within 5 minutes.",
        }
        
    except Exception as e:
        logger.error(f"Error escalating conversation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to escalate conversation",
        )
