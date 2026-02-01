"""Chat API endpoints - Version 2 with real Ollama integration"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime

from ...models.inference import OllamaInferenceService, InferenceRequest
from ...services.banking_service import BankingService
from ...security.pii_detection import PIIDetector, PIIResult
from ...security.audit_logger import AuditLogger
from ...utils.logging import logger

router = APIRouter()

# Initialize services (singleton pattern)
_inference_service = None
_banking_service = None
_pii_detector = None
_audit_logger = None

def get_services():
    """Get or create service instances"""
    global _inference_service, _banking_service, _pii_detector, _audit_logger
    
    if _inference_service is None:
        _inference_service = OllamaInferenceService()
    if _banking_service is None:
        _banking_service = BankingService()
    if _pii_detector is None:
        _pii_detector = PIIDetector()
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    
    return _inference_service, _banking_service, _pii_detector, _audit_logger


class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str = Field(..., min_length=1, max_length=4096)
    customer_id: str = Field(..., description="Customer identifier")
    session_id: Optional[str] = Field(None, description="Session ID for tracking")
    conversation_id: Optional[str] = Field(None, description="Conversation ID")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is my account balance?",
                "customer_id": "CUST-123456",
                "session_id": "session-456",
                "context": {
                    "account_number": "1234567890"
                }
            }
        }


class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str
    conversation_id: str
    session_id: str
    timestamp: datetime
    latency_ms: float
    model: str
    tokens_generated: int
    pii_detected: bool
    escalated: bool = False


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle customer chat queries
    
    Process customer messages with PII detection, banking context retrieval,
    and model inference.
    
    **Features**:
    - Automatic PII detection and masking
    - Banking context retrieval
    - Real-time model inference with Ollama
    - Compliance audit logging
    - Intelligent response generation
    """
    start_time = datetime.now()
    
    try:
        # Get service instances
        inference_service, banking_service, pii_detector, audit_logger = get_services()
        
        logger.info(f"Chat request from customer: {request.customer_id}")
        
        # 1. Detect and mask PII
        pii_result: PIIResult = pii_detector.detect_and_mask(request.message)
        masked_message = pii_result.masked_text
        
        if pii_result.has_pii:
            logger.warning(f"PII detected in message, {len(pii_result.entities)} entities masked")
        
        # 2. Get banking context
        banking_context = banking_service.get_customer_context(
            request.customer_id,
            request.context
        )
        
        # 3. Build prompt with context
        prompt = f"""You are a professional banking assistant for Bank of America. 

Customer Context:
{banking_context}

Customer Question: {masked_message}

Instructions:
- Provide accurate, helpful information
- Be professional and empathetic
- Never share sensitive information
- If unsure, offer to escalate to a human agent
- Keep responses concise (2-3 sentences)

Response:"""
        
        # 4. Generate response using Ollama
        inference_request = InferenceRequest(
            prompt=prompt,
            max_tokens=512,
            temperature=0.7,
            stream=False
        )
        
        inference_response = inference_service.generate(inference_request)
        
        # 5. Log for compliance (7-year retention)
        audit_logger.log_customer_interaction(
            customer_id=request.customer_id,
            conversation_id=request.conversation_id,
            message=masked_message,  # Never log original message with PII
            response=inference_response.text,
            latency_ms=inference_response.latency_ms,
            model=inference_response.model,
            pii_detected=pii_result.has_pii
        )
        
        # 6. Build and return response
        return ChatResponse(
            response=inference_response.text,
            conversation_id=request.conversation_id or f"conv_{int(datetime.now().timestamp())}",
            session_id=request.session_id or f"sess_{int(datetime.now().timestamp())}",
            timestamp=datetime.now(),
            latency_ms=inference_response.latency_ms,
            model=inference_response.model,
            tokens_generated=inference_response.tokens_generated,
            pii_detected=pii_result.has_pii,
            escalated=False
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred processing your request: {str(e)}"
        )


@router.post("/escalate", status_code=status.HTTP_200_OK)
def escalate_to_human(
    conversation_id: str,
    reason: str,
    customer_id: str
):
    """
    Escalate conversation to human agent
    
    Used when the AI cannot handle the request or customer requests human support.
    """
    logger.info(f"Escalation requested for conversation {conversation_id}: {reason}")
    
    # TODO: Implement actual escalation logic (webhook, queue, etc.)
    
    return {
        "status": "escalated",
        "conversation_id": conversation_id,
        "estimated_wait_time_minutes": 5,
        "message": "Your request has been escalated to our support team. An agent will assist you shortly."
    }
