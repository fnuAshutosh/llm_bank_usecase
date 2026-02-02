"""Chat API endpoints - Real LLM Integration"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from ...database.supabase_client import get_supabase_client, supabase_client
from ...llm import LLMService
from ...observability.tracing import trace_function
from ...security.audit_logger import audit_logger
from ...security.auth_service import get_current_user
from ...security.pii_detection import pii_detector
from ...services.banking_service import BankingService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

def get_banking_service():
    """Dependency to get banking service instance"""
    return BankingService(db=supabase_client)

def get_llm_service():
    """Dependency to get LLM service instance"""
    return LLMService()


class ChatRequest(BaseModel):
    """Chat request schema"""
    message: str = Field(..., min_length=1, max_length=4096)
    conversation_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Chat response schema"""
    response: str
    conversation_id: str
    timestamp: str
    latency_ms: int
    model: str
    pii_detected: bool


@router.post("/", response_model=ChatResponse)
@trace_function("chat")
async def chat(
    request: ChatRequest,
    customer_id: str = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service)
):
    """
    Banking chat assistant with real LLM
    
    Uses customer context and conversation history
    """
    start_time = datetime.utcnow()
    
    try:
        supabase = get_supabase_client()
        
        # 1. PII Detection
        pii_result = await pii_detector.detect_and_mask(request.message)
        
        if pii_result.pii_detected:
            logger.warning(f"PII detected: {pii_result.pii_types}")
        
        # 2. Get basic banking context from Supabase
        accounts = await supabase.get_customer_accounts(customer_id)
        context = {
            "customer_id": customer_id,
            "accounts": accounts
        }
        
        # 3. Get conversation history
        conversation_id = request.conversation_id
        conversation_history = []
        
        if conversation_id:
            messages = await supabase.get_conversation_messages(conversation_id, limit=10)
            conversation_history = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in messages
            ]
        else:
            # Create new conversation
            conversation = await supabase.create_conversation({
                "customer_id": customer_id,
                "status": "active"
            })
            conversation_id = conversation["conversation_id"]
        
        # 4. Generate LLM response (fallback if provider unavailable)
        try:
            response_text = await llm_service.generate_banking_response(
                user_message=pii_result.masked_text,
                customer_context=context,
                conversation_history=conversation_history
            )
        except Exception as llm_error:
            logger.warning(f"LLM unavailable, returning fallback response: {llm_error}")
            response_text = (
                "I'm temporarily unable to access the language model. "
                "Please try again in a few minutes."
            )
        
        # 5. Save messages
        await supabase.create_message({
            "conversation_id": conversation_id,
            "role": "user",
            "content": pii_result.masked_text
        })
        
        await supabase.create_message({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": response_text
        })
        
        # 6. Calculate latency
        end_time = datetime.utcnow()
        latency_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # 7. Audit log
        await audit_logger.log_chat_interaction(
            customer_id=customer_id,
            conversation_id=conversation_id,
            message=pii_result.masked_text,
            response=response_text,
            model="ollama",
            pii_detected=pii_result.pii_detected,
            latency_ms=latency_ms
        )
        
        return ChatResponse(
            response=response_text,
            conversation_id=conversation_id,
            timestamp=end_time.isoformat(),
            latency_ms=latency_ms,
            model=llm_service.model,
            pii_detected=pii_result.pii_detected
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request"
        )


@router.get("/conversations")
@trace_function("list_conversations")
async def list_conversations(
    limit: int = 50,
    customer_id: str = Depends(get_current_user)
):
    """Get user's conversation history"""
    try:
        supabase = get_supabase_client()
        conversations = await supabase.get_customer_conversations(customer_id, limit)
        return {"conversations": conversations}
        
    except Exception as e:
        logger.error(f"Failed to list conversations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversations"
        )


@router.get("/conversations/{conversation_id}")
@trace_function("get_conversation")
async def get_conversation(
    conversation_id: str,
    customer_id: str = Depends(get_current_user)
):
    """Get conversation details and messages"""
    try:
        supabase = get_supabase_client()
        
        # Verify ownership
        conversation = await supabase.get_conversation_by_id(conversation_id)
        if not conversation or conversation["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this conversation"
            )
        
        messages = await supabase.get_conversation_messages(conversation_id, limit=100)
        
        return {
            "conversation": conversation,
            "messages": messages
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve conversation"
        )


@router.post("/escalate")
@trace_function("escalate")
async def escalate_to_human(
    conversation_id: str,
    reason: str,
    customer_id: str = Depends(get_current_user)
):
    """Escalate conversation to human agent"""
    try:
        supabase = get_supabase_client()
        
        # Verify ownership
        conversation = await supabase.get_conversation_by_id(conversation_id)
        if not conversation or conversation["customer_id"] != customer_id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this conversation"
            )
        
        # Update conversation status
        await supabase.update_conversation(conversation_id, {
            "status": "escalated",
            "escalation_reason": reason
        })
        
        # Log escalation
        await audit_logger.log_escalation(
            customer_id=customer_id,
            conversation_id=conversation_id,
            reason=reason
        )
        
        return {
            "status": "escalated",
            "conversation_id": conversation_id,
            "message": "Conversation escalated to human agent"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to escalate: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to escalate conversation"
        )
