"""Audit logging for compliance"""

import json
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class AuditLogger:
    """Audit logger for compliance"""
    
    async def log_interaction(
        self,
        customer_id: str,
        message: str,
        response: str,
        latency_ms: int,
        model: str,
        pii_detected: bool,
        conversation_id: Optional[str] = None,
    ) -> None:
        """
        Log customer interaction for audit trail
        
        This creates an immutable audit log entry for compliance.
        Logs are retained for 7 years per banking regulations.
        """
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "customer_interaction",
            "customer_id": customer_id,
            "conversation_id": conversation_id,
            "message_masked": message,  # Already masked by PII detector
            "response": response,
            "latency_ms": latency_ms,
            "model": model,
            "pii_detected": pii_detected,
        }
        
        # TODO: Write to immutable audit log storage
        # For now, log to audit logger
        audit_logger = logging.getLogger("audit")
        audit_logger.info(json.dumps(audit_entry))

    async def log_chat_interaction(
        self,
        customer_id: str,
        message: str,
        response: str,
        model: str,
        pii_detected: bool,
        conversation_id: Optional[str] = None,
        latency_ms: Optional[int] = None,
    ) -> None:
        """Compatibility wrapper for chat audit logging."""
        await self.log_interaction(
            customer_id=customer_id,
            message=message,
            response=response,
            latency_ms=latency_ms or 0,
            model=model,
            pii_detected=pii_detected,
            conversation_id=conversation_id,
        )

    async def log_escalation(
        self,
        customer_id: str,
        conversation_id: str,
        reason: str,
    ) -> None:
        """Log escalation events for compliance."""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "conversation_escalation",
            "customer_id": customer_id,
            "conversation_id": conversation_id,
            "reason": reason,
        }
        audit_logger = logging.getLogger("audit")
        audit_logger.info(json.dumps(audit_entry))


# Create global instance
audit_logger = AuditLogger()
