"""PII Detection Service"""

from typing import List
from pydantic import BaseModel
import logging
import re

logger = logging.getLogger(__name__)


class PIIResult(BaseModel):
    """PII detection result"""
    pii_detected: bool
    pii_types: List[str]
    masked_text: str
    confidence: float


class PIIDetector:
    """Detect and mask PII in text"""
    
    # Regex patterns for common PII
    PATTERNS = {
        'SSN': r'\b\d{3}-\d{2}-\d{4}\b',
        'ACCOUNT': r'\b\d{8,17}\b',
        'CARD': r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'PHONE': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
    }
    
    async def detect_and_mask(self, text: str) -> PIIResult:
        """
        Detect and mask PII in text
        
        Returns:
            PIIResult with masked text and detected PII types
        """
        detected_types = []
        masked_text = text
        
        for pii_type, pattern in self.PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected_types.append(pii_type)
                # Mask each match
                for match in matches:
                    mask = f"[{pii_type}_{'*' * 6}]"
                    masked_text = masked_text.replace(match, mask)
        
        pii_detected = len(detected_types) > 0
        
        if pii_detected:
            logger.warning(f"PII detected: {detected_types}")
        
        return PIIResult(
            pii_detected=pii_detected,
            pii_types=detected_types,
            masked_text=masked_text,
            confidence=1.0 if pii_detected else 0.0,
        )
