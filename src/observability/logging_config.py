"""Enhanced logging configuration with structured logging"""

import json
import logging
import logging.config
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pythonjsonlogger import jsonlogger

from ..utils.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service info
        log_record['service'] = settings.APP_NAME
        log_record['environment'] = settings.APP_ENV
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add trace ID if available (from OpenTelemetry context)
        try:
            from opentelemetry import trace
            span = trace.get_current_span()
            if span:
                ctx = span.get_span_context()
                log_record['trace_id'] = format(ctx.trace_id, '032x')
                log_record['span_id'] = format(ctx.span_id, '016x')
        except Exception:
            pass


def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Logging configuration
    config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'json': {
                '()': CustomJsonFormatter,
                'format': '%(timestamp)s %(level)s %(name)s %(message)s'
            },
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': settings.LOG_LEVEL,
                'formatter': 'json' if settings.LOG_FORMAT == 'json' else 'standard',
                'stream': sys.stdout,
            },
            'file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': settings.LOG_LEVEL,
                'formatter': 'json',
                'filename': settings.LOG_FILE,
                'maxBytes': 10485760,  # 10MB
                'backupCount': 10,
            },
            'audit_file': {
                'class': 'logging.handlers.RotatingFileHandler',
                'level': 'INFO',
                'formatter': 'json',
                'filename': settings.AUDIT_LOG_PATH,
                'maxBytes': 10485760,
                'backupCount': 100,  # Keep more audit logs
            },
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': settings.LOG_LEVEL,
                'propagate': False,
            },
            'audit': {
                'handlers': ['audit_file', 'console'],
                'level': 'INFO',
                'propagate': False,
            },
            'uvicorn': {
                'handlers': ['console'],
                'level': 'INFO',
                'propagate': False,
            },
            'sqlalchemy.engine': {
                'handlers': ['console', 'file'],
                'level': 'WARNING',  # Only show warnings/errors from SQLAlchemy
                'propagate': False,
            },
        },
    }
    
    logging.config.dictConfig(config)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized - Level: {settings.LOG_LEVEL}, Format: {settings.LOG_FORMAT}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(name)


def log_audit_event(
    event_type: str,
    action: str,
    customer_id: Optional[str] = None,
    user_id: Optional[str] = None,
    ip_address: Optional[str] = None,
    result: str = "success",
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Log an audit event
    
    Args:
        event_type: Type of event (e.g., "authentication", "transaction")
        action: Action performed (e.g., "login", "payment")
        customer_id: Customer ID if applicable
        user_id: User ID if applicable
        ip_address: IP address of the request
        result: Result of the action (success, failure, error)
        metadata: Additional metadata
    """
    audit_logger = logging.getLogger('audit')
    
    audit_data = {
        'event_type': event_type,
        'action': action,
        'customer_id': customer_id,
        'user_id': user_id,
        'ip_address': ip_address,
        'result': result,
        'metadata': metadata or {},
    }
    
    audit_logger.info(json.dumps(audit_data))


# ============================================================================
# Correlation ID middleware helper
# ============================================================================

import contextvars

correlation_id_var = contextvars.ContextVar('correlation_id', default=None)


def get_correlation_id() -> Optional[str]:
    """Get current correlation ID"""
    return correlation_id_var.get()


def set_correlation_id(correlation_id: str):
    """Set correlation ID for current context"""
    correlation_id_var.set(correlation_id)


class CorrelationIdFilter(logging.Filter):
    """Add correlation ID to log records"""
    
    def filter(self, record):
        correlation_id = get_correlation_id()
        record.correlation_id = correlation_id if correlation_id else 'N/A'
        return True
