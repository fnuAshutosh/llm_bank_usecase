"""Observability module - Metrics, tracing, and logging"""

from .logging_config import (
    get_logger,
    log_audit_event,
    setup_logging,
)
from .metrics import (
    get_metrics_handler,
    metrics_middleware,
    track_database_query,
    track_fraud_check,
    track_http_request,
    track_model_inference,
    track_pii_detection,
)
from .tracing import (
    create_span,
    get_tracer,
    init_tracer,
    trace_function,
)

__all__ = [
    # Metrics
    "metrics_middleware",
    "track_http_request",
    "track_model_inference",
    "track_database_query",
    "track_pii_detection",
    "track_fraud_check",
    "get_metrics_handler",
    # Tracing
    "init_tracer",
    "get_tracer",
    "trace_function",
    "create_span",
    # Logging
    "setup_logging",
    "get_logger",
    "log_audit_event",
]
