"""Prometheus metrics for FastAPI application"""

import logging
import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# ============================================================================
# HTTP Metrics
# ============================================================================

http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'HTTP requests in progress',
    ['method', 'endpoint']
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

# ============================================================================
# Model Inference Metrics
# ============================================================================

model_inference_total = Counter(
    'model_inference_total',
    'Total model inference requests',
    ['model', 'status']
)

model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency',
    ['model'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0]
)

model_tokens_generated = Counter(
    'model_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

model_inference_cost_usd = Counter(
    'model_inference_cost_usd_total',
    'Total inference cost in USD',
    ['model']
)

model_tokens_per_second = Gauge(
    'model_tokens_per_second',
    'Model throughput in tokens/second',
    ['model']
)

# ============================================================================
# Database Metrics
# ============================================================================

database_queries_total = Counter(
    'database_queries_total',
    'Total database queries',
    ['operation', 'table', 'status']
)

database_query_duration_seconds = Histogram(
    'database_query_duration_seconds',
    'Database query latency',
    ['operation', 'table'],
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

database_connections_active = Gauge(
    'database_connections_active',
    'Active database connections'
)

database_connection_pool_size = Gauge(
    'database_connection_pool_size',
    'Database connection pool size'
)

# ============================================================================
# Business Metrics
# ============================================================================

customer_queries_total = Counter(
    'customer_queries_total',
    'Total customer queries',
    ['query_type', 'customer_segment']
)

escalation_total = Counter(
    'escalation_total',
    'Total escalations to human agents',
    ['reason']
)

pii_detections_total = Counter(
    'pii_detections_total',
    'Total PII detections',
    ['pii_type']
)

fraud_alerts_total = Counter(
    'fraud_alerts_total',
    'Total fraud alerts',
    ['severity', 'alert_type']
)

customer_satisfaction_score = Histogram(
    'customer_satisfaction_score',
    'Customer satisfaction score (1-5)',
    buckets=[1, 2, 3, 4, 5]
)

cost_per_query_usd = Histogram(
    'cost_per_query_usd',
    'Cost per query in USD',
    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]
)

# ============================================================================
# Security Metrics
# ============================================================================

auth_attempts_total = Counter(
    'auth_attempts_total',
    'Total authentication attempts',
    ['method', 'status']
)

rate_limit_exceeded_total = Counter(
    'rate_limit_exceeded_total',
    'Total rate limit exceeded events',
    ['endpoint', 'customer_id']
)

security_events_total = Counter(
    'security_events_total',
    'Total security events',
    ['event_type', 'severity']
)

# ============================================================================
# System Metrics
# ============================================================================

app_info = Info('banking_llm_app', 'Application information')
app_info.info({
    'version': '1.0.0',
    'environment': 'development',
    'service': 'banking-llm-api'
})

active_users = Gauge(
    'active_users',
    'Currently active users'
)

cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_type']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_type']
)

cache_hit_ratio = Gauge(
    'cache_hit_ratio',
    'Cache hit ratio',
    ['cache_type']
)

# ============================================================================
# Middleware
# ============================================================================

class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track HTTP metrics"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get method and path
        method = request.method
        path = request.url.path
        
        # Skip metrics endpoint
        if path == "/metrics":
            return await call_next(request)
        
        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=path).inc()
        
        # Track request size
        if request.headers.get("content-length"):
            http_request_size_bytes.labels(
                method=method,
                endpoint=path
            ).observe(int(request.headers["content-length"]))
        
        # Time the request
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            status_code = response.status_code
            
            # Track metrics
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            # Track response size
            if "content-length" in response.headers:
                http_response_size_bytes.labels(
                    method=method,
                    endpoint=path
                ).observe(int(response.headers["content-length"]))
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            http_requests_total.labels(
                method=method,
                endpoint=path,
                status_code=500
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=path
            ).observe(duration)
            
            raise
            
        finally:
            http_requests_in_progress.labels(method=method, endpoint=path).dec()


def metrics_middleware():
    """Get the Prometheus middleware instance"""
    return PrometheusMiddleware


# ============================================================================
# Helper Functions
# ============================================================================

def track_http_request(method: str, endpoint: str, status_code: int, duration: float):
    """Track HTTP request metrics"""
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status_code=status_code
    ).inc()
    
    http_request_duration_seconds.labels(
        method=method,
        endpoint=endpoint
    ).observe(duration)


def track_model_inference(model: str, duration: float, tokens: int, cost: float, success: bool = True):
    """Track model inference metrics"""
    status = "success" if success else "failure"
    
    model_inference_total.labels(model=model, status=status).inc()
    model_inference_duration_seconds.labels(model=model).observe(duration)
    
    if success:
        model_tokens_generated.labels(model=model).inc(tokens)
        model_inference_cost_usd.labels(model=model).inc(cost)
        
        if duration > 0:
            tokens_per_sec = tokens / duration
            model_tokens_per_second.labels(model=model).set(tokens_per_sec)


def track_database_query(operation: str, table: str, duration: float, success: bool = True):
    """Track database query metrics"""
    status = "success" if success else "failure"
    
    database_queries_total.labels(
        operation=operation,
        table=table,
        status=status
    ).inc()
    
    database_query_duration_seconds.labels(
        operation=operation,
        table=table
    ).observe(duration)


def track_pii_detection(pii_type: str, count: int = 1):
    """Track PII detection metrics"""
    pii_detections_total.labels(pii_type=pii_type).inc(count)


def track_fraud_check(severity: str, alert_type: str):
    """Track fraud detection metrics"""
    fraud_alerts_total.labels(severity=severity, alert_type=alert_type).inc()


def get_metrics_handler():
    """Return metrics endpoint handler"""
    def metrics_endpoint():
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST
        )
    return metrics_endpoint
