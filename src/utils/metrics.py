"""Prometheus metrics setup"""

from prometheus_client import Counter, Histogram, Gauge
import time

# Request metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

# Model metrics
model_inference_duration_seconds = Histogram(
    'model_inference_duration_seconds',
    'Model inference latency',
    ['model']
)

model_tokens_generated = Counter(
    'model_tokens_generated_total',
    'Total tokens generated',
    ['model']
)

# Business metrics
customer_queries_total = Counter(
    'customer_queries_total',
    'Total customer queries',
    ['query_type']
)

pii_detections_total = Counter(
    'pii_detections_total',
    'Total PII detections',
    ['pii_type']
)

escalations_total = Counter(
    'escalations_total',
    'Total escalations to human',
    ['reason']
)

# System metrics
active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits'
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses'
)


def setup_metrics() -> None:
    """Initialize metrics collection"""
    pass  # Metrics are initialized on import
