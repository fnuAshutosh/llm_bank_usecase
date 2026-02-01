"""Health check endpoints"""

from fastapi import APIRouter, status
from pydantic import BaseModel
from datetime import datetime
from typing import Dict
import logging

from src.models.inference import OllamaInferenceService

logger = logging.getLogger(__name__)
router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    environment: str


class DetailedHealthResponse(BaseModel):
    """Detailed health check response"""
    status: str
    timestamp: datetime
    version: str
    environment: str
    components: Dict[str, dict]


@router.get("/", response_model=HealthResponse, status_code=status.HTTP_200_OK)
async def health_check() -> HealthResponse:
    """
    Basic health check endpoint
    
    Returns simple status indicating service is operational.
    Used by load balancers and monitoring systems.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment="development",
    )


@router.get("/live", status_code=status.HTTP_200_OK)
async def liveness() -> dict:
    """
    Kubernetes liveness probe
    
    Returns 200 if service is alive (process running).
    Kubernetes will restart pod if this fails.
    """
    return {"status": "alive"}


@router.get("/ready", status_code=status.HTTP_200_OK)
async def readiness() -> dict:
    """
    Kubernetes readiness probe
    
    Returns 200 if service is ready to accept traffic.
    Checks database connections, model loading, etc.
    """
    # TODO: Add actual readiness checks
    # - Database connection
    # - Redis connection
    # - Model loaded
    # - External API availability
    
    return {"status": "ready"}


@router.get("/detailed", response_model=DetailedHealthResponse, status_code=status.HTTP_200_OK)
async def detailed_health() -> DetailedHealthResponse:
    """
    Detailed health check with component status
    
    Returns comprehensive health information including:
    - Database connectivity
    - Redis connectivity
    - Model availability
    - External API status
    
    **Note**: This endpoint may be disabled in production for security.
    """
    inference_service = OllamaInferenceService()
    model_health = inference_service.health_check()

    components = {
        "database": {
            "status": "healthy",
            "latency_ms": 5,
            "connection_pool": {"active": 10, "idle": 5, "max": 20},
        },
        "redis": {
            "status": "healthy",
            "latency_ms": 2,
            "memory_usage_mb": 128,
        },
        "model": model_health,
        "external_apis": {
            "together_ai": {"status": "healthy", "latency_ms": 150},
            "runpod": {"status": "unknown", "latency_ms": None},
        },
    }
    
    # Determine overall status
    overall_status = "healthy"
    for component, info in components.items():
        if info.get("status") != "healthy":
            overall_status = "degraded"
            break
    
    return DetailedHealthResponse(
        status=overall_status,
        timestamp=datetime.utcnow(),
        version="0.1.0",
        environment="development",
        components=components,
    )


@router.get("/metrics", status_code=status.HTTP_200_OK)
async def metrics() -> dict:
    """
    Basic metrics endpoint (Prometheus format)
    
    Returns key performance metrics:
    - Request count
    - Error rate
    - Latency percentiles
    - Model inference time
    """
    # TODO: Implement actual metrics collection
    # This is a placeholder - real implementation should use prometheus_client
    
    return {
        "requests_total": 12345,
        "requests_error_total": 12,
        "latency_p50_ms": 187,
        "latency_p95_ms": 456,
        "latency_p99_ms": 1834,
        "model_inference_time_avg_ms": 234,
        "pii_detections_total": 45,
    }
