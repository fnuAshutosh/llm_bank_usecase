"""Distributed tracing with OpenTelemetry and Jaeger"""

import logging
from functools import wraps
from typing import Any, Callable, Optional

from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from ..utils.config import settings

logger = logging.getLogger(__name__)

# Global tracer
_tracer: Optional[trace.Tracer] = None


def init_tracer(service_name: str = "banking-llm-api") -> trace.Tracer:
    """Initialize OpenTelemetry tracer with Jaeger exporter"""
    global _tracer
    
    if not settings.JAEGER_ENABLED:
        logger.info("Jaeger tracing is disabled")
        # Return a no-op tracer
        trace.set_tracer_provider(TracerProvider())
        _tracer = trace.get_tracer(__name__)
        return _tracer
    
    try:
        # Create resource
        resource = Resource(attributes={
            SERVICE_NAME: service_name,
            "environment": settings.APP_ENV,
            "version": "1.0.0",
        })
        
        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        
        # Create Jaeger exporter
        jaeger_exporter = JaegerExporter(
            agent_host_name=settings.JAEGER_AGENT_HOST,
            agent_port=settings.JAEGER_AGENT_PORT,
        )
        
        # Add span processor
        span_processor = BatchSpanProcessor(jaeger_exporter)
        tracer_provider.add_span_processor(span_processor)
        
        # Set the tracer provider
        trace.set_tracer_provider(tracer_provider)
        
        # Get tracer
        _tracer = trace.get_tracer(__name__)
        
        logger.info(
            f"Jaeger tracing initialized: "
            f"{settings.JAEGER_AGENT_HOST}:{settings.JAEGER_AGENT_PORT}"
        )
        
        return _tracer
        
    except Exception as e:
        logger.error(f"Failed to initialize Jaeger tracer: {e}")
        # Fallback to no-op tracer
        trace.set_tracer_provider(TracerProvider())
        _tracer = trace.get_tracer(__name__)
        return _tracer


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance"""
    global _tracer
    
    if _tracer is None:
        _tracer = init_tracer()
    
    return _tracer


def instrument_app(app):
    """Instrument FastAPI app and libraries"""
    if not settings.JAEGER_ENABLED:
        logger.info("Skipping instrumentation - Jaeger is disabled")
        return
    
    try:
        # Instrument FastAPI
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented for tracing")
        
        # Instrument SQLAlchemy
        SQLAlchemyInstrumentor().instrument()
        logger.info("SQLAlchemy instrumented for tracing")
        
        # Instrument Redis
        RedisInstrumentor().instrument()
        logger.info("Redis instrumented for tracing")
        
        # Instrument HTTPX
        HTTPXClientInstrumentor().instrument()
        logger.info("HTTPX instrumented for tracing")
        
    except Exception as e:
        logger.error(f"Failed to instrument app: {e}")


def create_span(name: str, attributes: Optional[dict] = None):
    """
    Create a new span context manager
    
    Usage:
        with create_span("database_query", {"table": "customers"}):
            # Your code here
            pass
    """
    tracer = get_tracer()
    span = tracer.start_span(name)
    
    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, str(value))
    
    return span


def trace_function(name: Optional[str] = None, attributes: Optional[dict] = None):
    """
    Decorator to trace a function
    
    Usage:
        @trace_function("process_payment")
        async def process_payment(amount: float):
            pass
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()
            
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = await func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            tracer = get_tracer()
            
            with tracer.start_as_current_span(span_name) as span:
                # Add attributes
                if attributes:
                    for key, value in attributes.items():
                        span.set_attribute(key, str(value))
                
                # Add function arguments as attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("status", "success")
                    return result
                except Exception as e:
                    span.set_attribute("status", "error")
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# ============================================================================
# Context propagation helpers
# ============================================================================

def get_current_span() -> Optional[trace.Span]:
    """Get the current active span"""
    return trace.get_current_span()


def add_span_attribute(key: str, value: Any):
    """Add attribute to current span"""
    span = get_current_span()
    if span:
        span.set_attribute(key, str(value))


def add_span_event(name: str, attributes: Optional[dict] = None):
    """Add event to current span"""
    span = get_current_span()
    if span:
        span.add_event(name, attributes=attributes or {})


def record_exception(exception: Exception):
    """Record exception in current span"""
    span = get_current_span()
    if span:
        span.record_exception(exception)
