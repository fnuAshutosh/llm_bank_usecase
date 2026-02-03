"""Banking LLM API - Production Ready Main Application"""

import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.routes import accounts, admin, auth, chat, fraud, health, kyc, transactions
from src.observability.logging_config import setup_logging

# Setup logging first
setup_logging()
logger = logging.getLogger(__name__)

# Import search routes conditionally (requires heavy ML dependencies)
try:
    from src.api.routes import search
    SEARCH_ENABLED = True
except ImportError as e:
    logger.warning(f"Search routes disabled due to missing dependencies: {e}")
    search = None
    SEARCH_ENABLED = False
from src.observability.metrics import PrometheusMiddleware
from src.observability.tracing import init_tracer, instrument_app
from src.utils.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info("🚀 Starting Banking LLM API...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"LLM Provider: {settings.LLM_PROVIDER}")
    logger.info(f"Supabase URL: {settings.SUPABASE_URL}")
    
    # Initialize tracing
    if settings.ENABLE_TRACING:
        init_tracer(service_name="banking-llm-api")
        logger.info("✅ Distributed tracing initialized")
    
    logger.info("✅ Banking LLM API started successfully")
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down Banking LLM API...")
    logger.info("✅ Banking LLM API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Banking LLM API",
    description="Production Banking AI Assistant with LLM, Fraud Detection, KYC, and Compliance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# Add custom middleware
app.add_middleware(PrometheusMiddleware)  # Prometheus metrics
app.add_middleware(LoggingMiddleware)      # Request logging
app.add_middleware(RateLimitMiddleware)    # Rate limiting

# Instrument for tracing
if settings.ENABLE_TRACING:
    instrument_app(app)

# Include routers (MUST come before static mount)
app.include_router(health.router, tags=["Health"])
app.include_router(auth.router, tags=["Authentication"])
app.include_router(accounts.router, tags=["Accounts"])
app.include_router(transactions.router, tags=["Transactions"])
app.include_router(fraud.router, tags=["Fraud Detection"])
app.include_router(kyc.router, tags=["KYC"])
app.include_router(admin.router, tags=["Admin"])
app.include_router(chat.router, tags=["Chat"])

# Include search router only if enabled
if SEARCH_ENABLED and search:
    app.include_router(search.router, tags=["Semantic Search"])
    logger.info("✅ Semantic search routes enabled")
else:
    logger.warning("⚠️  Semantic search routes disabled")

logger.info("✅ All routes registered")

# Mount static files LAST (so API routes take precedence)
frontend_path = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")
    logger.info(f"✅ Static files mounted from {frontend_path}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler"""
    logger.error(
        f"Unhandled exception: {exc}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client": request.client.host if request.client else None,
        },
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": request.state.request_id if hasattr(request.state, "request_id") else None,
        },
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint"""
    return {
        "name": "Banking LLM API",
        "version": "1.0.0",
        "status": "operational",
        "environment": settings.ENVIRONMENT,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/info")
async def info() -> dict:
    """System information"""
    return {
        "llm_provider": settings.LLM_PROVIDER,
        "features": {
            "authentication": True,
            "fraud_detection": True,
            "kyc_verification": True,
            "compliance": True,
            "observability": settings.ENABLE_MONITORING,
            "tracing": settings.ENABLE_TRACING
        },
        "database": "Supabase PostgreSQL",
        "observability": {
            "metrics": "/metrics",
            "health": "/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="info",
        access_log=True,
    )
