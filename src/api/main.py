"""Banking LLM API - Main Application Entry Point"""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator

from src.api.routes import health, admin
from src.api.routes import chat_v2 as chat
from src.api.middleware.auth import AuthMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware
from src.api.middleware.logging_middleware import LoggingMiddleware
from src.utils.config import get_settings
from src.utils.logging import setup_logging
from src.utils.metrics import setup_metrics

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager"""
    logger.info("Starting Banking LLM API...")
    
    # Initialize metrics
    setup_metrics()
    
    # Initialize database connection pool
    # await init_database()
    
    # Initialize Redis connection
    # await init_redis()
    
    # Load ML models (if needed)
    # await load_models()
    
    logger.info("Banking LLM API started successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Banking LLM API...")
    # await cleanup_database()
    # await cleanup_redis()
    logger.info("Banking LLM API shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="Banking LLM API",
    description="Enterprise-grade LLM API for banking operations",
    version="0.1.0",
    docs_url="/docs" if settings.APP_ENV == "development" else None,
    redoc_url="/redoc" if settings.APP_ENV == "development" else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_CREDENTIALS,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Process-Time"],
)

# Add trusted host middleware (security)
if settings.APP_ENV == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.ALLOWED_HOSTS,
    )

# Add custom middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(RateLimitMiddleware)
# app.add_middleware(AuthMiddleware)  # Enable when auth is configured

# Include routers
app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(admin.router, prefix="/admin", tags=["Admin"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler"""
    logger.error(
        f"Unhandled exception: {exc}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client_host": request.client.host if request.client else None,
        },
        exc_info=True,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "request_id": request.state.request_id if hasattr(request.state, "request_id") else None,
        },
    )


@app.get("/")
async def root() -> dict:
    """Root endpoint"""
    return {
        "name": "Banking LLM API",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs" if settings.APP_ENV == "development" else "disabled",
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True,
    )
