"""Database connection management - Supabase PostgreSQL"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from ..utils.config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()

# Database engine
engine = None
AsyncSessionLocal = None


def get_database_url() -> str:
    """Construct database URL from Supabase settings"""
    # Supabase connection format
    # postgresql://postgres:[PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres
    
    if not settings.SUPABASE_URL or not settings.SUPABASE_DB_PASSWORD:
        logger.warning("Supabase credentials not configured, using default PostgreSQL")
        return "postgresql+asyncpg://postgres:postgres@localhost:5432/banking_llm"
    
    # Extract project ref from Supabase URL
    project_ref = settings.SUPABASE_URL.replace("https://", "").replace(".supabase.co", "")
    
    db_url = (
        f"postgresql+asyncpg://postgres:{settings.SUPABASE_DB_PASSWORD}"
        f"@db.{project_ref}.supabase.co:5432/postgres"
    )
    
    return db_url


async def init_db() -> None:
    """Initialize database connection"""
    global engine, AsyncSessionLocal
    
    database_url = get_database_url()
    logger.info(f"Initializing database connection to: {database_url.split('@')[1]}")
    
    # Create async engine
    engine = create_async_engine(
        database_url,
        echo=settings.DEBUG,
        poolclass=NullPool,  # Supabase handles connection pooling
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "application_name": "banking_llm_api",
                "jit": "off",  # Optimize for Supabase
            },
            "command_timeout": 60,
            "timeout": 30,
        },
    )
    
    # Create session factory
    AsyncSessionLocal = async_sessionmaker(
        engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    
    # Import models to register them with Base
    from . import models  # noqa: F401
    
    # Create tables (in production, use migrations)
    async with engine.begin() as conn:
        # await conn.run_sync(Base.metadata.drop_all)  # Uncomment for fresh start
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("Database initialized successfully")


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency to get database session
    
    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            result = await db.execute(select(User))
            return result.scalars().all()
    """
    if AsyncSessionLocal is None:
        await init_db()
    
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            await session.close()


async def close_db() -> None:
    """Close database connection"""
    global engine
    
    if engine:
        await engine.dispose()
        logger.info("Database connection closed")
