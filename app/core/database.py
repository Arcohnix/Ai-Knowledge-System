import time
from typing import AsyncGenerator, Dict, Any, Tuple
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy declarative models."""
    pass


def create_engine() -> AsyncEngine:
    """Creates the async SQLAlchemy engine based on application settings."""
    url = settings.async_database_url
    engine_kwargs: Dict[str, Any] = {
        "echo": settings.DEBUG,
        "future": True,
    }

    # SQLite does not support standard connection pool arguments
    if not url.startswith("sqlite"):
        engine_kwargs.update({
            "pool_size": settings.DB_POOL_SIZE,
            "max_overflow": settings.DB_MAX_OVERFLOW,
            "pool_timeout": settings.DB_POOL_TIMEOUT,
            "pool_recycle": settings.DB_POOL_RECYCLE,
            "pool_pre_ping": True,
        })

    return create_async_engine(url, **engine_kwargs)


# Global engine and sessionmaker
engine: AsyncEngine = create_engine()
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency that yields an async database session per request,
    ensuring rollback on unhandled exceptions and proper closing.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def ping_database() -> Tuple[bool, float, str]:
    """
    Executes a simple query to verify database connectivity.
    Returns:
        (is_healthy: bool, latency_ms: float, error_message: str)
    """
    start_time = time.perf_counter()
    try:
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return True, latency_ms, ""
    except Exception as exc:
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        logger.error(f"Database health check failed: {exc}")
        return False, latency_ms, str(exc)


async def close_database_connections() -> None:
    """Gracefully closes all connection pools on application shutdown."""
    logger.info("Closing database engine and connection pools...")
    await engine.dispose()
    logger.info("Database engine closed.")
