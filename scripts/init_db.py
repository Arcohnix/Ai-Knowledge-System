import asyncio
import sys
import time
from sqlalchemy import text
from app.core.config import settings
from app.core.database import engine, ping_database
from app.models import Base
from app.core.logging import get_logger, setup_logging

setup_logging()
logger = get_logger("scripts.init_db")


async def wait_for_database(max_retries: int = 30, retry_delay: float = 2.0) -> bool:
    """Waits for the database to become reachable."""
    logger.info(f"Waiting for database at {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}...")
    for attempt in range(1, max_retries + 1):
        healthy, latency, err = await ping_database()
        if healthy:
            logger.info(f"Database reachable (attempt {attempt}/{max_retries}, latency {latency}ms)")
            return True
        logger.warning(f"Database not ready (attempt {attempt}/{max_retries}): {err}. Retrying in {retry_delay}s...")
        await asyncio.sleep(retry_delay)
    return False


async def initialize_database() -> None:
    """Initializes database schema and pgvector extension if available."""
    db_ready = await wait_for_database(max_retries=15, retry_delay=2.0)
    if not db_ready:
        logger.error("Database connection timed out. Exiting.")
        sys.exit(1)

    async with engine.begin() as conn:
        # Enable pgvector extension if running on PostgreSQL
        if "postgresql" in settings.async_database_url:
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                logger.info("pgvector extension verified/enabled.")
            except Exception as e:
                logger.warning(f"Could not enable pgvector extension (non-critical in Phase 1): {e}")

        # Create all tables registered with Base
        logger.info("Synchronizing metadata tables...")
        await conn.run_sync(Base.metadata.create_all)

        # Ensure embedding vector column exists on existing tables (schema evolution)
        if "postgresql" in settings.async_database_url:
            try:
                await conn.execute(text(
                    f"ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS embedding vector({settings.EMBEDDING_DIMENSION});"
                ))
                logger.info("Verified/added embedding vector column on document_chunks.")
            except Exception as e:
                logger.warning(f"Embedding column verification note: {e}")

            # Create HNSW vector index for pgvector
            try:
                await conn.execute(text(
                    "CREATE INDEX IF NOT EXISTS ix_document_chunks_embedding_hnsw "
                    "ON document_chunks USING hnsw (embedding vector_cosine_ops);"
                ))
                logger.info("pgvector HNSW cosine index verified/created.")
            except Exception as e:
                logger.warning(f"HNSW index creation note: {e}")

        logger.info("Database initialization completed successfully.")


if __name__ == "__main__":
    asyncio.run(initialize_database())
