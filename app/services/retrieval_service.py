from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.exceptions import ValidationException
from app.core.logging import get_logger
from app.ingestion.base import BaseEmbedder
from app.ingestion.embeddings.factory import get_embedder
from app.retrieval.base import RetrievedChunk
from app.retrieval.pgvector_store import PGVectorStore

logger = get_logger(__name__)


class RetrievalService:
    """
    Reusable retrieval service responsible for semantic vector search across document chunks.
    Encapsulates query embedding, threshold evaluation, and metadata filtering.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.session = session
        self.embedder = embedder or get_embedder()
        self.vector_store = PGVectorStore(session)

    async def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes semantic similarity search for a query string.
        """
        cleaned_query = query.strip() if query else ""
        if not cleaned_query:
            raise ValidationException("Search query cannot be empty")

        k = top_k or settings.DEFAULT_SEARCH_TOP_K
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.DEFAULT_SIMILARITY_THRESHOLD
        )

        logger.info(
            f"Executing semantic search: query='{cleaned_query[:60]}...', top_k={k}, threshold={threshold}, filters={filters}"
        )

        # 1. Generate query embedding vector
        query_vector = await self.embedder.embed_query(cleaned_query)

        # 2. Query Vector Store for nearest neighbors
        results = await self.vector_store.similarity_search(
            query_embedding=query_vector,
            top_k=k,
            filters=filters,
            similarity_threshold=threshold,
        )

        logger.info(f"Retrieved {len(results)} matching chunks for query='{cleaned_query[:40]}'")
        return results
