import math
from typing import Any, Dict, List, Optional
from uuid import UUID
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.core.logging import get_logger
from app.models.document import DocumentChunkModel, DocumentModel
from app.retrieval.base import BaseVectorStore, RetrievedChunk

logger = get_logger(__name__)


def compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Calculates cosine similarity between two float vectors in Python."""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


class PGVectorStore(BaseVectorStore):
    """
    Vector Store implementation utilizing PostgreSQL + pgvector for dense vector storage
    and cosine distance index-accelerated similarity search.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    @property
    def store_name(self) -> str:
        return "pgvector"

    async def add_chunks(
        self,
        chunks: List[RetrievedChunk],
        embeddings: List[List[float]],
    ) -> List[str]:
        """Update or insert chunk embeddings."""
        chunk_ids: List[str] = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = UUID(chunk.chunk_id) if chunk.chunk_id else None
            if chunk_id:
                chunk_model = await self.session.get(DocumentChunkModel, chunk_id)
                if chunk_model:
                    chunk_model.embedding = embedding
                    chunk_ids.append(str(chunk_model.id))
        await self.session.commit()
        return chunk_ids

    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedChunk]:
        """
        Executes semantic cosine similarity search.
        Uses pgvector `<=>` cosine distance on PostgreSQL, with an in-memory cosine fallback for SQLite in test suites.
        """
        bind = self.session.bind
        dialect_name = bind.dialect.name if bind else "postgresql"

        if dialect_name == "postgresql":
            # Native PostgreSQL pgvector cosine distance: d in [0, 2], similarity = 1 - d
            distance_expr = DocumentChunkModel.embedding.cosine_distance(query_embedding).label("distance")

            statement = (
                select(DocumentChunkModel, DocumentModel, distance_expr)
                .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
                .where(DocumentChunkModel.embedding.is_not(None))
            )

            # Metadata filtering
            if filters:
                if "document_id" in filters and filters["document_id"]:
                    doc_id = filters["document_id"]
                    doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
                    statement = statement.where(DocumentChunkModel.document_id == doc_uuid)
                if "document_type" in filters and filters["document_type"]:
                    statement = statement.where(DocumentModel.document_type == filters["document_type"].lower())

            # Filter by threshold if specified (similarity = 1 - distance >= threshold => distance <= 1 - threshold)
            if similarity_threshold is not None:
                max_distance = 1.0 - float(similarity_threshold)
                statement = statement.where(distance_expr <= max_distance)

            statement = statement.order_by(distance_expr.asc()).limit(top_k)

            result = await self.session.execute(statement)
            rows = result.all()

            retrieved: List[RetrievedChunk] = []
            for chunk_row, doc_row, distance in rows:
                score = round(max(0.0, 1.0 - float(distance)), 4)
                metadata = {
                    **chunk_row.chunk_metadata,
                    "document_title": doc_row.title,
                    "filename": doc_row.filename,
                    "document_type": doc_row.document_type,
                    "source": doc_row.source,
                    "version": doc_row.version,
                }
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_row.id),
                        document_id=str(doc_row.id),
                        content=chunk_row.content,
                        score=score,
                        metadata=metadata,
                    )
                )
            return retrieved

        else:
            # Fallback for SQLite in test suites
            statement = (
                select(DocumentChunkModel, DocumentModel)
                .join(DocumentModel, DocumentChunkModel.document_id == DocumentModel.id)
                .where(DocumentChunkModel.embedding.is_not(None))
            )

            if filters:
                if "document_id" in filters and filters["document_id"]:
                    doc_id = filters["document_id"]
                    doc_uuid = UUID(doc_id) if isinstance(doc_id, str) else doc_id
                    statement = statement.where(DocumentChunkModel.document_id == doc_uuid)
                if "document_type" in filters and filters["document_type"]:
                    statement = statement.where(DocumentModel.document_type == filters["document_type"].lower())

            result = await self.session.execute(statement)
            rows = result.all()

            scored_candidates = []
            for chunk_row, doc_row in rows:
                if chunk_row.embedding:
                    score = compute_cosine_similarity(query_embedding, list(chunk_row.embedding))
                    if similarity_threshold is None or score >= similarity_threshold:
                        scored_candidates.append((score, chunk_row, doc_row))

            scored_candidates.sort(key=lambda x: x[0], reverse=True)
            top_candidates = scored_candidates[:top_k]

            retrieved = []
            for score, chunk_row, doc_row in top_candidates:
                metadata = {
                    **chunk_row.chunk_metadata,
                    "document_title": doc_row.title,
                    "filename": doc_row.filename,
                    "document_type": doc_row.document_type,
                    "source": doc_row.source,
                    "version": doc_row.version,
                }
                retrieved.append(
                    RetrievedChunk(
                        chunk_id=str(chunk_row.id),
                        document_id=str(doc_row.id),
                        content=chunk_row.content,
                        score=round(score, 4),
                        metadata=metadata,
                    )
                )
            return retrieved

    async def delete_by_document_id(self, document_id: str) -> bool:
        """Cascaded by foreign key, but provided for interface compliance."""
        doc_uuid = UUID(document_id) if isinstance(document_id, str) else document_id
        statement = delete(DocumentChunkModel).where(DocumentChunkModel.document_id == doc_uuid)
        result = await self.session.execute(statement)
        await self.session.commit()
        return result.rowcount > 0
