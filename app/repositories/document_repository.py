from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.document import DocumentChunkModel, DocumentModel
from app.repositories.base import BaseRepository


class DocumentRepository(BaseRepository[DocumentModel]):
    """Repository handling database operations for documents and their chunks."""

    def __init__(self, session: AsyncSession):
        super().__init__(DocumentModel, session)

    async def get_by_checksum(self, checksum: str) -> Optional[DocumentModel]:
        """Find an existing document matching a SHA-256 hash."""
        statement = select(DocumentModel).where(DocumentModel.checksum == checksum)
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def get_by_id_with_chunks(self, document_id: Union[UUID, str]) -> Optional[DocumentModel]:
        """Fetch a document along with all its chunks."""
        statement = (
            select(DocumentModel)
            .where(DocumentModel.id == document_id)
            .options(selectinload(DocumentModel.chunks))
        )
        result = await self.session.execute(statement)
        return result.scalars().first()

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        document_type: Optional[str] = None,
    ) -> Tuple[List[DocumentModel], int]:
        """List documents with optional type filtering and total count."""
        base_query = select(DocumentModel)
        count_query = select(func.count()).select_from(DocumentModel)

        if document_type:
            base_query = base_query.where(DocumentModel.document_type == document_type.lower())
            count_query = count_query.where(DocumentModel.document_type == document_type.lower())

        # Total count
        count_res = await self.session.execute(count_query)
        total = count_res.scalar_one()

        # Paginated items
        statement = (
            base_query.order_by(DocumentModel.created_at.desc())
            .offset(skip)
            .limit(limit)
            .options(selectinload(DocumentModel.chunks))
        )
        result = await self.session.execute(statement)
        documents = list(result.scalars().all())

        return documents, total

    async def create_with_chunks(
        self,
        document_fields: Dict[str, Any],
        chunks_data: List[Dict[str, Any]],
    ) -> DocumentModel:
        """Atomically persist a document and all associated chunks."""
        doc = DocumentModel(**document_fields)
        self.session.add(doc)
        await self.session.flush()  # Populates doc.id

        chunk_models = [
            DocumentChunkModel(
                document_id=doc.id,
                chunk_index=chunk_dict["chunk_index"],
                content=chunk_dict["content"],
                chunk_metadata=chunk_dict.get("chunk_metadata", {}),
            )
            for chunk_dict in chunks_data
        ]
        self.session.add_all(chunk_models)
        await self.session.commit()
        await self.session.refresh(doc)
        return doc
