import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import UUID
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationException
from app.core.logging import get_logger
from app.ingestion.chunker import SensibleTextChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.parsers.parser_factory import ParserFactory
from app.models.document import DocumentModel
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


class IngestionService:
    """
    Coordinates the document ingestion pipeline:
    Validation -> Checksum / Duplicate Check -> Parsing -> Text Extraction -> Cleaning -> Chunking -> Persistence.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.repository = DocumentRepository(session)

    @staticmethod
    def calculate_checksum(file_bytes: bytes) -> str:
        """Computes SHA-256 checksum for byte payload."""
        return hashlib.sha256(file_bytes).hexdigest()

    async def validate_file(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> str:
        """Validates file presence, size limits, and supported extension/type."""
        if not filename or not filename.strip():
            raise ValidationException("Filename cannot be empty")

        if not content or len(content) == 0:
            raise ValidationException(f"Uploaded file '{filename}' is empty (0 bytes)")

        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise ValidationException(
                f"File '{filename}' exceeds maximum allowed size of {max_mb:.1f} MB"
            )

        # Ensure a parser exists for this format
        document_type = ParserFactory.get_document_type(filename, content_type)
        return document_type

    async def ingest_document(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        source: str = "upload",
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> DocumentModel:
        """
        Executes complete ingestion lifecycle.
        """
        # 1. Validation
        doc_type = await self.validate_file(filename, content, content_type)

        # 2. Checksum calculation & Duplicate detection
        checksum = self.calculate_checksum(content)
        existing_doc = await self.repository.get_by_checksum(checksum)
        if existing_doc:
            logger.warning(
                f"Duplicate document rejected: '{filename}' matches existing document '{existing_doc.filename}' (ID: {existing_doc.id})"
            )
            raise ConflictError(
                f"Duplicate document detected: '{filename}' has already been ingested.",
                details={
                    "existing_document_id": str(existing_doc.id),
                    "existing_filename": existing_doc.filename,
                    "checksum": checksum,
                },
            )

        # 3. Parsing & Text Extraction
        parser = ParserFactory.get_parser(filename, content_type)
        raw_documents = await parser.parse(content, filename)
        if not raw_documents or not raw_documents[0].content.strip():
            raise ValidationException(
                f"Document '{filename}' could not be parsed or contains no extractable text."
            )

        raw_doc = raw_documents[0]

        # 4. Text Cleaning
        cleaned_text = TextCleaner.clean(raw_doc.content)
        if not cleaned_text:
            raise ValidationException(
                f"Document '{filename}' contains no usable text after normalization."
            )

        # 5. Chunking
        chunker = SensibleTextChunker(
            chunk_size=chunk_size or settings.DEFAULT_CHUNK_SIZE,
            chunk_overlap=chunk_overlap
            if chunk_overlap is not None
            else settings.DEFAULT_CHUNK_OVERLAP,
        )
        chunk_splits = chunker.split_text(cleaned_text)

        if not chunk_splits:
            raise ValidationException(
                f"Failed to generate chunks for document '{filename}'."
            )

        # 6. Metadata Assembly
        title = raw_doc.metadata.get("title") or os.path.splitext(filename)[0]
        doc_metadata = {
            **raw_doc.metadata,
            "file_size_bytes": len(content),
            "character_count": len(cleaned_text),
            "chunk_count": len(chunk_splits),
            "chunk_size_configured": chunker.chunk_size,
            "chunk_overlap_configured": chunker.chunk_overlap,
        }

        # Prepare chunk records
        chunks_data = [
            {
                "chunk_index": split["chunk_index"],
                "content": split["content"],
                "chunk_metadata": {
                    "char_count": split["char_count"],
                    "token_estimate": split["token_estimate"],
                    "start_char": split["start_char"],
                    "end_char": split["end_char"],
                    "document_title": title,
                },
            }
            for split in chunk_splits
        ]

        document_fields = {
            "filename": filename,
            "document_type": doc_type,
            "title": title,
            "source": source,
            "version": "1.0",
            "checksum": checksum,
            "doc_metadata": doc_metadata,
        }

        # 7. Persistence
        created_doc = await self.repository.create_with_chunks(document_fields, chunks_data)
        logger.info(
            f"Successfully ingested document '{filename}' (ID: {created_doc.id}, Chunks: {len(chunks_data)}, Size: {len(content)} bytes)"
        )

        return created_doc

    async def list_documents(
        self,
        skip: int = 0,
        limit: int = 100,
        document_type: Optional[str] = None,
    ) -> Tuple[List[DocumentModel], int]:
        """Fetch paginated documents list."""
        return await self.repository.list_documents(skip=skip, limit=limit, document_type=document_type)

    async def get_document(self, document_id: Union[UUID, str]) -> DocumentModel:
        """Retrieve document along with all its chunks."""
        doc = await self.repository.get_by_id_with_chunks(document_id)
        if not doc:
            raise NotFoundError(f"Document with ID '{document_id}' not found")
        return doc

    async def delete_document(self, document_id: Union[UUID, str]) -> bool:
        """Delete document and its cascaded chunks."""
        doc = await self.repository.get_by_id(document_id)
        if not doc:
            raise NotFoundError(f"Document with ID '{document_id}' not found")
        deleted = await self.repository.delete_by_id(document_id)
        logger.info(f"Deleted document '{doc.filename}' (ID: {document_id})")
        return deleted
