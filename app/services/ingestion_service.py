from datetime import date
import hashlib
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from uuid import UUID
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.exceptions import ConflictError, NotFoundError, ValidationException
from app.core.logging import get_logger
from app.core.security import sanitize_filename, validate_file_magic
from app.ingestion.base import BaseEmbedder
from app.ingestion.chunker import SensibleTextChunker
from app.ingestion.cleaner import TextCleaner
from app.ingestion.embeddings import get_embedder
from app.ingestion.parsers.parser_factory import ParserFactory
from app.models.document import DocumentModel
from app.repositories.document_repository import DocumentRepository

logger = get_logger(__name__)


class IngestionService:
    """
    Coordinates the document ingestion pipeline:
    Validation -> Checksum / Duplicate Check -> Parsing -> Text Extraction -> Cleaning -> Chunking -> Embedding -> Persistence.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Optional[BaseEmbedder] = None,
    ):
        self.session = session
        self.repository = DocumentRepository(session)
        self.embedder = embedder or get_embedder()

    @staticmethod
    def calculate_checksum(file_bytes: bytes) -> str:
        """Computes SHA-256 checksum for byte payload."""
        return hashlib.sha256(file_bytes).hexdigest()

    async def validate_file(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Validates file presence, safe filename, size limits, format, and magic signatures."""
        if not filename or not filename.strip():
            raise ValidationException("Filename cannot be empty")

        clean_filename = sanitize_filename(filename)

        if not content or len(content) == 0:
            raise ValidationException(f"Uploaded file '{clean_filename}' is empty (0 bytes)")

        if len(content) > settings.MAX_FILE_SIZE_BYTES:
            max_mb = settings.MAX_FILE_SIZE_BYTES / (1024 * 1024)
            raise ValidationException(
                f"File '{clean_filename}' exceeds maximum allowed size of {max_mb:.1f} MB"
            )

        # Ensure a parser exists for this format
        document_type = ParserFactory.get_document_type(clean_filename, content_type)

        # Verify binary magic signature
        if not validate_file_magic(content, document_type):
            raise ValidationException(
                f"File '{clean_filename}' contains invalid file signature or corrupted binary structure for format '{document_type}'."
            )

        return clean_filename, document_type

    async def ingest_document(
        self,
        filename: str,
        content: bytes,
        content_type: Optional[str] = None,
        source: str = "upload",
        version: str = "1.0",
        status: str = "ACTIVE",
        effective_date: Optional[date] = None,
        expiration_date: Optional[date] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> DocumentModel:
        """
        Executes complete ingestion lifecycle with document lifecycle metadata.
        """
        # 1. Validation & File Sanitization
        clean_filename, doc_type = await self.validate_file(filename, content, content_type)

        # 2. Checksum calculation & Duplicate detection
        checksum = self.calculate_checksum(content)
        existing_doc = await self.repository.get_by_checksum(checksum)
        if existing_doc:
            logger.warning(
                f"Duplicate document rejected: '{clean_filename}' matches existing document '{existing_doc.filename}' (ID: {existing_doc.id})"
            )
            raise ConflictError(
                f"Duplicate document detected: '{clean_filename}' has already been ingested.",
                details={
                    "existing_document_id": str(existing_doc.id),
                    "existing_filename": existing_doc.filename,
                    "checksum": checksum,
                },
            )

        # 3. Parsing & Text Extraction
        parser = ParserFactory.get_parser(clean_filename, content_type)
        raw_documents = await parser.parse(content, clean_filename)
        if not raw_documents or not raw_documents[0].content.strip():
            raise ValidationException(
                f"Document '{clean_filename}' could not be parsed or contains no extractable text."
            )

        raw_doc = raw_documents[0]

        # 4. Text Cleaning
        cleaned_text = TextCleaner.clean(raw_doc.content)
        if not cleaned_text:
            raise ValidationException(
                f"Document '{clean_filename}' contains no usable text after normalization."
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

        # 6. Embedding Generation
        chunk_texts = [split["content"] for split in chunk_splits]
        logger.info(f"Generating embeddings for {len(chunk_texts)} chunks using {self.embedder.provider_name} ({self.embedder.model_name})...")
        chunk_embeddings = await self.embedder.embed_documents(chunk_texts)

        # 7. Metadata Assembly
        title = raw_doc.metadata.get("title") or os.path.splitext(clean_filename)[0]
        effective_str = effective_date.isoformat() if effective_date else None
        expiration_str = expiration_date.isoformat() if expiration_date else None

        doc_metadata = {
            **raw_doc.metadata,
            "version": version,
            "status": status,
            "effective_date": effective_str,
            "expiration_date": expiration_str,
            "file_size_bytes": len(content),
            "character_count": len(cleaned_text),
            "chunk_count": len(chunk_splits),
            "chunk_size_configured": chunker.chunk_size,
            "chunk_overlap_configured": chunker.chunk_overlap,
            "embedding_provider": self.embedder.provider_name,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dimension,
        }

        # Prepare chunk records with vector embeddings and lifecycle metadata
        chunks_data = [
            {
                "chunk_index": split["chunk_index"],
                "content": split["content"],
                "embedding": emb,
                "chunk_metadata": {
                    "char_count": split["char_count"],
                    "token_estimate": split["token_estimate"],
                    "start_char": split["start_char"],
                    "end_char": split["end_char"],
                    "document_title": title,
                    "filename": clean_filename,
                    "version": version,
                    "status": status,
                    "effective_date": effective_str,
                    "expiration_date": expiration_str,
                },
            }
            for split, emb in zip(chunk_splits, chunk_embeddings)
        ]

        document_fields = {
            "filename": clean_filename,
            "document_type": doc_type,
            "title": title,
            "source": source,
            "version": version,
            "status": status,
            "effective_date": effective_date,
            "expiration_date": expiration_date,
            "checksum": checksum,
            "doc_metadata": doc_metadata,
        }

        # 8. Persistence (Document + Vector-indexed Chunks)
        created_doc = await self.repository.create_with_chunks(document_fields, chunks_data)
        logger.info(
            f"Successfully ingested document '{filename}' (ID: {created_doc.id}, Version: {version}, Status: {status}, Chunks: {len(chunks_data)}, Size: {len(content)} bytes)"
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
