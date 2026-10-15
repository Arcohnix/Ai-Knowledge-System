import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import ForeignKey, Integer, String, Text, JSON
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from app.core.config import settings
from app.core.database import Base
from app.models.base import TimestampMixin, UUIDMixin


class DocumentModel(Base, UUIDMixin, TimestampMixin):
    """
    SQLAlchemy model representing an ingested knowledge document.
    """

    __tablename__ = "documents"

    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    document_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(255), nullable=False, default="upload")
    version: Mapped[str] = mapped_column(String(50), nullable=False, default="1.0")
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    doc_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    # One-to-many relationship with chunks
    chunks: Mapped[List["DocumentChunkModel"]] = relationship(
        "DocumentChunkModel",
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="DocumentChunkModel.chunk_index",
        lazy="selectin",
    )


class DocumentChunkModel(Base, UUIDMixin, TimestampMixin):
    """
    SQLAlchemy model representing an individual chunk of a document.
    """

    __tablename__ = "document_chunks"

    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(settings.EMBEDDING_DIMENSION),
        nullable=True,
    )

    # Many-to-one relationship with document
    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="chunks",
    )
