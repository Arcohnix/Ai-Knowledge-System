from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field


class ChunkResponse(BaseModel):
    """Schema for document chunk output."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    chunk_id: UUID = Field(..., validation_alias="id", serialization_alias="chunk_id")
    document_id: UUID
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="chunk_metadata",
        serialization_alias="metadata",
    )
    created_at: datetime


class DocumentResponse(BaseModel):
    """Schema for document summary output."""

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    document_id: UUID = Field(..., validation_alias="id", serialization_alias="document_id")
    filename: str
    document_type: str
    title: str
    source: str
    version: str
    checksum: str
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias="doc_metadata",
        serialization_alias="metadata",
    )
    created_at: datetime
    updated_at: datetime
    chunk_count: Optional[int] = None


class DocumentDetailResponse(DocumentResponse):
    """Schema for detailed document output including all extracted chunks."""

    chunks: List[ChunkResponse] = Field(default_factory=list)


class DocumentListResponse(BaseModel):
    """Paginated document list response."""

    total: int
    skip: int
    limit: int
    documents: List[DocumentResponse]
