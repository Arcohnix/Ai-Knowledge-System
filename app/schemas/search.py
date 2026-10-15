from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Schema for semantic search query request."""

    query: str = Field(..., min_length=1, max_length=1000, description="User search query")
    top_k: Optional[int] = Field(5, ge=1, le=50, description="Maximum chunks to retrieve")
    similarity_threshold: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Minimum cosine similarity score (0.0 to 1.0)"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None, description="Optional metadata filters (e.g. {'document_type': 'pdf'})"
    )


class DocumentSnippet(BaseModel):
    """Metadata of the parent document for a retrieved chunk."""

    document_id: str
    filename: str
    title: str
    document_type: str
    source: Optional[str] = "upload"
    version: Optional[str] = "1.0"


class SearchResultItem(BaseModel):
    """An individual semantic search result item."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    document: DocumentSnippet


class SearchResponse(BaseModel):
    """Schema for semantic search response."""

    query: str
    total_results: int
    results: List[SearchResultItem]
