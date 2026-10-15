from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Payload for asking a question against the knowledge system."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to answer using enterprise knowledge",
        examples=["What are the rules regarding code reviews and pair programming?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional maximum number of evidence chunks to retrieve",
    )
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity score",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters (e.g. document_id, document_type)",
    )


class AskSourceDocument(BaseModel):
    """Metadata describing the originating source document."""

    document_id: str = Field(..., description="Unique document UUID")
    title: Optional[str] = Field(None, description="Title of the document")
    filename: str = Field(..., description="Original filename")
    document_type: str = Field(..., description="Document type (pdf, docx, etc.)")
    source: Optional[str] = Field("upload", description="Source channel")
    version: Optional[str] = Field("1.0", description="Document version")


class AskSourceChunk(BaseModel):
    """Specific chunk used as retrieved evidence."""

    chunk_id: str = Field(..., description="Unique chunk UUID")
    chunk_index: int = Field(..., description="0-indexed position within document")
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Chunk-level metadata"
    )


class AskSource(BaseModel):
    """Attributed source citation linking chunk evidence to its parent document."""

    document: AskSourceDocument
    chunk: AskSourceChunk
    relevance_score: float = Field(
        ..., description="Cosine similarity score of the retrieved chunk"
    )


class AskResponse(BaseModel):
    """Unified response containing the synthesized answer and source citations."""

    question: str = Field(..., description="The original normalized question")
    answer: str = Field(..., description="Synthesized answer or explanation of unavailable context")
    sources: List[AskSource] = Field(
        default_factory=list,
        description="List of retrieved evidence chunks with attribution and relevance scores",
    )
    model_name: Optional[str] = Field(
        None, description="LLM model used for synthesis"
    )
