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


class SourceCitation(BaseModel):
    """
    Structured citation pointing directly to the retrieved knowledge chunk.
    Guarantees every factual assertion is traceable to its source document and version.
    """

    document: str = Field(
        ...,
        description="Document filename or display name",
        examples=["employee_handbook.pdf"],
    )
    version: str = Field(
        default="1.0",
        description="Version identifier of the originating document",
        examples=["2026.1", "1.0"],
    )
    chunk_id: str = Field(
        ...,
        description="Unique UUID identifying the source chunk",
        examples=["9377674e-b7fe-49f6-8fa5-127f6c136ada"],
    )
    score: float = Field(
        ...,
        description="Cosine similarity relevance score between query and chunk",
        examples=[0.87],
    )
    excerpt: str = Field(
        ...,
        description="Extracted text snippet or relevant quote from the chunk",
        examples=[
            "Customer personal identification data must be retained for at least 7 years pursuant to regulatory requirements."
        ],
    )
    chunk_index: Optional[int] = Field(
        default=None,
        description="0-indexed position of chunk within the document",
    )
    document_id: Optional[str] = Field(
        default=None,
        description="UUID of the parent document entity",
    )


class ReliabilityReport(BaseModel):
    """
    Encapsulates post-generation faithfulness verification and reliability decision.
    Validates whether claims in the generated answer are grounded in retrieved evidence.
    """

    status: str = Field(
        ...,
        description="Reliability action: 'ACCEPT', 'REGENERATE', or 'REFUSE'",
        examples=["ACCEPT", "REFUSE"],
    )
    faithfulness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Fraction of claims in the answer verified as grounded in the retrieved context",
        examples=[1.0, 0.0],
    )
    supported: bool = Field(
        ...,
        description="True if all claims are faithfully supported by evidence",
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of factual claims that could not be verified against the context",
    )
    reason: str = Field(
        ...,
        description="Diagnostic explanation of the verification outcome",
    )


class ContradictionReport(BaseModel):
    """
    Audited outcome of contradiction analysis across retrieved evidence chunks.
    Detects whether evidence sources agree, partially conflict, or directly conflict,
    and documents whether the conflict was deterministically resolved via document versioning
    or lifecycle precedence.
    """

    status: str = Field(
        ...,
        description="Contradiction status: 'AGREE', 'PARTIALLY_CONFLICT', or 'CONFLICT'",
        examples=["AGREE", "CONFLICT"],
    )
    has_conflict: bool = Field(
        ...,
        description="True if factual discrepancies or policy divergence were detected across sources",
    )
    claims: List[str] = Field(
        default_factory=list,
        description="Extracted factual claims evaluated across retrieved sources",
    )
    conflicting_sources: List[str] = Field(
        default_factory=list,
        description="Document filenames or identifiers involved in factual conflict",
        examples=[["vacation_policy_2024.docx", "vacation_policy_2026.docx"]],
    )
    resolution_status: Optional[str] = Field(
        default=None,
        description="Resolution status: 'NO_CONFLICT', 'RESOLVED_BY_ACTIVE_STATUS', 'RESOLVED_BY_EFFECTIVE_DATE', 'RESOLVED_BY_NEWER_VERSION', or 'UNRESOLVED'",
        examples=["RESOLVED_BY_ACTIVE_STATUS", "UNRESOLVED"],
    )
    resolution_rationale: Optional[str] = Field(
        default=None,
        description="Detailed rationale explaining how conflict was resolved or why it was refused",
    )


class AskResponse(BaseModel):
    """
    Unified question-answering response containing synthesized answer,
    calculated confidence level, optional machine-readable refusal reason, structured source citations,
    faithfulness reliability audit report, and contradiction/version analysis.
    """

    question: str = Field(..., description="The original normalized question")
    answer: str = Field(
        ...,
        description="Synthesized grounded answer or controlled refusal when evidence is insufficient or contradictory",
    )
    confidence: str = Field(
        ...,
        description="Calculated confidence level: 'high', 'medium', 'low', or 'none'",
        examples=["high", "medium", "low", "none"],
    )
    confidence_score: float = Field(
        default=0.0,
        description="Deterministic composite confidence score (0.0 to 1.0) derived from retrieval and lexical coverage",
        examples=[0.87],
    )
    reason: Optional[str] = Field(
        default=None,
        description="Machine-readable status code when question cannot be answered (e.g. 'INSUFFICIENT_EVIDENCE', 'CONFLICTING_EVIDENCE')",
        examples=["INSUFFICIENT_EVIDENCE", "UNVERIFIED_CLAIMS", "CONFLICTING_EVIDENCE"],
    )
    sources: List[SourceCitation] = Field(
        default_factory=list,
        description="Structured source citations used to generate the answer or conflicting sources",
    )
    reliability: Optional[ReliabilityReport] = Field(
        default=None,
        description="Faithfulness verification and reliability decision report",
    )
    contradiction: Optional[ContradictionReport] = Field(
        default=None,
        description="Contradiction analysis and version resolution report",
    )
    model_name: Optional[str] = Field(
        default=None,
        description="LLM model used for synthesis",
    )



# Backwards compatibility models
class AskSourceDocument(BaseModel):
    document_id: str
    title: Optional[str] = None
    filename: str
    document_type: str
    source: Optional[str] = "upload"
    version: Optional[str] = "1.0"


class AskSourceChunk(BaseModel):
    chunk_id: str
    chunk_index: int
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskSource(BaseModel):
    document: AskSourceDocument
    chunk: AskSourceChunk
    relevance_score: float
