from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class QuestionCategory(str, Enum):
    """Taxonomy of evaluation question types."""

    STRAIGHTFORWARD = "straightforward"
    MULTI_DOCUMENT = "multi_document"
    UNSUPPORTED = "unsupported"
    AMBIGUOUS = "ambiguous"
    VERSIONING = "versioning"
    CONTRADICTION = "contradiction"


class ExpectedBehavior(str, Enum):
    """Expected RAG pipeline behavioral outcome."""

    ANSWER_WITH_CITATIONS = "answer_with_citations"
    REFUSE_INSUFFICIENT_EVIDENCE = "refuse_insufficient_evidence"
    RESOLVE_SUPERSEDED_VERSION = "resolve_superseded_version"
    REFUSE_CONFLICTING_EVIDENCE = "refuse_conflicting_evidence"


class EvalExample(BaseModel):
    """Single test example in the RAG evaluation dataset."""

    id: str = Field(..., description="Unique scenario identifier")
    question: str = Field(..., description="User query prompt")
    category: QuestionCategory = Field(..., description="Classification category")
    is_answerable: bool = Field(..., description="Whether evidence exists to answer")
    expected_answer: str = Field(..., description="Ground truth answer or refusal note")
    expected_relevant_documents: List[str] = Field(
        default_factory=list,
        description="List of document filenames required to answer",
    )
    expected_relevant_chunks: List[str] = Field(
        default_factory=list,
        description="Key excerpt statements/facts expected in retrieved context",
    )
    expected_behavior: ExpectedBehavior = Field(
        default=ExpectedBehavior.ANSWER_WITH_CITATIONS,
        description="Expected pipeline behavior",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplemental metadata for scenario analysis",
    )


class EvalDataset(BaseModel):
    """Complete collection of benchmark evaluation examples."""

    name: str = Field(default="ai_knowledge_benchmark")
    version: str = Field(default="1.0.0")
    description: str = Field(default="Enterprise knowledge assistant evaluation benchmark")
    examples: List[EvalExample] = Field(default_factory=list)
