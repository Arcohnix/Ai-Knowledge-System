from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EvaluationMetric(BaseModel):
    """Represents a single evaluated metric (e.g. faithfulness, context_precision)."""

    name: str
    score: float = Field(..., ge=0.0, le=1.0)
    reasoning: Optional[str] = None


class RAGEvaluationReport(BaseModel):
    """Aggregated evaluation report across standard RAG reliability dimensions."""

    faithfulness: Optional[float] = Field(None, ge=0.0, le=1.0)
    answer_relevancy: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_precision: Optional[float] = Field(None, ge=0.0, le=1.0)
    context_recall: Optional[float] = Field(None, ge=0.0, le=1.0)
    passed_quality_threshold: bool = True
    metrics: List[EvaluationMetric] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseEvaluator(ABC):
    """
    Abstract interface for evaluating RAG pipeline outputs.
    Future implementation will wrap Ragas, TruLens, or custom LLM-as-judge prompts.
    """

    @abstractmethod
    async def evaluate_interaction(
        self,
        query: str,
        retrieved_contexts: List[str],
        generated_answer: str,
        ground_truth: Optional[str] = None,
    ) -> RAGEvaluationReport:
        """Run faithfulness, hallucination, and relevance checks against context and answer."""
        pass
