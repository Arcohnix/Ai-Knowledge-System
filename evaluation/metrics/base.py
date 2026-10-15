from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from evaluation.datasets.schemas import EvalExample


class MetricResult(BaseModel):
    """Detailed score and diagnostic metadata produced by an evaluation metric."""

    metric_name: str = Field(..., description="Canonical metric identifier")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized metric score between 0.0 and 1.0")
    details: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic data and intermediate calculations")
    explanation: str = Field(default="", description="Human-readable rationale for the score")


class BaseMetric(ABC):
    """Abstract base class for all RAG evaluation metrics."""

    name: str = "base_metric"
    description: str = "Abstract base metric"

    @abstractmethod
    async def compute(
        self,
        example: EvalExample,
        answer: str,
        retrieved_contexts: List[str],
        retrieved_chunks: List[Any],
        reliability_report: Optional[Any] = None,
        contradiction_report: Optional[Any] = None,
    ) -> MetricResult:
        """
        Calculates metric score for a given evaluation example and RAG execution output.
        """
        pass
