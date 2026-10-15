import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


class EvidenceSufficiencyResult(BaseModel):
    """
    Result of pre-generation evidence sufficiency evaluation.
    Determines whether retrieved context is adequate to attempt grounded answer generation.
    """

    is_sufficient: bool = Field(
        ...,
        description="True if evidence meets all quality and relevance thresholds",
    )
    top_score: float = Field(
        default=0.0,
        description="Cosine similarity score of the rank-1 retrieved chunk",
    )
    mean_score: float = Field(
        default=0.0,
        description="Average cosine similarity across top retrieved chunks",
    )
    keyword_coverage: float = Field(
        default=0.0,
        description="Fraction of query keywords present in retrieved text",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Machine-readable diagnostic code (e.g. 'INSUFFICIENT_EVIDENCE')",
    )
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Diagnostic metrics and threshold evaluation details",
    )


class EvidenceEvaluator:
    """
    Evaluates retrieval adequacy before invoking the LLM generation stage.
    Blocks generation if evidence is absent, irrelevant, or low-similarity,
    preventing hallucination by refusing upfront.
    """

    def __init__(
        self,
        min_similarity_threshold: Optional[float] = None,
        min_keyword_coverage: float = 0.25,
    ):
        self.min_similarity_threshold = (
            min_similarity_threshold
            if min_similarity_threshold is not None
            else settings.RAG_DEFAULT_SIMILARITY_THRESHOLD
        )
        self.min_keyword_coverage = min_keyword_coverage

    def evaluate_sufficiency(
        self,
        retrieved_chunks: List[RetrievedChunk],
        query_keywords: Optional[List[str]] = None,
        min_similarity_threshold: Optional[float] = None,
    ) -> EvidenceSufficiencyResult:
        """
        Evaluates whether retrieved chunks contain sufficient evidence for question answering.
        """
        threshold = (
            min_similarity_threshold
            if min_similarity_threshold is not None
            else self.min_similarity_threshold
        )

        # Rule 1: Zero chunks retrieved
        if not retrieved_chunks:
            logger.info("Evidence sufficiency check failed: 0 chunks retrieved.")
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                top_score=0.0,
                mean_score=0.0,
                keyword_coverage=0.0,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                details={"failure_stage": "empty_retrieval", "chunks_count": 0},
            )

        top_score = round(float(retrieved_chunks[0].score or 0.0), 4)
        top_k = retrieved_chunks[:3]
        mean_score = round(sum(float(c.score or 0.0) for c in top_k) / len(top_k), 4)

        # Rule 2: Top score below retrieval threshold
        if top_score < threshold:
            logger.info(
                f"Evidence sufficiency check failed: top score {top_score} < threshold {threshold}"
            )
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                top_score=top_score,
                mean_score=mean_score,
                keyword_coverage=0.0,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                details={
                    "failure_stage": "low_similarity",
                    "top_score": top_score,
                    "threshold": threshold,
                },
            )

        # Rule 3: Keyword coverage
        keywords = query_keywords or []
        coverage = 1.0
        if keywords:
            corpus = " ".join(c.content.lower() for c in retrieved_chunks)
            corpus_tokens = set(re.findall(r"\b\w{3,}\b", corpus))
            matched_keywords = [kw for kw in keywords if kw.lower() in corpus_tokens]
            coverage = round(len(matched_keywords) / len(keywords), 4)

            # If query has distinct keywords and NONE appear in retrieved context
            if (len(keywords) >= 2 and coverage < self.min_keyword_coverage) or (
                len(keywords) == 1 and coverage == 0.0
            ):
                logger.info(
                    f"Evidence sufficiency check failed: keyword coverage {coverage} < {self.min_keyword_coverage}."
                )
                return EvidenceSufficiencyResult(
                    is_sufficient=False,
                    top_score=top_score,
                    mean_score=mean_score,
                    keyword_coverage=coverage,
                    reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                    details={
                        "failure_stage": "poor_keyword_coverage",
                        "coverage": coverage,
                        "min_required": self.min_keyword_coverage,
                        "unmatched_keywords": [kw for kw in keywords if kw.lower() not in corpus_tokens],
                    },
                )

        logger.debug(
            f"Evidence sufficiency check passed: top_score={top_score}, mean_score={mean_score}, coverage={coverage}"
        )

        return EvidenceSufficiencyResult(
            is_sufficient=True,
            top_score=top_score,
            mean_score=mean_score,
            keyword_coverage=coverage,
            reason=None,
            details={"chunks_count": len(retrieved_chunks)},
        )
