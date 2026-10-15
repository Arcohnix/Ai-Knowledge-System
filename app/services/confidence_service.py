import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


class ConfidenceResult(BaseModel):
    """
    Transparent, deterministic confidence calculation result.
    Fully explainable and derived strictly from retrieval metrics and lexical coverage.
    """

    level: str = Field(
        ...,
        description="Categorical confidence rating: 'high', 'medium', 'low', or 'none'",
    )
    score: float = Field(
        ...,
        description="Calculated composite confidence score bounded between 0.0 and 1.0",
    )
    reason: Optional[str] = Field(
        default=None,
        description="Machine-readable diagnostic code when evidence is insufficient (e.g. 'INSUFFICIENT_EVIDENCE')",
    )
    breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Component scores contributing to the composite calculation",
    )


class ConfidenceScorer:
    """
    Evaluates evidence sufficiency and computes an objective confidence score.

    Mathematical Formulation:
    -------------------------
    Composite Confidence = (w1 * TopSimilarity) + (w2 * MeanSimilarity) + (w3 * LexicalCoverage)
    - TopSimilarity (w1 = 0.50): Cosine similarity of the rank-1 retrieved chunk.
    - MeanSimilarity (w2 = 0.30): Average similarity of the top-k (max 3) retrieved chunks.
    - LexicalCoverage (w3 = 0.20): Proportion of query keywords verified present in retrieved chunks.

    Confidence Levels:
    ------------------
    - 'high': Composite >= 0.75 (Strong semantic alignment, multi-chunk concordance, high keyword recall).
    - 'medium': 0.55 <= Composite < 0.75 (Moderate semantic alignment or single-source evidence).
    - 'low': 0.40 <= Composite < 0.55 (Marginal evidence; caution advised).
    - 'none': Composite < 0.40 OR top score < threshold OR model refusal -> 'INSUFFICIENT_EVIDENCE'.
    """

    def __init__(
        self,
        w_top: float = 0.50,
        w_mean: float = 0.30,
        w_lex: float = 0.20,
        high_threshold: Optional[float] = None,
        medium_threshold: Optional[float] = None,
        sufficiency_threshold: Optional[float] = None,
    ):
        self.w_top = w_top
        self.w_mean = w_mean
        self.w_lex = w_lex
        self.high_threshold = high_threshold or settings.RAG_HIGH_CONFIDENCE_THRESHOLD
        self.medium_threshold = medium_threshold or settings.RAG_MEDIUM_CONFIDENCE_THRESHOLD
        self.sufficiency_threshold = (
            sufficiency_threshold or settings.RAG_DEFAULT_SIMILARITY_THRESHOLD
        )

    def calculate_lexical_coverage(
        self, keywords: List[str], chunks: List[RetrievedChunk]
    ) -> float:
        """
        Computes the fraction of query keywords that appear in the retrieved text evidence.
        """
        if not keywords:
            return 1.0
        if not chunks:
            return 0.0

        corpus = " ".join(c.content.lower() for c in chunks)
        tokens = set(re.findall(r"\b\w{3,}\b", corpus))

        matched = sum(1 for kw in keywords if kw.lower() in tokens)
        return round(matched / len(keywords), 4)

    def evaluate(
        self,
        retrieved_chunks: List[RetrievedChunk],
        query_keywords: Optional[List[str]] = None,
        is_refusal: bool = False,
    ) -> ConfidenceResult:
        """
        Computes the confidence result for a question and its retrieved chunks.
        """
        # Case 1: Model explicitly refused or determined information is unavailable
        if is_refusal:
            logger.info("Confidence evaluated as 'none' due to explicit model refusal.")
            return ConfidenceResult(
                level="none",
                score=0.0,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                breakdown={"refusal": 1.0},
            )

        # Case 2: Zero chunks retrieved
        if not retrieved_chunks:
            logger.info("Confidence evaluated as 'none' due to zero retrieved chunks.")
            return ConfidenceResult(
                level="none",
                score=0.0,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                breakdown={"retrieved_chunks_count": 0.0},
            )

        top_score = float(retrieved_chunks[0].score or 0.0)

        # Case 3: Top score below minimum sufficiency threshold
        if top_score < self.sufficiency_threshold:
            logger.info(
                f"Confidence evaluated as 'none' because top score {top_score:.4f} < {self.sufficiency_threshold}"
            )
            return ConfidenceResult(
                level="none",
                score=round(top_score, 4),
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                breakdown={"top_similarity": top_score, "threshold": self.sufficiency_threshold},
            )

        # Compute Mean Similarity across top 3 chunks
        top_k_chunks = retrieved_chunks[:3]
        mean_score = sum(float(c.score or 0.0) for c in top_k_chunks) / len(top_k_chunks)

        # Compute Lexical Keyword Coverage
        keywords = query_keywords or []
        lex_coverage = self.calculate_lexical_coverage(keywords, top_k_chunks)

        # Weighted Composite Score
        composite = (
            (self.w_top * top_score)
            + (self.w_mean * mean_score)
            + (self.w_lex * lex_coverage)
        )
        composite = min(1.0, max(0.0, round(composite, 4)))

        # Categorize
        if composite >= self.high_threshold:
            level = "high"
            reason = None
        elif composite >= self.medium_threshold:
            level = "medium"
            reason = None
        elif composite >= 0.40:
            level = "low"
            reason = None
        else:
            level = "none"
            reason = settings.RAG_INSUFFICIENT_EVIDENCE_REASON

        breakdown = {
            "top_similarity": round(top_score, 4),
            "mean_similarity": round(mean_score, 4),
            "lexical_coverage": round(lex_coverage, 4),
            "composite": composite,
        }

        logger.debug(f"Confidence score computed: {level} ({composite}) breakdown: {breakdown}")

        return ConfidenceResult(
            level=level,
            score=composite,
            reason=reason,
            breakdown=breakdown,
        )
