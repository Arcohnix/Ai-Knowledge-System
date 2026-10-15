import re
from typing import Any, List, Optional
import numpy as np
from app.ingestion.base import BaseEmbedder
from app.ingestion.embeddings.factory import get_embedder
from evaluation.datasets.schemas import EvalExample, ExpectedBehavior
from evaluation.metrics.base import BaseMetric, MetricResult


class AnswerRelevancyMetric(BaseMetric):
    """
    Measures how pertinent and directly responsive the generated answer is to the user's question.

    Definition (Ragas):
        Answer Relevancy evaluates whether the answer directly addresses the question's intent
        and constraints, penalizing evasive answers, off-topic rambling, or incomplete coverage.

    Why it is useful:
        A system can retrieve valid text and output a faithful sentence that completely ignores
        the specific question asked. Answer relevancy isolates response pertinence from factual accuracy.
    """

    name = "answer_relevancy"
    description = "Measures how pertinent and responsive the answer is to the user's question."

    def __init__(self, embedder: Optional[BaseEmbedder] = None):
        self.embedder = embedder or get_embedder()

    async def compute(
        self,
        example: EvalExample,
        answer: str,
        retrieved_contexts: List[str],
        retrieved_chunks: List[Any],
        reliability_report: Optional[Any] = None,
        contradiction_report: Optional[Any] = None,
    ) -> MetricResult:
        # Refusal cases: if expected to refuse and correctly refused, response is perfectly relevant
        if example.expected_behavior in (
            ExpectedBehavior.REFUSE_INSUFFICIENT_EVIDENCE,
            ExpectedBehavior.REFUSE_CONFLICTING_EVIDENCE,
        ):
            is_refusal = (
                "couldn't find sufficient information" in answer.lower()
                or "conflicting information" in answer.lower()
                or "unavailable" in answer.lower()
                or (reliability_report and getattr(reliability_report, "status", "") == "REFUSE")
            )
            if is_refusal:
                return MetricResult(
                    metric_name=self.name,
                    score=1.0,
                    details={"is_refusal": True, "appropriate_refusal": True},
                    explanation="Answer correctly identified that the knowledge base lacks authoritative evidence and responded with a direct, unambiguous refusal.",
                )

        # Non-refusal case: Check semantic similarity between question and answer
        if not answer.strip():
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"empty_answer": True},
                explanation="Generated answer was empty, yielding zero relevancy.",
            )

        q_vec = await self.embedder.embed_query(example.question)
        a_vec = await self.embedder.embed_query(answer)

        q_norm = np.linalg.norm(q_vec)
        a_norm = np.linalg.norm(a_vec)

        if q_norm == 0 or a_norm == 0:
            sim = 0.0
        else:
            sim = float(np.dot(q_vec, a_vec) / (q_norm * a_norm))

        # Check interrogative keyword overlap
        q_tokens = set(re.findall(r"\b\w{4,}\b", example.question.lower()))
        a_tokens = set(re.findall(r"\b\w{4,}\b", answer.lower()))
        token_overlap = len(q_tokens.intersection(a_tokens)) / max(len(q_tokens), 1)

        # Combine semantic cosine similarity with lexical concept overlap
        # FastEmbed embeddings for Q&A typically have cosine sim between 0.55 and 0.90 for relevant answers
        normalized_sim = max(0.0, min(1.0, (sim - 0.2) / 0.7))
        combined_score = round(0.7 * normalized_sim + 0.3 * token_overlap, 4)
        combined_score = max(0.0, min(1.0, combined_score))

        return MetricResult(
            metric_name=self.name,
            score=combined_score,
            details={
                "cosine_similarity": round(sim, 4),
                "normalized_similarity": round(normalized_sim, 4),
                "token_overlap": round(token_overlap, 4),
            },
            explanation=(
                f"Computed answer relevancy: cosine semantic similarity={round(sim, 4)}, "
                f"key concept token overlap={round(token_overlap, 4)}. "
                f"Combined normalized relevancy={combined_score}."
            ),
        )
