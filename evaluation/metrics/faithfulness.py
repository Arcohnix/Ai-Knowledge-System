from typing import Any, List, Optional
from app.evaluation.answer_verifier import AnswerVerifier
from evaluation.datasets.schemas import EvalExample, ExpectedBehavior
from evaluation.metrics.base import BaseMetric, MetricResult


class FaithfulnessMetric(BaseMetric):
    """
    Measures the factual consistency of the generated answer against the retrieved context.

    Definition (Ragas / Hallucination-prevention):
        Faithfulness = (Number of claims in the generated answer supported by context) /
                       (Total number of claims in the generated answer)

    Why it is useful:
        A model can generate fluent, persuasive answers that hallucinate statements absent
        from corporate documents. Faithfulness guarantees that every statement made in the answer
        is strictly traceable to and supported by retrieved evidence.
    """

    name = "faithfulness"
    description = "Measures factual consistency of the answer against retrieved context."

    def __init__(self, verifier: Optional[AnswerVerifier] = None):
        self.verifier = verifier or AnswerVerifier()

    async def compute(
        self,
        example: EvalExample,
        answer: str,
        retrieved_contexts: List[str],
        retrieved_chunks: List[Any],
        reliability_report: Optional[Any] = None,
        contradiction_report: Optional[Any] = None,
    ) -> MetricResult:
        full_context = "\n\n".join(retrieved_contexts)

        # Handle canonical refusal scenarios
        if example.expected_behavior in (
            ExpectedBehavior.REFUSE_INSUFFICIENT_EVIDENCE,
            ExpectedBehavior.REFUSE_CONFLICTING_EVIDENCE,
        ):
            # Check if system appropriately refused without hallucinating false facts
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
                    details={
                        "is_expected_refusal": True,
                        "refused_appropriately": True,
                        "unsupported_claims": [],
                    },
                    explanation="Pipeline correctly recognized lack of sufficient authoritative evidence and issued a controlled refusal without fabricating claims.",
                )

        # Use verifier to assess claim support against context
        verification = self.verifier.verify_answer(answer=answer, context_text=full_context)
        score = round(verification.score, 4)
        supported_count = max(0, verification.claims_evaluated - len(verification.unsupported_claims))

        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "claims_evaluated": verification.claims_evaluated,
                "supported_claims_count": supported_count,
                "unsupported_claims": verification.unsupported_claims,
                "supported": verification.supported,
            },
            explanation=(
                f"Generated answer contains {verification.claims_evaluated} claim(s), "
                f"of which {supported_count} are supported by the retrieved context. "
                f"Unsupported claims: {verification.unsupported_claims}"
            ),
        )
