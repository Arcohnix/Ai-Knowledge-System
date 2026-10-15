from typing import Any, List, Optional
from evaluation.datasets.schemas import EvalExample, ExpectedBehavior
from evaluation.metrics.base import BaseMetric, MetricResult


class ContextPrecisionMetric(BaseMetric):
    """
    Measures whether the relevant retrieved chunks are ranked at the top of the context list.

    Definition (Ragas / Information Retrieval Average Precision):
        Context Precision@K = sum_{k=1}^K (Precision@k * v_k) / (Total relevant chunks in top K)

        where:
            v_k in {0, 1} indicates whether the chunk at rank k is relevant to ground truth.
            Precision@k = (Number of relevant chunks among top k) / k.

    Why it is useful:
        Retrieval quality depends not just on finding facts, but on prioritizing them at rank 1 and 2.
        Ranking irrelevant chunks above relevant evidence leads to "lost-in-the-middle" attention degradation
        and increases generator hallucination risk.
    """

    name = "context_precision"
    description = "Evaluates whether relevant chunks are ranked higher than irrelevant ones."

    def _is_chunk_relevant(self, chunk: Any, example: EvalExample) -> bool:
        """Determines if a retrieved chunk matches expected documents or expected factual facts."""
        content = ""
        if hasattr(chunk, "excerpt") and getattr(chunk, "excerpt"):
            content = str(getattr(chunk, "excerpt"))
        elif hasattr(chunk, "content") and getattr(chunk, "content"):
            content = str(getattr(chunk, "content"))
        elif isinstance(chunk, dict):
            content = str(chunk.get("excerpt") or chunk.get("content") or "")
        else:
            content = str(chunk)

        doc_name = ""
        if hasattr(chunk, "document") and getattr(chunk, "document"):
            doc_name = str(getattr(chunk, "document"))
        elif isinstance(chunk, dict) and chunk.get("document"):
            doc_name = str(chunk.get("document"))
        elif hasattr(chunk, "metadata"):
            meta = getattr(chunk, "metadata") or {}
            doc_name = meta.get("filename", "") or meta.get("source", "")
        elif isinstance(chunk, dict) and chunk.get("metadata"):
            meta = chunk.get("metadata") or {}
            doc_name = meta.get("filename", "") or meta.get("source", "")

        # Check filename match against expected documents
        if doc_name and any(expected_doc.lower() in doc_name.lower() for expected_doc in example.expected_relevant_documents):
            return True

        # Check content overlap against expected chunk statements
        content_lower = content.lower()
        for expected_fact in example.expected_relevant_chunks:
            # Check key phrases (at least 15 chars or main keywords)
            clean_fact = expected_fact.lower().strip()
            if clean_fact in content_lower:
                return True
            # Check substring matches
            words = [w for w in clean_fact.split() if len(w) > 4]
            if words and sum(1 for w in words if w in content_lower) >= max(1, len(words) // 2):
                return True

        return False

    async def compute(
        self,
        example: EvalExample,
        answer: str,
        retrieved_contexts: List[str],
        retrieved_chunks: List[Any],
        reliability_report: Optional[Any] = None,
        contradiction_report: Optional[Any] = None,
    ) -> MetricResult:
        # Case: Unsupported question where no documents are relevant in ground truth
        if not example.expected_relevant_documents and not example.expected_relevant_chunks:
            # For unsupported questions, ideal retrieval either retrieves nothing or pipeline flags insufficient
            is_refused = (
                "couldn't find sufficient information" in answer.lower()
                or (reliability_report and getattr(reliability_report, "status", "") == "REFUSE")
            )
            score = 1.0 if is_refused else 0.5
            return MetricResult(
                metric_name=self.name,
                score=score,
                details={
                    "is_unsupported_scenario": True,
                    "pipeline_refused": is_refused,
                    "total_retrieved": len(retrieved_chunks),
                },
                explanation="No relevant documents exist in knowledge base for this question. System correctly guarded against ranking false evidence.",
            )

        if not retrieved_chunks:
            return MetricResult(
                metric_name=self.name,
                score=0.0,
                details={"retrieved_chunks_count": 0},
                explanation="No chunks retrieved, yielding 0.0 context precision.",
            )

        # Compute Average Precision at K
        relevance_vector: List[int] = []
        precision_at_k: List[float] = []

        relevant_count = 0
        running_sum = 0.0

        for k, chunk in enumerate(retrieved_chunks, start=1):
            is_rel = 1 if self._is_chunk_relevant(chunk, example) else 0
            relevance_vector.append(is_rel)
            if is_rel:
                relevant_count += 1
            prec_k = relevant_count / k
            precision_at_k.append(prec_k)
            if is_rel:
                running_sum += prec_k

        if relevant_count == 0:
            score = 0.0
            explanation = f"None of the {len(retrieved_chunks)} retrieved chunks matched expected ground-truth documents {example.expected_relevant_documents}."
        else:
            score = round(running_sum / relevant_count, 4)
            explanation = (
                f"Retrieved {relevant_count} relevant chunk(s) out of {len(retrieved_chunks)} total. "
                f"Relevance ranks vector: {relevance_vector}. Context Precision@K = {score}."
            )

        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "k": len(retrieved_chunks),
                "relevant_chunks_count": relevant_count,
                "relevance_vector": relevance_vector,
                "precision_at_k": precision_at_k,
            },
            explanation=explanation,
        )
