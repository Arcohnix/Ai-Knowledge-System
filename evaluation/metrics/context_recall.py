from typing import Any, List, Optional
from evaluation.datasets.schemas import EvalExample, ExpectedBehavior
from evaluation.metrics.base import BaseMetric, MetricResult


class ContextRecallMetric(BaseMetric):
    """
    Measures the proportion of ground-truth relevant documents and facts successfully retrieved.

    Definition (Ragas / Retrieval Coverage):
        Context Recall = (Number of expected ground-truth documents/facts retrieved) /
                         (Total number of expected ground-truth documents/facts)

    Why it is useful:
        A generator cannot generate what the retriever does not provide. In multi-document queries,
        retrieving only 1 of 2 required policies cuts context recall to 0.5, inevitably causing
        incomplete answers or hallucination.
    """

    name = "context_recall"
    description = "Measures whether all required ground-truth evidence was retrieved."

    def _chunk_matches_doc(self, chunk: Any, expected_doc: str) -> bool:
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

        if doc_name and expected_doc.lower() in doc_name.lower():
            return True
        return False

    def _context_contains_fact(self, contexts: List[str], expected_fact: str) -> bool:
        fact_lower = expected_fact.lower().strip()
        full_text = " ".join(contexts).lower()
        if fact_lower in full_text:
            return True
        words = [w for w in fact_lower.split() if len(w) > 4]
        if words and sum(1 for w in words if w in full_text) >= max(1, len(words) // 2):
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
        # Unsupported question: no documents are expected
        if not example.expected_relevant_documents and not example.expected_relevant_chunks:
            return MetricResult(
                metric_name=self.name,
                score=1.0,
                details={"expected_documents": [], "missed_documents": []},
                explanation="No documents expected in ground truth for unsupported question. Context recall is 1.0.",
            )

        expected_docs = example.expected_relevant_documents
        matched_docs: List[str] = []
        missed_docs: List[str] = []

        for expected_doc in expected_docs:
            found = any(self._chunk_matches_doc(chunk, expected_doc) for chunk in retrieved_chunks)
            if found:
                matched_docs.append(expected_doc)
            else:
                missed_docs.append(expected_doc)

        # Fallback to chunk statements if expected documents list is empty
        if not expected_docs and example.expected_relevant_chunks:
            matched_facts = [
                f for f in example.expected_relevant_chunks if self._context_contains_fact(retrieved_contexts, f)
            ]
            score = round(len(matched_facts) / len(example.expected_relevant_chunks), 4)
            return MetricResult(
                metric_name=self.name,
                score=score,
                details={
                    "total_expected_facts": len(example.expected_relevant_chunks),
                    "matched_facts_count": len(matched_facts),
                    "matched_facts": matched_facts,
                },
                explanation=f"Retrieved {len(matched_facts)}/{len(example.expected_relevant_chunks)} expected factual statements.",
            )

        total_expected = len(expected_docs)
        if total_expected == 0:
            score = 1.0
        else:
            score = round(len(matched_docs) / total_expected, 4)

        return MetricResult(
            metric_name=self.name,
            score=score,
            details={
                "total_expected_documents": total_expected,
                "retrieved_expected_documents": matched_docs,
                "missed_documents": missed_docs,
            },
            explanation=(
                f"Retrieved {len(matched_docs)}/{total_expected} expected documents. "
                f"Matched: {matched_docs}. Missed: {missed_docs}."
            ),
        )
