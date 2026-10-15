import pytest
from app.retrieval.base import RetrievedChunk
from app.evaluation.evidence_evaluator import EvidenceEvaluator


def test_evidence_evaluator_empty_retrieval():
    evaluator = EvidenceEvaluator()
    result = evaluator.evaluate_sufficiency(retrieved_chunks=[])
    assert not result.is_sufficient
    assert result.reason == "INSUFFICIENT_EVIDENCE"
    assert result.details["failure_stage"] == "empty_retrieval"
    assert result.top_score == 0.0


def test_evidence_evaluator_low_similarity():
    evaluator = EvidenceEvaluator(min_similarity_threshold=0.60)
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Some irrelevant text about office printers.",
        score=0.45,
    )
    result = evaluator.evaluate_sufficiency(retrieved_chunks=[chunk])
    assert not result.is_sufficient
    assert result.reason == "INSUFFICIENT_EVIDENCE"
    assert result.details["failure_stage"] == "low_similarity"
    assert result.top_score == 0.45


def test_evidence_evaluator_poor_keyword_coverage():
    evaluator = EvidenceEvaluator(min_similarity_threshold=0.30, min_keyword_coverage=0.30)
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="All pull requests require two peer approvals before merging to main branch.",
        score=0.75,
    )
    # Query has keywords entirely unrelated to PR approvals
    result = evaluator.evaluate_sufficiency(
        retrieved_chunks=[chunk],
        query_keywords=["quantum", "encryption", "cryptography", "key"],
    )
    assert not result.is_sufficient
    assert result.reason == "INSUFFICIENT_EVIDENCE"
    assert result.details["failure_stage"] == "poor_keyword_coverage"
    assert result.keyword_coverage == 0.0


def test_evidence_evaluator_sufficient_evidence():
    evaluator = EvidenceEvaluator(min_similarity_threshold=0.40)
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Customer personal identification data must be retained for at least 7 years pursuant to regulatory requirements.",
        score=0.82,
    )
    result = evaluator.evaluate_sufficiency(
        retrieved_chunks=[chunk],
        query_keywords=["customer", "identification", "retained", "years"],
    )
    assert result.is_sufficient
    assert result.reason is None
    assert result.top_score == 0.82
    assert result.keyword_coverage == 1.0


def test_evidence_evaluator_threshold_override():
    evaluator = EvidenceEvaluator(min_similarity_threshold=0.40)
    chunk = RetrievedChunk(
        chunk_id="chunk-1",
        document_id="doc-1",
        content="Production deployments are permitted Tuesday through Thursday.",
        score=0.55,
    )
    # Default threshold 0.40 passes
    res_default = evaluator.evaluate_sufficiency(retrieved_chunks=[chunk])
    assert res_default.is_sufficient

    # Query threshold override to 0.70 fails
    res_override = evaluator.evaluate_sufficiency(
        retrieved_chunks=[chunk],
        min_similarity_threshold=0.70,
    )
    assert not res_override.is_sufficient
    assert res_override.details["failure_stage"] == "low_similarity"
