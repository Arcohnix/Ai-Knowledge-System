import pytest
from app.retrieval.base import RetrievedChunk
from app.services.confidence_service import ConfidenceScorer


def test_confidence_high_score():
    scorer = ConfidenceScorer()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Customer personal identification data must be retained for at least 7 years pursuant to regulatory requirements.",
            score=0.92,
            metadata={"filename": "retention.pdf", "version": "1.0"},
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-1",
            content="Financial transaction ledgers must be archived in immutable storage for compliance.",
            score=0.85,
            metadata={"filename": "retention.pdf", "version": "1.0"},
        ),
    ]
    keywords = ["customer", "data", "retained", "years"]

    result = scorer.evaluate(retrieved_chunks=chunks, query_keywords=keywords)
    assert result.level == "high"
    assert result.score >= 0.75
    assert result.reason is None
    assert result.breakdown["lexical_coverage"] == 1.0
    assert result.breakdown["top_similarity"] == 0.92


def test_confidence_medium_score():
    scorer = ConfidenceScorer()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="General system architecture overview with high level overview.",
            score=0.62,
            metadata={"filename": "overview.pdf"},
        ),
    ]
    keywords = ["system", "architecture", "deployment"]

    result = scorer.evaluate(retrieved_chunks=chunks, query_keywords=keywords)
    assert result.level in ("medium", "low")
    assert result.score < 0.75
    assert result.score >= 0.40


def test_confidence_insufficient_evidence_empty():
    scorer = ConfidenceScorer()
    result = scorer.evaluate(retrieved_chunks=[])
    assert result.level == "none"
    assert result.score == 0.0
    assert result.reason == "INSUFFICIENT_EVIDENCE"


def test_confidence_insufficient_evidence_low_score():
    scorer = ConfidenceScorer(sufficiency_threshold=0.50)
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Barely relevant chunk with minimal overlap.",
            score=0.35,
            metadata={"filename": "misc.txt"},
        )
    ]
    result = scorer.evaluate(retrieved_chunks=chunks, query_keywords=["quantum", "physics"])
    assert result.level == "none"
    assert result.score == 0.35
    assert result.reason == "INSUFFICIENT_EVIDENCE"


def test_confidence_refusal():
    scorer = ConfidenceScorer()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Relevant text.",
            score=0.88,
        )
    ]
    result = scorer.evaluate(retrieved_chunks=chunks, is_refusal=True)
    assert result.level == "none"
    assert result.score == 0.0
    assert result.reason == "INSUFFICIENT_EVIDENCE"


def test_lexical_coverage_computation():
    scorer = ConfidenceScorer()
    chunks = [
        RetrievedChunk(
            chunk_id="c-1",
            document_id="d-1",
            content="Mutual TLS authentication is required for network traffic.",
        )
    ]
    keywords = ["mutual", "tls", "authentication", "unrelated"]
    coverage = scorer.calculate_lexical_coverage(keywords, chunks)
    assert coverage == 0.75  # 3 of 4 keywords found
