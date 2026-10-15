from app.retrieval.base import RetrievedChunk
from app.services.response_formatting_service import ResponseFormattingService


def test_response_formatter_with_sources():
    formatter = ResponseFormattingService()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-123",
            document_id="doc-456",
            content="All incoming network traffic must authenticate via mutual TLS with hardware tokens.",
            score=0.8912,
            metadata={
                "document_title": "Security Architecture",
                "filename": "security.md",
                "document_type": "markdown",
                "source": "upload",
                "version": "2026.1",
                "chunk_index": 2,
            },
        )
    ]

    response = formatter.format_response(
        question="What is the network authentication rule?",
        answer="Based on [Document 1], all network traffic must authenticate via mutual TLS.",
        evidence_chunks=chunks,
        query_keywords=["network", "traffic", "authenticate"],
        model_name="mock-llm",
    )

    assert response.question == "What is the network authentication rule?"
    assert "mutual TLS" in response.answer
    assert response.model_name == "mock-llm"
    assert response.confidence in ("high", "medium")
    assert response.confidence_score > 0.6
    assert response.reason is None
    assert len(response.sources) == 1

    source = response.sources[0]
    assert source.document == "security.md"
    assert source.version == "2026.1"
    assert source.chunk_id == "chunk-123"
    assert source.score == 0.8912
    assert "mutual TLS" in source.excerpt


def test_response_formatter_no_context():
    formatter = ResponseFormattingService()
    response = formatter.format_no_context_response(
        question="What is the unknown fact?",
        model_name="mock-llm",
    )

    assert response.question == "What is the unknown fact?"
    assert (
        "couldn't find sufficient information" in response.answer.lower()
        or "not have enough information" in response.answer.lower()
    )
    assert response.confidence == "none"
    assert response.confidence_score == 0.0
    assert response.reason == "INSUFFICIENT_EVIDENCE"
    assert len(response.sources) == 0


def test_response_formatter_refusal_clears_sources():
    formatter = ResponseFormattingService()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-999",
            document_id="doc-999",
            content="Unrelated text about office supplies.",
            score=0.48,
            metadata={"filename": "supplies.txt", "version": "1.0"},
        )
    ]
    response = formatter.format_response(
        question="What is the quantum encryption protocol?",
        answer="I couldn't find sufficient information in the available knowledge base to answer this question reliably.",
        evidence_chunks=chunks,
        query_keywords=["quantum", "encryption"],
        model_name="mock-llm",
    )

    assert response.confidence == "none"
    assert response.confidence_score == 0.0
    assert response.reason == "INSUFFICIENT_EVIDENCE"
    assert len(response.sources) == 0
