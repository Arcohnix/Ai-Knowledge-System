from app.retrieval.base import RetrievedChunk
from app.services.response_formatting_service import ResponseFormattingService


def test_response_formatter_with_sources():
    formatter = ResponseFormattingService()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-123",
            document_id="doc-456",
            content="Sample evidence text.",
            score=0.8912,
            metadata={
                "document_title": "Security Handbook",
                "filename": "security.md",
                "document_type": "markdown",
                "source": "upload",
                "version": "1.0",
                "chunk_index": 2,
                "token_estimate": 10,
            },
        )
    ]

    response = formatter.format_response(
        question="What is the security rule?",
        answer="Sample answer based on [Document 1].",
        evidence_chunks=chunks,
        model_name="mock-llm",
    )

    assert response.question == "What is the security rule?"
    assert response.answer == "Sample answer based on [Document 1]."
    assert response.model_name == "mock-llm"
    assert len(response.sources) == 1

    source = response.sources[0]
    assert source.relevance_score == 0.8912
    assert source.document.document_id == "doc-456"
    assert source.document.title == "Security Handbook"
    assert source.document.filename == "security.md"
    assert source.document.document_type == "markdown"
    assert source.chunk.chunk_id == "chunk-123"
    assert source.chunk.chunk_index == 2
    assert source.chunk.content == "Sample evidence text."


def test_response_formatter_no_context():
    formatter = ResponseFormattingService()
    response = formatter.format_no_context_response(
        question="What is the unknown fact?",
        model_name="mock-llm",
    )

    assert response.question == "What is the unknown fact?"
    assert "not have enough information" in response.answer.lower()
    assert len(response.sources) == 0
