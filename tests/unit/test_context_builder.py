from app.retrieval.base import RetrievedChunk
from app.services.context_service import ContextConstructionService


def test_context_builder_empty():
    builder = ContextConstructionService()
    ctx = builder.build_context([])
    assert ctx.has_context is False
    assert ctx.context_text == "No relevant documentation found."
    assert len(ctx.included_chunks) == 0


def test_context_builder_formatting():
    builder = ContextConstructionService()
    chunks = [
        RetrievedChunk(
            chunk_id="chunk-1",
            document_id="doc-1",
            content="Customer data must be retained for 7 years.",
            score=0.88,
            metadata={
                "document_title": "Data Retention Policy",
                "filename": "policy.pdf",
                "chunk_index": 0,
            },
        ),
        RetrievedChunk(
            chunk_id="chunk-2",
            document_id="doc-2",
            content="Financial records must be archived in immutable storage.",
            score=0.81,
            metadata={
                "document_title": "Financial Guide",
                "filename": "financial.pdf",
                "chunk_index": 1,
            },
        ),
    ]

    ctx = builder.build_context(chunks)
    assert ctx.has_context is True
    assert len(ctx.included_chunks) == 2
    assert "[Document 1] Title: Data Retention Policy | File: policy.pdf | Chunk: 0 | Relevance: 0.88" in ctx.context_text
    assert "Customer data must be retained for 7 years." in ctx.context_text
    assert "[Document 2] Title: Financial Guide | File: financial.pdf | Chunk: 1 | Relevance: 0.81" in ctx.context_text
    assert "Financial records must be archived in immutable storage." in ctx.context_text


def test_context_builder_character_budget():
    builder = ContextConstructionService(max_context_chars=150)
    chunks = [
        RetrievedChunk(
            chunk_id=f"chunk-{i}",
            document_id=f"doc-{i}",
            content=f"Long detailed content chunk number {i} " * 5,
            score=0.9 - (i * 0.1),
            metadata={"filename": f"doc_{i}.txt", "chunk_index": 0},
        )
        for i in range(5)
    ]

    ctx = builder.build_context(chunks)
    assert ctx.has_context is True
    assert len(ctx.included_chunks) < 5
    assert ctx.total_chars <= 150 or len(ctx.included_chunks) == 1
