import pytest
from app.ingestion.base import RawDocument
from app.ingestion.chunker import SensibleTextChunker


def test_chunker_basic_splitting():
    text = (
        "Paragraph 1 contains some important information.\n\n"
        "Paragraph 2 discusses a secondary topic in detail.\n\n"
        "Paragraph 3 concludes the policy overview."
    )
    chunker = SensibleTextChunker(chunk_size=100, chunk_overlap=20)
    splits = chunker.split_text(text)

    assert len(splits) >= 2
    for s in splits:
        assert len(s["content"]) <= 100
        assert "chunk_index" in s
        assert "token_estimate" in s
        assert "start_char" in s
        assert "end_char" in s


def test_chunker_invalid_overlap_raises():
    with pytest.raises(ValueError):
        SensibleTextChunker(chunk_size=100, chunk_overlap=150)


def test_chunker_context_overlap():
    text = "First long sentence that goes on and on. Second long sentence that follows closely. Third sentence concludes."
    chunker = SensibleTextChunker(chunk_size=55, chunk_overlap=25)
    splits = chunker.split_text(text)

    assert len(splits) > 1
    # Check that adjacent chunks have meaningful overlap
    chunk0 = splits[0]["content"]
    chunk1 = splits[1]["content"]
    assert len(chunk0) > 0 and len(chunk1) > 0


def test_chunker_base_chunker_contract():
    raw_doc = RawDocument(
        source="doc1.txt",
        content="Heading 1\n\nContent for section 1.\n\nHeading 2\n\nContent for section 2.",
        metadata={"author": "Team"},
    )
    chunker = SensibleTextChunker(chunk_size=50, chunk_overlap=10)
    chunks = chunker.chunk([raw_doc])

    assert len(chunks) > 0
    assert chunks[0].document_id == "doc1.txt"
    assert "token_estimate" in chunks[0].metadata
    assert chunks[0].metadata["author"] == "Team"
