from pathlib import Path
import pytest
from httpx import AsyncClient
from app.core.config import settings
from app.evaluation.qa_dataset import load_qa_eval_dataset
from app.generation.providers.mock_provider import MockLLMProvider

FIXTURES_DIR = Path("tests/fixtures")


@pytest.fixture(autouse=True)
def setup_rag_test_env(monkeypatch):
    """Ensure mock LLM provider and local embedder are used for integration tests."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")


@pytest.mark.asyncio
async def test_ask_question_success_with_sources(async_client: AsyncClient):
    # 1. Ingest Knowledge Document
    with open(FIXTURES_DIR / "sample.docx", "rb") as f:
        doc_content = f.read()

    upload_res = await async_client.post(
        "/api/v1/documents",
        files={
            "file": (
                "sample.docx",
                doc_content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )
    assert upload_res.status_code == 201
    doc_id = upload_res.json()["document_id"]

    # 2. Ask question grounded in document
    ask_payload = {
        "question": "How long must customer personal identification data be retained?",
        "top_k": 3,
        "similarity_threshold": 0.3,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    # Verify Question & Answer
    assert data["question"] == "How long must customer personal identification data be retained?"
    assert len(data["answer"]) > 0
    assert "7 years" in data["answer"] or "retained" in data["answer"].lower()

    # 3. Verify Source Citation Structure
    sources = data["sources"]
    assert len(sources) >= 1
    source = sources[0]

    # Verify Document Attribution
    assert "document" in source
    assert source["document"]["document_id"] == doc_id
    assert source["document"]["filename"] == "sample.docx"
    assert source["document"]["document_type"] == "docx"

    # Verify Chunk Attribution
    assert "chunk" in source
    assert "chunk_id" in source["chunk"]
    assert source["chunk"]["chunk_index"] == 0
    assert "7 years" in source["chunk"]["content"]

    # Verify Relevance Score
    assert "relevance_score" in source
    assert isinstance(source["relevance_score"], float)
    assert source["relevance_score"] >= 0.3


@pytest.mark.asyncio
async def test_ask_no_relevant_context(async_client: AsyncClient):
    # Ask an out-of-scope question that has no matching document chunks in DB
    ask_payload = {
        "question": "What is the guest Wi-Fi password in the cafeteria?",
        "top_k": 3,
        "similarity_threshold": 0.85,  # High threshold guarantees 0 results
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    assert (
        "not have enough information" in data["answer"].lower()
        or "unavailable" in data["answer"].lower()
    )
    assert len(data["sources"]) == 0


@pytest.mark.asyncio
async def test_ask_llm_failure_handling(async_client: AsyncClient, monkeypatch):
    # Ingest a document so retrieval succeeds
    with open(FIXTURES_DIR / "sample.txt", "rb") as f:
        doc_content = f.read()

    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("operations.txt", doc_content, "text/plain")},
    )
    assert upload_res.status_code == 201

    # Configure MockLLM to simulate provider failure
    failing_provider = MockLLMProvider(should_fail=True)
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_provider", lambda *args, **kwargs: failing_provider
    )

    ask_payload = {
        "question": "Are code reviews mandatory?",
        "similarity_threshold": 0.2,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)

    # Should return HTTP 502 with structured error JSON
    assert res.status_code == 502
    data = res.json()
    assert data["success"] is False
    assert data["error"]["code"] == "LLM_GENERATION_ERROR"
    assert "Mock LLM failure" in data["error"]["message"]


@pytest.mark.asyncio
async def test_ask_malformed_requests(async_client: AsyncClient):
    # Empty question
    res_empty = await async_client.post("/api/v1/ask", json={"question": ""})
    assert res_empty.status_code == 422

    # Whitespace only question
    res_ws = await async_client.post("/api/v1/ask", json={"question": "   "})
    assert res_ws.status_code == 422

    # Missing question payload
    res_missing = await async_client.post("/api/v1/ask", json={})
    assert res_missing.status_code == 422

    # Invalid top_k
    res_k = await async_client.post(
        "/api/v1/ask", json={"question": "Valid question?", "top_k": 0}
    )
    assert res_k.status_code == 422


@pytest.mark.asyncio
async def test_qa_eval_dataset_benchmark(async_client: AsyncClient):
    """
    Executes benchmark evaluation dataset across ingested sample documents.
    Validates grounded answering for valid questions and strict refusal for out-of-scope questions.
    """
    # Ingest standard evaluation fixture documents
    fixture_files = [
        ("sample.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
        ("sample.md", "text/markdown"),
        ("sample.pdf", "application/pdf"),
        ("sample.txt", "text/plain"),
    ]

    for fname, mime in fixture_files:
        with open(FIXTURES_DIR / fname, "rb") as f:
            content = f.read()
        res = await async_client.post(
            "/api/v1/documents",
            files={"file": (fname, content, mime)},
        )
        assert res.status_code == 201

    eval_items = load_qa_eval_dataset()
    assert len(eval_items) >= 10

    for item in eval_items:
        res = await async_client.post(
            "/api/v1/ask",
            json={
                "question": item["question"],
                "similarity_threshold": 0.40,
                "top_k": 3,
            },
        )
        assert res.status_code == 200
        data = res.json()

        if item["is_answerable"]:
            # Answerable question: should return sources from expected document
            if len(data["sources"]) > 0:
                retrieved_files = [s["document"]["filename"] for s in data["sources"]]
                assert item["expected_document"] in retrieved_files
                for expected_fact in item["expected_answer_contains"]:
                    assert expected_fact.lower() in data["answer"].lower() or any(
                        expected_fact.lower() in s["chunk"]["content"].lower()
                        for s in data["sources"]
                    )
        else:
            # Unanswerable question: should refuse
            assert (
                "not have enough information" in data["answer"].lower()
                or "unavailable" in data["answer"].lower()
            )
            assert len(data["sources"]) == 0
