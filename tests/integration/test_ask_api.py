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
async def test_ask_supported_question_with_citations(async_client: AsyncClient):
    """
    Test 1: Supported question.
    Verifies that a factual question grounded in evidence generates an answer,
    returns high/medium confidence, has reason=None, and includes complete source citations.
    """
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

    # Verify Confidence & Reason
    assert data["confidence"] in ("high", "medium")
    assert data["confidence_score"] > 0.5
    assert data["reason"] is None

    # Verify Source Citation Structure: document, version, chunk_id, score, excerpt
    sources = data["sources"]
    assert len(sources) >= 1
    source = sources[0]

    assert source["document"] == "sample.docx"
    assert source["version"] is not None
    assert isinstance(source["chunk_id"], str)
    assert isinstance(source["score"], float)
    assert source["score"] >= 0.3
    assert "retained" in source["excerpt"].lower() or "7 years" in source["excerpt"]


@pytest.mark.asyncio
async def test_ask_partially_supported_question(async_client: AsyncClient):
    """
    Test 2: Partially supported question.
    Verifies question with moderate evidence yields medium/low confidence with sources.
    """
    with open(FIXTURES_DIR / "sample.txt", "rb") as f:
        doc_content = f.read()

    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("operations.txt", doc_content, "text/plain")},
    )
    assert upload_res.status_code == 201

    # Query that is partially supported by deployment windows guidelines
    ask_payload = {
        "question": "When are production deployments permitted for updates?",
        "top_k": 2,
        "similarity_threshold": 0.3,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["confidence"] in ("medium", "low")
    assert data["reason"] is None
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["document"] == "operations.txt"
    assert len(data["sources"][0]["excerpt"]) > 0


@pytest.mark.asyncio
async def test_ask_unsupported_question_refusal(async_client: AsyncClient):
    """
    Test 3: Unsupported question.
    Verifies that unsupported questions return the controlled refusal message,
    confidence='none', machine-readable reason='INSUFFICIENT_EVIDENCE', and empty sources.
    """
    with open(FIXTURES_DIR / "sample.docx", "rb") as f:
        doc_content = f.read()
    await async_client.post(
        "/api/v1/documents",
        files={
            "file": (
                "sample.docx",
                doc_content,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        },
    )

    # Ask an out-of-scope question that is not covered in the retention policy
    ask_payload = {
        "question": "What is the company's policy on bringing dogs or pets into the office?",
        "similarity_threshold": 0.40,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    # Controlled refusal
    assert (
        "couldn't find sufficient information" in data["answer"].lower()
        or "not have enough information" in data["answer"].lower()
    )
    assert data["confidence"] == "none"
    assert data["confidence_score"] == 0.0
    assert data["reason"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["sources"]) == 0


@pytest.mark.asyncio
async def test_ask_low_retrieval_score_triggers_insufficient_evidence(async_client: AsyncClient):
    """
    Test 4: Low retrieval score.
    When similarity threshold is set high (e.g. 0.90) and retrieval cannot meet it,
    the system must refuse with INSUFFICIENT_EVIDENCE.
    """
    with open(FIXTURES_DIR / "sample.md", "rb") as f:
        doc_content = f.read()
    await async_client.post(
        "/api/v1/documents",
        files={"file": ("sample.md", doc_content, "text/markdown")},
    )

    ask_payload = {
        "question": "What is the procedure for zero trust mutual TLS authentication?",
        "similarity_threshold": 0.95,  # Unattainable threshold
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["confidence"] == "none"
    assert data["reason"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["sources"]) == 0
    assert "couldn't find sufficient information" in data["answer"].lower() or "not have enough information" in data["answer"].lower()


@pytest.mark.asyncio
async def test_ask_empty_retrieval_results(async_client: AsyncClient):
    """
    Test 5: Empty retrieval results.
    When database has no documents, system must immediately return controlled refusal.
    """
    ask_payload = {
        "question": "Where are corporate tax ledgers stored?",
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    assert data["confidence"] == "none"
    assert data["confidence_score"] == 0.0
    assert data["reason"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["sources"]) == 0
    assert "couldn't find sufficient information" in data["answer"].lower() or "not have enough information" in data["answer"].lower()


@pytest.mark.asyncio
async def test_ask_llm_failure_handling(async_client: AsyncClient, monkeypatch):
    """
    Test 6: LLM failure.
    Verifies that provider connectivity errors translate to HTTP 502 with LLM_GENERATION_ERROR.
    """
    with open(FIXTURES_DIR / "sample.txt", "rb") as f:
        doc_content = f.read()

    await async_client.post(
        "/api/v1/documents",
        files={"file": ("operations.txt", doc_content, "text/plain")},
    )

    failing_provider = MockLLMProvider(should_fail=True)
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_provider", lambda *args, **kwargs: failing_provider
    )

    ask_payload = {
        "question": "Are code reviews mandatory?",
        "similarity_threshold": 0.2,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 502
    data = res.json()
    assert data["success"] is False
    assert data["error"]["code"] == "LLM_GENERATION_ERROR"
    assert "Mock LLM failure" in data["error"]["message"]


@pytest.mark.asyncio
async def test_ask_malformed_requests(async_client: AsyncClient):
    """
    Test 7: Malformed request payloads.
    """
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
    Test 8: Benchmark evaluation dataset execution.
    Validates grounded answering for answerable questions and strict refusal for unsupported questions.
    """
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
            assert data["confidence"] in ("high", "medium", "low")
            if len(data["sources"]) > 0:
                retrieved_files = [s["document"] for s in data["sources"]]
                assert item["expected_document"] in retrieved_files
                for expected_fact in item["expected_answer_contains"]:
                    assert expected_fact.lower() in data["answer"].lower() or any(
                        expected_fact.lower() in s["excerpt"].lower()
                        for s in data["sources"]
                    )
        else:
            assert data["confidence"] == "none"
            assert data["reason"] == "INSUFFICIENT_EVIDENCE"
            assert (
                "couldn't find sufficient information" in data["answer"].lower()
                or "not have enough information" in data["answer"].lower()
                or "unavailable" in data["answer"].lower()
            )
            assert len(data["sources"]) == 0
