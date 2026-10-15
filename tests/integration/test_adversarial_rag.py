from pathlib import Path
import pytest
from httpx import AsyncClient
from app.core.config import settings
from app.generation.base import BaseLLMProvider, LLMResponse
from app.generation.providers.mock_provider import MockLLMProvider

FIXTURES_DIR = Path("tests/fixtures")


@pytest.fixture(autouse=True)
def setup_rag_test_env(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")


@pytest.mark.asyncio
async def test_adversarial_fact_not_in_documents(async_client: AsyncClient):
    """
    Adversarial Scenario 1: Question about a fact not in documents.
    Knowledge base contains data retention rules. User asks about pet policy.
    The system must refuse rather than hallucinate corporate rules.
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

    # Question about unmentioned office pet policy
    res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "Can software engineers bring their dogs to the office on Fridays?",
            "similarity_threshold": 0.40,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["confidence"] == "none"
    assert data["confidence_score"] == 0.0
    assert data["reason"] in ("INSUFFICIENT_EVIDENCE", "UNVERIFIED_CLAIMS")
    assert len(data["sources"]) == 0
    assert (
        "couldn't find sufficient information" in data["answer"].lower()
        or "not have enough information" in data["answer"].lower()
    )

    # Reliability audit report is visible
    assert "reliability" in data
    assert data["reliability"]["status"] == "REFUSE"


@pytest.mark.asyncio
async def test_adversarial_unrelated_documents_retrieval(async_client: AsyncClient):
    """
    Adversarial Scenario 2: Unrelated documents in the knowledge base.
    Knowledge base contains mutual TLS security rules. User asks about tax filing penalties.
    Pre-generation evidence sufficiency must catch low similarity or 0 keyword coverage.
    """
    with open(FIXTURES_DIR / "sample.md", "rb") as f:
        doc_content = f.read()

    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("security.md", doc_content, "text/markdown")},
    )
    assert upload_res.status_code == 201

    res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "What is the penalty fee if corporate income tax is submitted late?",
            "similarity_threshold": 0.50,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["confidence"] == "none"
    assert data["reason"] == "INSUFFICIENT_EVIDENCE"
    assert len(data["sources"]) == 0
    assert data["reliability"]["status"] == "REFUSE"
    assert data["reliability"]["supported"] is False


@pytest.mark.asyncio
async def test_adversarial_numeric_hallucination_detected_and_refused(
    async_client: AsyncClient, monkeypatch
):
    """
    Adversarial Scenario 3: Model is tempted to guess or distorts numbers.
    Context says 7 years, but adversarial LLM generates 3 years.
    AnswerVerifier catches the numeric discrepancy and refuses publication.
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

    # Inject an adversarial LLM provider that outputs hallucinated numbers
    hallucinating_provider = MockLLMProvider(
        custom_response="Customer personal identification data must be retained for 3 years."
    )
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_provider",
        lambda *args, **kwargs: hallucinating_provider,
    )
    monkeypatch.setattr(settings, "RAG_MAX_REGENERATION_ATTEMPTS", 0)

    res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "How long must customer personal identification data be retained?",
            "similarity_threshold": 0.30,
        },
    )
    assert res.status_code == 200
    data = res.json()

    # The system must prefer "I don't have enough reliable information to answer this." over the plausible hallucination
    assert data["answer"] == settings.RAG_UNVERIFIED_REFUSAL_RESPONSE
    assert data["confidence"] == "none"
    assert data["reason"] == "UNVERIFIED_CLAIMS"
    assert len(data["sources"]) == 0

    # Verification report must be visible and explain why it was rejected
    rel = data["reliability"]
    assert rel is not None
    assert rel["status"] == "REFUSE"
    assert rel["supported"] is False
    assert rel["faithfulness_score"] < 0.5
    assert len(rel["unsupported_claims"]) >= 1
    assert any("3" in c for c in rel["unsupported_claims"])
    assert "unverified claims" in rel["reason"].lower()


@pytest.mark.asyncio
async def test_adversarial_unsupported_concept_refused(
    async_client: AsyncClient, monkeypatch
):
    """
    Adversarial Scenario 4: LLM outputs an ungrounded fabricated policy.
    AnswerVerifier catches unsupported concepts and refuses.
    """
    with open(FIXTURES_DIR / "sample.txt", "rb") as f:
        doc_content = f.read()

    upload_res = await async_client.post(
        "/api/v1/documents",
        files={"file": ("operations.txt", doc_content, "text/plain")},
    )
    assert upload_res.status_code == 201

    fabricated_answer = (
        "Production deployments are permitted Tuesday through Thursday. "
        "In addition, all engineers receive a complimentary 5000 dollar annual travel stipend."
    )
    hallucinating_provider = MockLLMProvider(custom_response=fabricated_answer)
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_provider",
        lambda *args, **kwargs: hallucinating_provider,
    )
    monkeypatch.setattr(settings, "RAG_MAX_REGENERATION_ATTEMPTS", 0)

    res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "When are production deployments permitted?",
            "similarity_threshold": 0.20,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert data["answer"] == settings.RAG_UNVERIFIED_REFUSAL_RESPONSE
    assert data["reason"] == "UNVERIFIED_CLAIMS"
    assert len(data["sources"]) == 0
    assert data["reliability"]["status"] == "REFUSE"
    assert data["reliability"]["supported"] is False
    assert len(data["reliability"]["unsupported_claims"]) >= 1


@pytest.mark.asyncio
async def test_valid_grounded_answer_is_accepted_with_reliability_report(
    async_client: AsyncClient,
):
    """
    Valid Scenario: A fully grounded query and answer.
    Verifies that supported answers are ACCEPTED with faithfulness score 1.0,
    valid sources, and complete verification report.
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

    res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "How long must customer personal identification data be retained?",
            "similarity_threshold": 0.30,
        },
    )
    assert res.status_code == 200
    data = res.json()

    assert "7 years" in data["answer"] or "retained" in data["answer"].lower()
    assert data["confidence"] in ("high", "medium")
    assert data["reason"] is None
    assert len(data["sources"]) >= 1

    # Verification report checks
    rel = data["reliability"]
    assert rel is not None
    assert rel["status"] == "ACCEPT"
    assert rel["supported"] is True
    assert rel["faithfulness_score"] == 1.0
    assert rel["unsupported_claims"] == []
    assert "grounded in" in rel["reason"].lower()


@pytest.mark.asyncio
async def test_regeneration_recovery_workflow(
    async_client: AsyncClient, monkeypatch
):
    """
    Regeneration Scenario: Initial answer hallucinates, but corrective generation recovers.
    """
    with open(FIXTURES_DIR / "sample.txt", "rb") as f:
        doc_content = f.read()

    await async_client.post(
        "/api/v1/documents",
        files={"file": ("operations.txt", doc_content, "text/plain")},
    )

    class MultiAttemptProvider(BaseLLMProvider):
        def __init__(self):
            self.call_count = 0

        @property
        def provider_name(self) -> str:
            return "mock-multi"

        @property
        def model_name(self) -> str:
            return "mock-multi-v1"

        async def agenerate(self, prompt: str, system_prompt=None, **kwargs) -> LLMResponse:
            self.call_count += 1
            if self.call_count == 1:
                # First attempt: hallucinated number
                content = "Production deployments are permitted for 10 hours on Sundays."
            else:
                # Corrective attempt: grounded
                content = "Production deployments are permitted Tuesday through Thursday."
            return LLMResponse(
                content=content,
                model_name="mock-multi-v1",
                token_usage={"total_tokens": 10},
                raw_response={},
            )

        def generate(self, prompt: str, system_prompt=None, **kwargs) -> LLMResponse:
            raise NotImplementedError

    multi_provider = MultiAttemptProvider()
    monkeypatch.setattr(
        "app.services.rag_service.get_llm_provider",
        lambda *args, **kwargs: multi_provider,
    )
    monkeypatch.setattr(settings, "RAG_MAX_REGENERATION_ATTEMPTS", 1)
    monkeypatch.setattr(settings, "RAG_ENABLE_REGENERATION", True)

    res = await async_client.post(
        "/api/v1/ask",
        json={
            "question": "When are production deployments permitted?",
            "similarity_threshold": 0.20,
        },
    )
    assert res.status_code == 200
    data = res.json()

    # Should have executed 2 calls (initial + corrective retry)
    assert multi_provider.call_count == 2
    # Recovered answer accepted!
    assert data["reliability"]["status"] == "ACCEPT"
    assert data["reliability"]["supported"] is True
    assert "Tuesday through Thursday" in data["answer"]
    assert len(data["sources"]) >= 1
