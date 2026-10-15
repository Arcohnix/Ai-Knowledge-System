from datetime import date
from pathlib import Path
import pytest
from httpx import AsyncClient
from app.core.config import settings

FIXTURES_DIR = Path("tests/fixtures")


@pytest.fixture(autouse=True)
def setup_rag_test_env(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mock")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "local")


@pytest.mark.asyncio
async def test_concordant_documents_agree(async_client: AsyncClient):
    """
    Scenario 1: Same Information / Concordant Documents.
    Multiple ingested documents state matching facts without contradiction.
    """
    doc_1_content = b"Data Retention Standard. Customer personal identification records must be retained for at least 7 years."
    doc_2_content = b"Financial Compliance Guide. Customer personal identification records must be retained for at least 7 years pursuant to regulatory rules."

    res_1 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("retention_std.txt", doc_1_content, "text/plain")},
        data={"version": "1.0", "status": "ACTIVE"},
    )
    assert res_1.status_code == 201

    res_2 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("finance_guide.txt", doc_2_content, "text/plain")},
        data={"version": "1.0", "status": "ACTIVE"},
    )
    assert res_2.status_code == 201

    ask_payload = {
        "question": "How long must customer personal identification records be retained?",
        "similarity_threshold": 0.30,
        "top_k": 3,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    assert "7 years" in data["answer"]
    assert data["confidence"] in ("high", "medium")
    assert data["reason"] is None
    assert len(data["sources"]) >= 1

    # Contradiction report confirms agreement
    contra = data.get("contradiction")
    assert contra is not None
    assert contra["status"] == "AGREE"
    assert contra["has_conflict"] is False
    assert contra["resolution_status"] == "NO_CONFLICT"


@pytest.mark.asyncio
async def test_unresolved_direct_contradiction_between_active_documents(
    async_client: AsyncClient,
):
    """
    Scenario 2: Direct Contradiction Between Active Documents of Equal Precedence.
    Document A: 'Employees receive 20 annual vacation days.' (ACTIVE, v1.0)
    Document B: 'Employees receive 25 annual vacation days.' (ACTIVE, v1.0)
    The system must NOT blindly combine these statements or silently pick one based on vector score.
    It must inform the user and show the conflicting sources.
    """
    doc_a = b"General Employee Handbook. Employees receive 20 annual vacation days upon hiring."
    doc_b = b"Engineering Benefits Guide. Employees receive 25 annual vacation days upon hiring."

    res_a = await async_client.post(
        "/api/v1/documents",
        files={"file": ("handbook_a.txt", doc_a, "text/plain")},
        data={"version": "1.0", "status": "ACTIVE"},
    )
    assert res_a.status_code == 201

    res_b = await async_client.post(
        "/api/v1/documents",
        files={"file": ("handbook_b.txt", doc_b, "text/plain")},
        data={"version": "1.0", "status": "ACTIVE"},
    )
    assert res_b.status_code == 201

    ask_payload = {
        "question": "How many annual vacation days do employees receive?",
        "similarity_threshold": 0.25,
        "top_k": 4,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    # User receives explicit conflict notification
    assert data["answer"] == settings.RAG_CONFLICT_REFUSAL_RESPONSE
    assert (
        "I found conflicting information in the knowledge base and cannot determine which policy is currently authoritative."
        in data["answer"]
    )
    assert data["reason"] == "CONFLICTING_EVIDENCE"

    # Conflicting sources must be shown to the user
    assert len(data["sources"]) >= 2
    source_names = {s["document"] for s in data["sources"]}
    assert "handbook_a.txt" in source_names
    assert "handbook_b.txt" in source_names

    # Contradiction report details
    contra = data["contradiction"]
    assert contra is not None
    assert contra["status"] == "CONFLICT"
    assert contra["has_conflict"] is True
    assert contra["resolution_status"] == "UNRESOLVED"
    assert "handbook_a.txt" in contra["conflicting_sources"]
    assert "handbook_b.txt" in contra["conflicting_sources"]


@pytest.mark.asyncio
async def test_old_vs_new_policy_resolved_by_version_and_superseded_status(
    async_client: AsyncClient,
):
    """
    Scenario 3: Old vs New Policy.
    Old Policy: 20 vacation days (v2024.1, SUPERSEDED).
    New Policy: 25 vacation days (v2026.1, ACTIVE).
    The system resolves in favor of the active/newer policy and answers accurately.
    """
    doc_old = b"Corporate Vacation Policy 2024. Employees receive 20 annual vacation days upon hiring."
    doc_new = b"Corporate Vacation Policy 2026. Employees receive 25 annual vacation days upon hiring."

    res_old = await async_client.post(
        "/api/v1/documents",
        files={"file": ("vacation_2024.txt", doc_old, "text/plain")},
        data={"version": "2024.1", "status": "SUPERSEDED"},
    )
    assert res_old.status_code == 201

    res_new = await async_client.post(
        "/api/v1/documents",
        files={"file": ("vacation_2026.txt", doc_new, "text/plain")},
        data={"version": "2026.1", "status": "ACTIVE"},
    )
    assert res_new.status_code == 201

    ask_payload = {
        "question": "How many annual vacation days do employees receive?",
        "similarity_threshold": 0.25,
        "top_k": 4,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    # The active/newer document is authoritative
    assert "25" in data["answer"]
    assert data["confidence"] in ("high", "medium")
    assert data["reason"] is None

    # Source citations cite the active/newer document
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["document"] == "vacation_2026.txt"
    assert data["sources"][0]["version"] == "2026.1"

    # Contradiction report records successful resolution
    contra = data["contradiction"]
    assert contra is not None
    assert contra["status"] == "CONFLICT"
    assert contra["has_conflict"] is True
    assert contra["resolution_status"] in (
        "RESOLVED_BY_ACTIVE_STATUS",
        "RESOLVED_BY_NEWER_VERSION",
    )
    assert "vacation_2026.txt" in contra["resolution_rationale"]


@pytest.mark.asyncio
async def test_expired_vs_active_policy_resolved_by_effective_date(
    async_client: AsyncClient,
):
    """
    Scenario 4: Expired vs Active Policy.
    Expired Policy: Deployments permitted on Fridays (expired on 2022-12-31, EXPIRED).
    Active Policy: Deployments prohibited on Fridays (effective 2023-01-01, ACTIVE).
    The system resolves in favor of the active policy.
    """
    doc_exp = b"Engineering Deployments 2022. Production deployments are permitted on Fridays with manager approval."
    doc_act = b"Engineering Deployments 2025. Production deployments are strictly prohibited on Fridays under all circumstances."

    res_exp = await async_client.post(
        "/api/v1/documents",
        files={"file": ("deploy_2022.txt", doc_exp, "text/plain")},
        data={
            "version": "1.0",
            "status": "EXPIRED",
            "expiration_date": "2022-12-31",
        },
    )
    assert res_exp.status_code == 201

    res_act = await async_client.post(
        "/api/v1/documents",
        files={"file": ("deploy_2025.txt", doc_act, "text/plain")},
        data={
            "version": "2.0",
            "status": "ACTIVE",
            "effective_date": "2023-01-01",
        },
    )
    assert res_act.status_code == 201

    ask_payload = {
        "question": "Are production deployments permitted on Fridays?",
        "similarity_threshold": 0.25,
        "top_k": 4,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    # Active policy wins
    assert "prohibited" in data["answer"].lower()
    assert data["confidence"] in ("high", "medium")
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["document"] == "deploy_2025.txt"

    contra = data["contradiction"]
    assert contra is not None
    assert contra["status"] == "CONFLICT"
    assert contra["resolution_status"] in (
        "RESOLVED_BY_ACTIVE_STATUS",
        "RESOLVED_BY_EFFECTIVE_DATE",
        "RESOLVED_BY_NEWER_VERSION",
    )


@pytest.mark.asyncio
async def test_system_never_silently_picks_higher_vector_score_on_contradiction(
    async_client: AsyncClient,
):
    """
    Scenario 5: Vector Similarity Trap.
    An outdated superseded document contains heavy keyword repetition, yielding a higher
    vector score than the concise active document.
    The system must NOT pick the superseded document based on vector similarity alone.
    """
    # Heavily keyword-stuffed outdated policy
    doc_outdated = (
        b"Remote Work Policy 2020. Remote work policy guidelines: employees must work from office 5 days a week. "
        b"Remote work is strictly disallowed for all employees 5 days a week."
    )
    # Concise modern active policy
    doc_active = (
        b"Remote Work Policy 2026. Employees are permitted to work remotely up to 3 days a week."
    )

    await async_client.post(
        "/api/v1/documents",
        files={"file": ("remote_2020.txt", doc_outdated, "text/plain")},
        data={"version": "1.0", "status": "SUPERSEDED"},
    )
    await async_client.post(
        "/api/v1/documents",
        files={"file": ("remote_2026.txt", doc_active, "text/plain")},
        data={"version": "2.0", "status": "ACTIVE"},
    )

    ask_payload = {
        "question": "Can employees work remotely during the week?",
        "similarity_threshold": 0.20,
        "top_k": 4,
    }
    res = await async_client.post("/api/v1/ask", json=ask_payload)
    assert res.status_code == 200
    data = res.json()

    # The active policy (remote allowed 3 days) must prevail over the superseded 5 days in-office policy
    assert "remotely" in data["answer"].lower() or "3 days" in data["answer"].lower()
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["document"] == "remote_2026.txt"
    assert data["contradiction"]["resolution_status"] in (
        "RESOLVED_BY_ACTIVE_STATUS",
        "RESOLVED_BY_NEWER_VERSION",
    )
