import pytest
from httpx import AsyncClient
from app.models.document import DocumentChunkModel, DocumentModel
from app.models.trace import QueryTraceModel


@pytest.mark.asyncio
async def test_admin_overview_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/admin/overview")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "total_chunks" in data
    assert "total_questions" in data
    assert "acceptance_rate" in data
    assert "refusal_rate" in data
    assert "average_retrieval_score" in data
    assert "average_faithfulness_score" in data
    assert "recent_system_errors" in data


@pytest.mark.asyncio
async def test_admin_queries_and_trace_endpoints(async_client: AsyncClient, test_db_session):
    # Seed a trace
    trace = QueryTraceModel(
        request_id="trace-test-uuid-99",
        question="How do we deploy to staging?",
        cleaned_query="how do we deploy to staging",
        top_k=5,
        similarity_threshold=0.5,
        retrieval_count=1,
        top_similarity_score=0.91,
        mean_similarity_score=0.91,
        is_sufficient=True,
        final_decision="ACCEPTED",
        faithfulness_score=1.0,
        supported=True,
        citations=[{"document": "deploy.md", "chunk_id": "c-1", "score": 0.91}],
        trace_details={"pipeline_stages": [{"stage": "test_stage", "status": "ok"}]},
        generated_answer="Use git push to deploy.",
        latency_ms=120.5,
    )
    test_db_session.add(trace)
    await test_db_session.commit()

    # List queries
    list_resp = await async_client.get("/api/v1/admin/queries")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(q["request_id"] == "trace-test-uuid-99" for q in list_data["items"])

    # Query with decision filter
    filter_resp = await async_client.get("/api/v1/admin/queries?decision=ACCEPTED")
    assert filter_resp.status_code == 200
    assert any(q["request_id"] == "trace-test-uuid-99" for q in filter_resp.json()["items"])

    # Query trace detail by request_id
    detail_resp = await async_client.get("/api/v1/admin/queries/trace-test-uuid-99")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["request_id"] == "trace-test-uuid-99"
    assert detail["question"] == "How do we deploy to staging?"
    assert detail["final_decision"] == "ACCEPTED"
    assert len(detail["pipeline_stages"]) == 1

    # Query trace detail 404 for unknown request_id
    missing_resp = await async_client.get("/api/v1/admin/queries/does-not-exist")
    assert missing_resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_retrieval_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/admin/retrieval")
    assert response.status_code == 200
    data = response.json()
    assert "current_retrieval_threshold" in data
    assert "score_distribution" in data
    assert "top_retrieved_documents" in data


@pytest.mark.asyncio
async def test_admin_evaluation_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/admin/evaluation")
    assert response.status_code == 200
    data = response.json()
    assert "total_benchmark_runs" in data
    assert "runs" in data


@pytest.mark.asyncio
async def test_admin_knowledge_base_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/admin/knowledge-base")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data


@pytest.mark.asyncio
async def test_admin_reliability_endpoint(async_client: AsyncClient):
    response = await async_client.get("/api/v1/admin/reliability")
    assert response.status_code == 200
    data = response.json()
    assert "total_queries_evaluated" in data
    assert "insufficient_evidence_cases" in data
    assert "contradiction_cases" in data
    assert "hallucinations_detected_count" in data
    assert "unsupported_claims_log" in data
    assert "refusal_reasons_breakdown" in data


@pytest.mark.asyncio
async def test_dashboard_ui_endpoint(async_client: AsyncClient):
    response = await async_client.get("/dashboard")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "AI Knowledge System" in response.text
    assert "Control Plane & Observability" in response.text


@pytest.mark.asyncio
async def test_end_to_end_query_creates_trace_record(async_client: AsyncClient):
    # Ingest a simple question with no retrieved documents to trigger controlled refusal
    ask_resp = await async_client.post(
        "/api/v1/ask",
        json={"question": "What is the secret recipe for Martian stew?"},
    )
    assert ask_resp.status_code == 200
    ask_data = ask_resp.json()
    assert ask_data["reason"] == "INSUFFICIENT_EVIDENCE"

    # Verify a query trace record was created in the database and visible via admin queries
    queries_resp = await async_client.get("/api/v1/admin/queries")
    assert queries_resp.status_code == 200
    queries_data = queries_resp.json()
    assert queries_data["total"] >= 1
    recent_query = queries_data["items"][0]
    assert "Martian stew" in recent_query["question"]
    assert recent_query["final_decision"] == "REFUSED"
