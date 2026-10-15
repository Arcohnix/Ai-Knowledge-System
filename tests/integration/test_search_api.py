import pytest
from httpx import AsyncClient


@pytest.fixture(autouse=True)
def setup_search_embedder(monkeypatch):
    """Ensure FastEmbed is used consistently across search integration tests."""
    monkeypatch.setenv("EMBEDDING_PROVIDER", "local")


@pytest.mark.asyncio
async def test_semantic_search_end_to_end(async_client: AsyncClient):
    # 1. Ingest Realistic Knowledge Documents
    doc1_content = (
        b"Zero Trust Security Policy.\n\n"
        b"All employees must authenticate using multi-factor authentication (MFA) "
        b"with hardware security keys when accessing internal production infrastructure."
    )
    doc2_content = (
        b"# Data Retention and Compliance Standards\n\n"
        b"Customer financial transaction logs and tax records must be safely preserved "
        b"in immutable storage for a minimum duration of 7 years."
    )
    doc3_content = (
        b"Company Vacation and Leave Policy.\n\n"
        b"Full-time staff receive 25 days of paid annual leave, which must be approved "
        b"by their departmental lead at least two weeks in advance."
    )

    # Upload Doc 1 (Security)
    res1 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("security_policy.txt", doc1_content, "text/plain")},
    )
    assert res1.status_code == 201
    doc1_id = res1.json()["document_id"]

    # Upload Doc 2 (Retention)
    res2 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("retention_policy.md", doc2_content, "text/markdown")},
    )
    assert res2.status_code == 201
    doc2_id = res2.json()["document_id"]

    # Upload Doc 3 (Vacation)
    res3 = await async_client.post(
        "/api/v1/documents",
        files={"file": ("leave_policy.txt", doc3_content, "text/plain")},
    )
    assert res3.status_code == 201
    doc3_id = res3.json()["document_id"]

    # 2. Test Vector Persistence: Verify chunk in DB has embedding
    detail_res = await async_client.get(f"/api/v1/documents/{doc1_id}")
    assert detail_res.status_code == 200
    doc1_detail = detail_res.json()
    assert doc1_detail["chunk_count"] >= 1
    assert doc1_detail["metadata"]["embedding_provider"] == "fastembed"
    assert doc1_detail["metadata"]["embedding_dimension"] == 384

    # 3. Test Semantic Search Query 1: Security & MFA
    search_payload_1 = {
        "query": "How do employees authenticate with multi-factor MFA security keys?",
        "top_k": 3,
        "similarity_threshold": 0.3,
    }
    search_res_1 = await async_client.post("/api/v1/search", json=search_payload_1)
    assert search_res_1.status_code == 200
    search_data_1 = search_res_1.json()
    assert search_data_1["total_results"] >= 1
    top_result_1 = search_data_1["results"][0]
    assert "multi-factor authentication" in top_result_1["content"].lower()
    assert top_result_1["document_id"] == doc1_id
    assert top_result_1["similarity_score"] > 0.4
    assert top_result_1["document"]["title"] is not None

    # 4. Test Semantic Search Query 2: Financial retention duration
    search_payload_2 = {
        "query": "How many years should customer financial transaction records be preserved?",
        "top_k": 3,
        "similarity_threshold": 0.3,
    }
    search_res_2 = await async_client.post("/api/v1/search", json=search_payload_2)
    assert search_res_2.status_code == 200
    search_data_2 = search_res_2.json()
    assert search_data_2["total_results"] >= 1
    top_result_2 = search_data_2["results"][0]
    assert "7 years" in top_result_2["content"].lower()
    assert top_result_2["document_id"] == doc2_id

    # 5. Test top_k parameter enforcement
    search_top1 = await async_client.post(
        "/api/v1/search",
        json={"query": "policy guidelines", "top_k": 1, "similarity_threshold": 0.0},
    )
    assert search_top1.status_code == 200
    assert len(search_top1.json()["results"]) == 1

    search_top2 = await async_client.post(
        "/api/v1/search",
        json={"query": "policy guidelines", "top_k": 2, "similarity_threshold": 0.0},
    )
    assert search_top2.status_code == 200
    assert len(search_top2.json()["results"]) == 2

    # 6. Test similarity threshold filtering
    # Unrealistically high threshold should filter out results
    search_high_thresh = await async_client.post(
        "/api/v1/search",
        json={"query": "totally unrelated quantum physics", "similarity_threshold": 0.95},
    )
    assert search_high_thresh.status_code == 200
    assert search_high_thresh.json()["total_results"] == 0

    # 7. Test metadata filtering by document_type
    search_filter_type = await async_client.post(
        "/api/v1/search",
        json={
            "query": "standards and records",
            "top_k": 5,
            "similarity_threshold": 0.0,
            "filters": {"document_type": "markdown"},
        },
    )
    assert search_filter_type.status_code == 200
    results = search_filter_type.json()["results"]
    assert len(results) >= 1
    for item in results:
        assert item["document"]["document_type"] == "markdown"

    # 8. Test metadata filtering by specific document_id
    search_filter_id = await async_client.post(
        "/api/v1/search",
        json={
            "query": "authentication records",
            "top_k": 5,
            "similarity_threshold": 0.0,
            "filters": {"document_id": doc3_id},
        },
    )
    assert search_filter_id.status_code == 200
    results_id = search_filter_id.json()["results"]
    assert len(results_id) == 1
    assert results_id[0]["document_id"] == doc3_id


@pytest.mark.asyncio
async def test_search_validation_errors(async_client: AsyncClient):
    # Empty query string
    res = await async_client.post("/api/v1/search", json={"query": "   "})
    assert res.status_code == 422

    # Negative top_k
    res_k = await async_client.post("/api/v1/search", json={"query": "hello", "top_k": -1})
    assert res_k.status_code == 422
