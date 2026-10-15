import pytest
from unittest.mock import AsyncMock, patch
from app.ingestion.embeddings import (
    APIEmbedder,
    FastEmbedEmbedder,
    MockEmbedder,
    get_embedder,
)


@pytest.mark.asyncio
async def test_mock_embedder_deterministic_generation():
    embedder = MockEmbedder(dimension=384)
    assert embedder.embedding_dimension == 384
    assert embedder.provider_name == "mock"

    vec1 = await embedder.embed_query("Multi-factor authentication policy")
    vec2 = await embedder.embed_query("Multi-factor authentication policy")
    vec3 = await embedder.embed_query("Different topic entirely")

    assert len(vec1) == 384
    assert vec1 == vec2  # Deterministic
    assert vec1 != vec3  # Different text yields different vector


@pytest.mark.asyncio
async def test_fastembed_embedder_generation():
    embedder = FastEmbedEmbedder(model_name="BAAI/bge-small-en-v1.5", dimension=384)
    assert embedder.embedding_dimension == 384
    assert embedder.provider_name == "fastembed"

    texts = [
        "Company data security policy requires encryption at rest.",
        "Employees must complete compliance training annually.",
    ]
    embeddings = await embedder.embed_documents(texts)
    assert len(embeddings) == 2
    assert len(embeddings[0]) == 384
    assert len(embeddings[1]) == 384

    query_vec = await embedder.embed_query("data security encryption")
    assert len(query_vec) == 384


@pytest.mark.asyncio
async def test_api_embedder_with_mocked_http():
    embedder = APIEmbedder(
        model_name="text-embedding-3-small",
        dimension=1536,
        api_key="test-api-key",
        base_url="https://api.openai.com/v1",
    )
    assert embedder.embedding_dimension == 1536
    assert embedder.provider_name == "api"

    mock_response = {
        "data": [
            {"index": 0, "embedding": [0.05] * 1536},
            {"index": 1, "embedding": [0.10] * 1536},
        ]
    }

    with patch("httpx.AsyncClient.post") as mock_post:
        mock_resp = AsyncMock()
        mock_resp.status_code = 200
        mock_resp.json = lambda: mock_response
        mock_post.return_value = mock_resp

        results = await embedder.embed_documents(["First text", "Second text"])
        assert len(results) == 2
        assert len(results[0]) == 1536
        assert results[0][0] == 0.05


def test_embedder_factory_resolution():
    mock_emb = get_embedder("mock")
    assert isinstance(mock_emb, MockEmbedder)

    local_emb = get_embedder("local")
    assert isinstance(local_emb, FastEmbedEmbedder)

    api_emb = get_embedder("api")
    assert isinstance(api_emb, APIEmbedder)

    with pytest.raises(ValueError):
        get_embedder("unsupported_provider_xyz")
