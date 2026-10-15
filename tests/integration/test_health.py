import pytest
from httpx import AsyncClient
from unittest.mock import patch


@pytest.mark.asyncio
async def test_health_endpoint_healthy(async_client: AsyncClient):
    """Verify GET /api/v1/health returns 200 with structured component data when DB is healthy."""
    with patch("app.services.health_service.ping_database", return_value=(True, 2.5, "")):
        response = await async_client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert data["app_name"] == "AI Knowledge System"
    assert "version" in data
    assert "environment" in data
    assert "uptime_seconds" in data
    assert "timestamp" in data

    components = data["components"]
    assert components["database"]["status"] == "healthy"
    assert components["database"]["latency_ms"] == 2.5
    assert components["system"]["python_version"] is not None


@pytest.mark.asyncio
async def test_health_endpoint_degraded(async_client: AsyncClient):
    """Verify GET /api/v1/health returns 503 when the database ping fails."""
    with patch(
        "app.services.health_service.ping_database",
        return_value=(False, 10.0, "Connection refused"),
    ):
        response = await async_client.get("/api/v1/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert data["components"]["database"]["status"] == "unhealthy"
    assert data["components"]["database"]["error"] == "Connection refused"


@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Verify GET / returns basic service discovery payload."""
    response = await async_client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI Knowledge System"
    assert data["docs"] == "/docs"
    assert data["health"] == "/api/v1/health"


@pytest.mark.asyncio
async def test_middleware_request_id_and_timing(async_client: AsyncClient):
    """Verify request context middleware attaches X-Request-ID and X-Process-Time headers."""
    response = await async_client.get("/")
    assert "x-request-id" in response.headers
    assert "x-process-time" in response.headers
    assert response.headers["x-process-time"].endswith("ms")

    # Custom caller-provided X-Request-ID should be preserved
    custom_id = "test-custom-req-id-12345"
    response_custom = await async_client.get("/", headers={"X-Request-ID": custom_id})
    assert response_custom.headers["x-request-id"] == custom_id


@pytest.mark.asyncio
async def test_structured_error_response_404(async_client: AsyncClient):
    """Verify unmapped routes return standardized error payload."""
    response = await async_client.get("/api/v1/non-existent-endpoint")
    assert response.status_code == 404
    data = response.json()

    assert data["success"] is False
    assert "error" in data
    assert data["error"]["code"] == "NOT_FOUND"
    assert "timestamp" in data
    assert "x-request-id" in response.headers
