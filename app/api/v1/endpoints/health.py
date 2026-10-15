from datetime import datetime, timezone
from fastapi import APIRouter, Response, status
from app.schemas.health import HealthCheckResponse
from app.services.health_service import HealthService

router = APIRouter()


@router.get(
    "/health/live",
    summary="Liveness Probe",
    description="Container and orchestrator liveness probe confirming the web application process is active.",
    status_code=status.HTTP_200_OK,
)
async def check_liveness() -> dict:
    """Returns 200 OK if the application server is up and responsive."""
    return {"status": "alive", "timestamp": datetime.now(timezone.utc).isoformat()}


@router.get(
    "/health/ready",
    summary="Readiness Probe",
    description="Container and orchestrator readiness probe verifying critical dependencies (PostgreSQL database).",
    responses={
        200: {"description": "All dependencies healthy and ready for queries"},
        503: {"description": "One or more dependencies not ready"},
    },
)
async def check_readiness(response: Response) -> dict:
    """Verifies that the database and downstream services are connected and ready."""
    health_report = await HealthService.get_health_status()
    if health_report.status != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "not_ready",
            "database": health_report.components.database.status,
            "error": health_report.components.database.error,
        }
    return {
        "status": "ready",
        "database": "connected",
        "latency_ms": health_report.components.database.latency_ms,
    }


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    summary="System Health Check",
    description="Returns application health status, uptime, and database connectivity metrics.",
    responses={
        200: {"description": "All systems operational", "model": HealthCheckResponse},
        503: {"description": "One or more critical subsystems unhealthy", "model": HealthCheckResponse},
    },
)
async def check_health(response: Response) -> HealthCheckResponse:
    """
    Evaluates health of the application and its critical dependencies (PostgreSQL).
    Sets HTTP status to 503 if the database is unreachable, otherwise 200.
    """
    health_report = await HealthService.get_health_status()
    if health_report.status != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return health_report
