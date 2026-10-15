from fastapi import APIRouter, Response, status
from app.schemas.health import HealthCheckResponse
from app.services.health_service import HealthService

router = APIRouter()


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
