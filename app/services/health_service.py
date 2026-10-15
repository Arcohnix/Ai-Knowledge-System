import platform
import sys
import time
from datetime import datetime, timezone
from app.core.config import settings
from app.core.database import ping_database
from app.schemas.health import (
    DatabaseHealth,
    HealthCheckResponse,
    HealthComponents,
    SystemHealth,
)

# Application start time for uptime tracking
APP_START_TIME = time.time()


class HealthService:
    """Service to evaluate system component health status."""

    @staticmethod
    async def get_health_status() -> HealthCheckResponse:
        db_healthy, latency_ms, db_error = await ping_database()

        overall_status = "healthy" if db_healthy else "degraded"

        uptime_seconds = round(time.time() - APP_START_TIME, 2)

        return HealthCheckResponse(
            status=overall_status,
            app_name=settings.APP_NAME,
            version=settings.APP_VERSION,
            environment=settings.ENVIRONMENT,
            uptime_seconds=uptime_seconds,
            timestamp=datetime.now(timezone.utc),
            components=HealthComponents(
                database=DatabaseHealth(
                    status="healthy" if db_healthy else "unhealthy",
                    latency_ms=latency_ms if db_healthy else None,
                    error=db_error if not db_healthy else None,
                ),
                system=SystemHealth(
                    python_version=sys.version.split()[0],
                    platform=platform.platform(),
                ),
            ),
        )
