from datetime import datetime
from typing import Dict, Optional
from pydantic import BaseModel, Field


class DatabaseHealth(BaseModel):
    """Database connectivity details."""

    status: str = Field(..., description="'healthy' or 'unhealthy'")
    latency_ms: Optional[float] = Field(None, description="Roundtrip query latency in milliseconds")
    error: Optional[str] = Field(None, description="Error details if connection failed")


class SystemHealth(BaseModel):
    """Runtime system information."""

    python_version: str
    platform: str


class HealthComponents(BaseModel):
    """Health breakdown of subcomponents."""

    database: DatabaseHealth
    system: SystemHealth


class HealthCheckResponse(BaseModel):
    """Comprehensive health check response."""

    status: str = Field(..., description="'healthy', 'degraded', or 'unhealthy'")
    app_name: str
    version: str
    environment: str
    uptime_seconds: float
    timestamp: datetime
    components: HealthComponents
