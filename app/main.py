from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.database import close_database_connections, ping_database
from app.core.exceptions import register_exception_handlers
from app.core.logging import get_logger, setup_logging
from app.observability.telemetry import RequestContextMiddleware

# Initialize logging before application starts
setup_logging()
logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifecycle management (startup and shutdown).
    """
    # Startup sequence
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION} [{settings.ENVIRONMENT}]")
    
    # Check DB connectivity on startup
    db_ok, latency, error = await ping_database()
    if db_ok:
        logger.info(f"Database connection verified successfully (latency: {latency}ms)")
    else:
        logger.warning(f"Database connection not ready on startup: {error}")

    yield

    # Shutdown sequence
    logger.info("Initiating graceful application shutdown...")
    await close_database_connections()
    logger.info("Application shutdown complete.")


def create_application() -> FastAPI:
    """FastAPI application factory."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Production-grade AI Knowledge System backend with reliable document retrieval.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Register CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register Telemetry & Request Context Middleware
    app.add_middleware(RequestContextMiddleware)

    # Register Structured Exception Handlers
    register_exception_handlers(app)

    # Mount API Routers
    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    @app.get("/dashboard", tags=["Dashboard"], summary="Enterprise Observability & Admin Dashboard")
    async def get_dashboard():
        """Serves the standalone Enterprise Observability & Admin Dashboard."""
        dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
        return FileResponse(dashboard_path, media_type="text/html")

    @app.get("/", tags=["Root"], summary="Root service discovery")
    async def root():
        """Returns fundamental service info and endpoints."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "environment": settings.ENVIRONMENT,
            "docs": "/docs",
            "health": f"{settings.API_V1_PREFIX}/health",
            "dashboard": "/dashboard",
        }

    return app


app = create_application()
