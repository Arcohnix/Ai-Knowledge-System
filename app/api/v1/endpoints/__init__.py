from app.api.v1.endpoints.ask import router as ask_router
from app.api.v1.endpoints.documents import router as documents_router
from app.api.v1.endpoints.health import router as health_router
from app.api.v1.endpoints.search import router as search_router

__all__ = ["health_router", "documents_router", "search_router", "ask_router"]
