from fastapi import APIRouter
from app.api.v1.endpoints import documents, health, search

api_v1_router = APIRouter()

api_v1_router.include_router(health.router, tags=["Health"])
api_v1_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_v1_router.include_router(search.router, prefix="/search", tags=["Search"])
