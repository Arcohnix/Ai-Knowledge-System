from typing import Optional
from app.core.config import settings
from app.ingestion.base import BaseEmbedder
from app.ingestion.embeddings.api_embedder import APIEmbedder
from app.ingestion.embeddings.local_embedder import FastEmbedEmbedder
from app.ingestion.embeddings.mock_embedder import MockEmbedder

_CACHED_EMBEDDER: Optional[BaseEmbedder] = None


def get_embedder(provider: Optional[str] = None) -> BaseEmbedder:
    """
    Factory function returning the configured embedder instance.
    Defaults to settings.EMBEDDING_PROVIDER ('local', 'api', 'mock').
    """
    global _CACHED_EMBEDDER
    prov = (provider or settings.EMBEDDING_PROVIDER).lower()

    if prov in ["mock", "test"]:
        return MockEmbedder(dimension=settings.EMBEDDING_DIMENSION)

    if prov in ["local", "fastembed"]:
        if _CACHED_EMBEDDER is None or _CACHED_EMBEDDER.provider_name != "fastembed":
            _CACHED_EMBEDDER = FastEmbedEmbedder(
                model_name=settings.EMBEDDING_MODEL_NAME,
                dimension=settings.EMBEDDING_DIMENSION,
            )
        return _CACHED_EMBEDDER

    if prov in ["openai", "api"]:
        return APIEmbedder(
            model_name=settings.EMBEDDING_MODEL_NAME,
            dimension=settings.EMBEDDING_DIMENSION,
            api_key=settings.EMBEDDING_API_KEY,
            base_url=settings.EMBEDDING_API_BASE_URL,
        )

    raise ValueError(f"Unknown embedding provider: '{prov}'. Supported: 'local', 'api', 'mock'.")
