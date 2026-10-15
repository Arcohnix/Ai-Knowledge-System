from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings
from app.core.exceptions import ServiceUnavailableError, ValidationException
from app.core.logging import get_logger
from app.ingestion.base import BaseEmbedder

logger = get_logger(__name__)


class APIEmbedder(BaseEmbedder):
    """
    API-based embedding provider supporting OpenAI or any OpenAI-compatible endpoint
    (e.g., Azure OpenAI, LiteLLM, Ollama /v1/embeddings, LocalAI).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        self._model_name = model_name or "text-embedding-3-small"
        self._dimension = dimension or 1536
        self._api_key = api_key or settings.EMBEDDING_API_KEY
        self._base_url = (base_url or settings.EMBEDDING_API_BASE_URL).rstrip("/")

    @property
    def provider_name(self) -> str:
        return "api"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        if not self._api_key and "localhost" not in self._base_url:
            raise ValidationException(
                "EMBEDDING_API_KEY is required for hosted API-based embeddings."
            )

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: Dict[str, Any] = {
            "model": self._model_name,
            "input": texts,
        }

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self._base_url}/embeddings",
                    json=payload,
                    headers=headers,
                )
                if response.status_code != 200:
                    raise ServiceUnavailableError(
                        f"Embedding API error: {response.status_code} - {response.text}"
                    )
                data = response.json()
                sorted_items = sorted(data["data"], key=lambda x: x["index"])
                return [item["embedding"] for item in sorted_items]
        except httpx.RequestError as exc:
            raise ServiceUnavailableError(f"Failed to communicate with Embedding API: {exc}")

    async def embed_query(self, text: str) -> List[float]:
        results = await self.embed_documents([text])
        return results[0]
