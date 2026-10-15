import asyncio
from typing import List, Optional
from fastembed import TextEmbedding
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.base import BaseEmbedder

logger = get_logger(__name__)


class FastEmbedEmbedder(BaseEmbedder):
    """
    Local embedding provider utilizing fastembed (ONNX-accelerated BGE models).
    Lightweight, fast, and executes locally inside Docker without heavy PyTorch dependencies.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
        batch_size: Optional[int] = None,
    ):
        self._model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self._dimension = dimension or settings.EMBEDDING_DIMENSION
        self._batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        self._model: Optional[TextEmbedding] = None

    @property
    def provider_name(self) -> str:
        return "fastembed"

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def _get_model(self) -> TextEmbedding:
        if self._model is None:
            logger.info(f"Initializing FastEmbed model: {self._model_name}")
            self._model = TextEmbedding(model_name=self._model_name)
        return self._model

    def _embed_sync(self, texts: List[str]) -> List[List[float]]:
        model = self._get_model()
        embeddings_generator = model.embed(texts, batch_size=self._batch_size)
        return [arr.tolist() for arr in embeddings_generator]

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of document chunk texts."""
        if not texts:
            return []
        # Execute compute in threadpool to keep async loop responsive
        return await asyncio.to_thread(self._embed_sync, texts)

    async def embed_query(self, text: str) -> List[float]:
        """Asynchronously embed a single query string."""
        results = await self.embed_documents([text])
        return results[0]
