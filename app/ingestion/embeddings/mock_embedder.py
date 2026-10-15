import hashlib
import math
from typing import List, Optional
from app.ingestion.base import BaseEmbedder


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for lightning-fast testing and offline CI.
    Generates normalized dense vectors based on text token hashing.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "mock-deterministic-embedder"

    @property
    def embedding_dimension(self) -> int:
        return self._dimension

    def _generate_vector(self, text: str) -> List[float]:
        vector = [0.0] * self._dimension
        words = text.lower().split()
        if not words:
            return vector

        for word in words:
            # Deterministic bucket based on md5
            digest = hashlib.md5(word.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self._dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._generate_vector(t) for t in texts]

    async def embed_query(self, text: str) -> List[float]:
        return self._generate_vector(text)
