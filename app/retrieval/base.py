from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    """Represents a text chunk retrieved from the vector store."""

    chunk_id: str
    document_id: str
    content: str
    score: Optional[float] = Field(None, description="Similarity or relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseVectorStore(ABC):
    """
    Abstract interface for Vector Store backends.
    Allows swappable implementations (e.g., PGVector, Chroma, Qdrant, Pinecone).
    """

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Name or type of vector store engine."""
        pass

    @abstractmethod
    async def add_chunks(self, chunks: List[RetrievedChunk], embeddings: List[List[float]]) -> List[str]:
        """Store document chunks with their corresponding dense embeddings."""
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedChunk]:
        """Retrieve most similar chunks to the query vector."""
        pass

    @abstractmethod
    async def delete_by_document_id(self, document_id: str) -> bool:
        """Purge all vector embeddings and chunks associated with a document ID."""
        pass


class BaseRetriever(ABC):
    """
    High-level retriever interface that coordinates embedding and retrieval.
    Can support dense, sparse (hybrid), or re-ranked retrieval.
    """

    @abstractmethod
    async def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedChunk]:
        """Retrieve the most relevant context chunks for a user query."""
        pass
