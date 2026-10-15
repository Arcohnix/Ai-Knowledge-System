from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, List, Optional, Union
from pydantic import BaseModel, Field
from app.retrieval.base import RetrievedChunk


class RawDocument(BaseModel):
    """Represents a document parsed from a source before chunking."""

    source: str = Field(..., description="Source path, URL, or identifier")
    content: str = Field(..., description="Extracted raw text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseDocumentParser(ABC):
    """Abstract parser interface for extracting clean text and metadata from files (PDF, DOCX, TXT, MD)."""

    @property
    @abstractmethod
    def supported_mime_types(self) -> List[str]:
        """List of MIME types supported by this parser."""
        pass

    @abstractmethod
    async def parse(self, content: Union[bytes, BinaryIO], filename: str) -> List[RawDocument]:
        """Parse raw file bytes into one or more RawDocuments."""
        pass


class BaseChunker(ABC):
    """Abstract chunker for splitting documents into semantically coherent segments."""

    @abstractmethod
    def chunk(self, documents: List[RawDocument]) -> List[RetrievedChunk]:
        """Split raw documents into chunks ready for embedding and indexing."""
        pass


class BaseEmbedder(ABC):
    """Abstract interface for embedding models (e.g. HuggingFace, OpenAI, Ollama)."""

    @property
    @abstractmethod
    def embedding_dimension(self) -> int:
        """The dimensionality of generated embeddings."""
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate dense vector embeddings for a batch of text chunks."""
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Generate dense vector embedding for a single search query."""
        pass
