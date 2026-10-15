from typing import List, Optional
from pydantic import BaseModel, Field
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.base import RetrievedChunk

logger = get_logger(__name__)


class ConstructedContext(BaseModel):
    """Encapsulates formatted context documentation ready for prompt injection."""

    context_text: str = Field(..., description="Structured context text with document citations")
    included_chunks: List[RetrievedChunk] = Field(
        default_factory=list, description="Chunks included within the token/character budget"
    )
    has_context: bool = Field(
        False, description="True if at least one evidence chunk was successfully formatted"
    )
    total_chars: int = Field(0, description="Total characters in the formatted context")


class ContextConstructionService:
    """
    Service responsible for assembling, budgeting, and structuring retrieved evidence chunks
    into a coherent context block for LLM prompt ingestion.
    """

    def __init__(self, max_context_chars: Optional[int] = None):
        self.max_context_chars = max_context_chars or settings.RAG_MAX_CONTEXT_CHARS

    def build_context(self, retrieved_chunks: List[RetrievedChunk]) -> ConstructedContext:
        """
        Formats retrieved chunks with numbered headers and enforces context budget.
        """
        if not retrieved_chunks:
            return ConstructedContext(
                context_text="No relevant documentation found.",
                included_chunks=[],
                has_context=False,
                total_chars=0,
            )

        formatted_blocks: List[str] = []
        included_chunks: List[RetrievedChunk] = []
        accumulated_chars = 0

        for idx, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.metadata or {}
            title = metadata.get("document_title") or metadata.get("filename") or "Document"
            filename = metadata.get("filename") or "unknown"
            chunk_index = metadata.get("chunk_index", 0)
            score_str = f" | Relevance: {chunk.score:.2f}" if chunk.score is not None else ""

            block = (
                f"[Document {idx}] Title: {title} | File: {filename} | Chunk: {chunk_index}{score_str}\n"
                f"{chunk.content.strip()}\n"
            )

            # Check character budget
            if accumulated_chars + len(block) > self.max_context_chars and included_chunks:
                logger.warning(
                    f"Context budget exceeded ({self.max_context_chars} chars). Truncating at chunk {idx - 1}."
                )
                break

            formatted_blocks.append(block)
            included_chunks.append(chunk)
            accumulated_chars += len(block)

        combined_context = "\n".join(formatted_blocks).strip()

        logger.debug(
            f"Context constructed: {len(included_chunks)}/{len(retrieved_chunks)} chunks included, {accumulated_chars} chars"
        )

        return ConstructedContext(
            context_text=combined_context,
            included_chunks=included_chunks,
            has_context=len(included_chunks) > 0,
            total_chars=accumulated_chars,
        )
