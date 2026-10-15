from typing import List, Optional
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.base import RetrievedChunk
from app.schemas.ask import (
    AskResponse,
    AskSource,
    AskSourceChunk,
    AskSourceDocument,
)

logger = get_logger(__name__)


class ResponseFormattingService:
    """
    Service responsible for converting pipeline outputs into standard, structured API response contracts.
    Maps evidence chunks to attributed source documents and preserves relevance scoring.
    """

    def format_response(
        self,
        question: str,
        answer: str,
        evidence_chunks: List[RetrievedChunk],
        model_name: Optional[str] = None,
    ) -> AskResponse:
        """
        Formats question, synthesized answer, and retrieved sources into an AskResponse.
        """
        cleaned_answer = answer.strip()
        # If the generated answer explicitly indicates that evidence was insufficient/unavailable,
        # avoid citing irrelevant retrieved chunks as sources.
        is_refusal = (
            settings.RAG_FALLBACK_RESPONSE.lower() in cleaned_answer.lower()
            or "not have enough information" in cleaned_answer.lower()
            or "information is unavailable" in cleaned_answer.lower()
        )
        if is_refusal:
            return AskResponse(
                question=question,
                answer=cleaned_answer,
                sources=[],
                model_name=model_name,
            )

        sources: List[AskSource] = []

        for chunk in evidence_chunks:
            meta = chunk.metadata or {}
            document = AskSourceDocument(
                document_id=chunk.document_id,
                title=meta.get("document_title"),
                filename=meta.get("filename", "unknown"),
                document_type=meta.get("document_type", "unknown"),
                source=meta.get("source", "upload"),
                version=meta.get("version", "1.0"),
            )

            source_chunk = AskSourceChunk(
                chunk_id=chunk.chunk_id,
                chunk_index=meta.get("chunk_index", 0),
                content=chunk.content,
                metadata={
                    k: v
                    for k, v in meta.items()
                    if k
                    not in (
                        "document_title",
                        "filename",
                        "document_type",
                        "source",
                        "version",
                    )
                },
            )

            score = chunk.score if chunk.score is not None else 0.0

            sources.append(
                AskSource(
                    document=document,
                    chunk=source_chunk,
                    relevance_score=score,
                )
            )

        return AskResponse(
            question=question,
            answer=answer.strip(),
            sources=sources,
            model_name=model_name,
        )

    def format_no_context_response(
        self,
        question: str,
        model_name: Optional[str] = None,
        custom_message: Optional[str] = None,
    ) -> AskResponse:
        """
        Formats standard fallback response when no relevant context was found in the knowledge base.
        """
        message = custom_message or settings.RAG_FALLBACK_RESPONSE
        return AskResponse(
            question=question,
            answer=message,
            sources=[],
            model_name=model_name,
        )
