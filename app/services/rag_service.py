from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.generation.base import BaseLLMProvider
from app.generation.factory import get_llm_provider
from app.ingestion.base import BaseEmbedder
from app.schemas.ask import AskResponse
from app.services.context_service import ContextConstructionService
from app.services.generation_service import LLMGenerationService
from app.services.query_service import QueryProcessingService
from app.services.response_formatting_service import ResponseFormattingService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RAGService:
    """
    Coordinator service orchestrating the end-to-end question answering pipeline:
    Query Processing -> Evidence Retrieval -> Context Construction -> LLM Generation -> Response Formatting.
    Keeps API controllers lightweight and decoupled.
    """

    def __init__(
        self,
        session: AsyncSession,
        query_processor: Optional[QueryProcessingService] = None,
        retrieval_service: Optional[RetrievalService] = None,
        context_builder: Optional[ContextConstructionService] = None,
        generation_service: Optional[LLMGenerationService] = None,
        response_formatter: Optional[ResponseFormattingService] = None,
        embedder: Optional[BaseEmbedder] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
    ):
        self.session = session
        self.query_processor = query_processor or QueryProcessingService()
        self.retrieval_service = retrieval_service or RetrievalService(
            session=session, embedder=embedder
        )
        self.context_builder = context_builder or ContextConstructionService()
        self.llm_provider = llm_provider or get_llm_provider()
        self.generation_service = generation_service or LLMGenerationService(
            provider=self.llm_provider
        )
        self.response_formatter = (
            response_formatter or ResponseFormattingService()
        )

    async def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        """
        Executes the full RAG pipeline for a given user question.
        """
        k = top_k or settings.RAG_DEFAULT_TOP_K
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.RAG_DEFAULT_SIMILARITY_THRESHOLD
        )

        logger.info(
            f"Beginning RAG pipeline: question='{question[:60]}...', top_k={k}, threshold={threshold}"
        )

        # Step 1: Query Processing
        processed_query = self.query_processor.process(question)

        # Step 2: Evidence Retrieval
        retrieved_chunks = await self.retrieval_service.search(
            query=processed_query.cleaned_query,
            top_k=k,
            similarity_threshold=threshold,
            filters=filters,
        )

        # If no relevant chunks meet the similarity threshold
        if not retrieved_chunks:
            logger.info("No evidence chunks found meeting similarity criteria; returning fallback response.")
            return self.response_formatter.format_no_context_response(
                question=processed_query.cleaned_query,
                model_name=self.llm_provider.model_name,
            )

        # Step 3: Context Construction
        context = self.context_builder.build_context(retrieved_chunks)

        if not context.has_context:
            logger.info("Context builder produced empty context; returning fallback response.")
            return self.response_formatter.format_no_context_response(
                question=processed_query.cleaned_query,
                model_name=self.llm_provider.model_name,
            )

        # Step 4: LLM Generation
        llm_response = await self.generation_service.generate_answer(
            question=processed_query.cleaned_query,
            context=context,
        )

        # Step 5: Response Formatting
        return self.response_formatter.format_response(
            question=processed_query.cleaned_query,
            answer=llm_response.content,
            evidence_chunks=context.included_chunks,
            model_name=llm_response.model_name,
        )
