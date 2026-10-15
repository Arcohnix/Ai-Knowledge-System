from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.evaluation.answer_verifier import AnswerVerifier
from app.evaluation.evidence_evaluator import EvidenceEvaluator
from app.evaluation.reliability_engine import DecisionAction, ReliabilityDecisionEngine
from app.generation.base import BaseLLMProvider
from app.generation.factory import get_llm_provider
from app.ingestion.base import BaseEmbedder
from app.schemas.ask import AskResponse, ReliabilityReport
from app.services.context_service import ContextConstructionService
from app.services.generation_service import LLMGenerationService
from app.services.query_service import QueryProcessingService
from app.services.response_formatting_service import ResponseFormattingService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RAGService:
    """
    Coordinator service orchestrating the multi-stage reliable question answering pipeline:
    Query Processing -> Evidence Retrieval -> Evidence Sufficiency Evaluation -> Context Construction
    -> LLM Generation -> Faithfulness Verification -> Reliability Decision (Accept/Regenerate/Refuse)
    -> Audited Response Formatting.
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
        evidence_evaluator: Optional[EvidenceEvaluator] = None,
        answer_verifier: Optional[AnswerVerifier] = None,
        reliability_engine: Optional[ReliabilityDecisionEngine] = None,
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
        self.evidence_evaluator = evidence_evaluator or EvidenceEvaluator()
        self.answer_verifier = answer_verifier or AnswerVerifier()
        self.reliability_engine = (
            reliability_engine or ReliabilityDecisionEngine()
        )

    async def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        """
        Executes the full RAG pipeline for a given user question with pre-generation
        evidence evaluation and post-generation faithfulness verification.
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

        # Step 3: Evidence Sufficiency Evaluation (Pre-generation check)
        sufficiency = self.evidence_evaluator.evaluate_sufficiency(
            retrieved_chunks=retrieved_chunks,
            query_keywords=processed_query.keywords,
            min_similarity_threshold=threshold,
        )

        if not sufficiency.is_sufficient:
            logger.info(
                f"Evidence sufficiency check failed: reason={sufficiency.reason}, details={sufficiency.details}"
            )
            reliability_report = ReliabilityReport(
                status=DecisionAction.REFUSE.value,
                faithfulness_score=0.0,
                supported=False,
                unsupported_claims=[],
                reason=f"Pre-generation check: {sufficiency.reason} ({sufficiency.details.get('failure_stage', 'inadequate_evidence')})",
            )
            return self.response_formatter.format_no_context_response(
                question=processed_query.cleaned_query,
                model_name=self.llm_provider.model_name,
                custom_message=settings.RAG_FALLBACK_RESPONSE,
                reason=sufficiency.reason or settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                reliability=reliability_report,
            )

        # Step 4: Context Construction
        context = self.context_builder.build_context(retrieved_chunks)

        if not context.has_context:
            logger.info("Context builder produced empty context; returning fallback response.")
            reliability_report = ReliabilityReport(
                status=DecisionAction.REFUSE.value,
                faithfulness_score=0.0,
                supported=False,
                unsupported_claims=[],
                reason="Context construction yielded empty context.",
            )
            return self.response_formatter.format_no_context_response(
                question=processed_query.cleaned_query,
                model_name=self.llm_provider.model_name,
                custom_message=settings.RAG_FALLBACK_RESPONSE,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                reliability=reliability_report,
            )

        # Step 5: LLM Generation (Initial Attempt)
        llm_response = await self.generation_service.generate_answer(
            question=processed_query.cleaned_query,
            context=context,
        )
        current_answer = llm_response.content

        # Step 6: Post-generation Faithfulness Verification & Reliability Decision
        verification = self.answer_verifier.verify_answer(
            answer=current_answer,
            context_text=context.context_text,
        )

        decision = self.reliability_engine.decide(
            verification=verification,
            current_answer=current_answer,
            regeneration_attempt=0,
        )

        # Step 7: Optional Regeneration if unsupported claims detected
        if decision.action == DecisionAction.REGENERATE:
            logger.warning(
                f"Reliability engine triggered REGENERATE. Prompting LLM with negative constraints. "
                f"Unsupported claims: {decision.unsupported_claims}"
            )
            regen_prompt = self.reliability_engine.build_regeneration_prompt(
                unsupported_claims=decision.unsupported_claims,
                question=processed_query.cleaned_query,
                context_text=context.context_text,
            )
            try:
                regen_response = await self.generation_service.generate_correction(regen_prompt)
                regen_answer = regen_response.content

                verification_retry = self.answer_verifier.verify_answer(
                    answer=regen_answer,
                    context_text=context.context_text,
                )
                decision = self.reliability_engine.decide(
                    verification=verification_retry,
                    current_answer=regen_answer,
                    regeneration_attempt=1,
                )
            except Exception as exc:
                logger.error(f"Regeneration attempt failed: {exc}. Falling back to REFUSE.")
                decision = self.reliability_engine.decide(
                    verification=verification,
                    current_answer=current_answer,
                    regeneration_attempt=settings.RAG_MAX_REGENERATION_ATTEMPTS,
                )

        reliability_report = ReliabilityReport(
            status=decision.action.value,
            faithfulness_score=decision.faithfulness_score,
            supported=decision.supported,
            unsupported_claims=decision.unsupported_claims,
            reason=decision.reason,
        )

        # Step 8: Response Formatting
        return self.response_formatter.format_response(
            question=processed_query.cleaned_query,
            answer=decision.final_answer,
            evidence_chunks=context.included_chunks,
            query_keywords=processed_query.keywords,
            model_name=llm_response.model_name,
            reliability=reliability_report,
        )
