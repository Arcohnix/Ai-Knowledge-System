import time
from typing import Any, Dict, List, Optional
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.models.trace import QueryTraceModel
from app.evaluation.answer_verifier import AnswerVerifier
from app.evaluation.evidence_evaluator import EvidenceEvaluator
from app.evaluation.reliability_engine import DecisionAction, ReliabilityDecisionEngine
from app.generation.base import BaseLLMProvider
from app.generation.factory import get_llm_provider
from app.ingestion.base import BaseEmbedder
from app.schemas.ask import AskResponse, ReliabilityReport
from app.services.context_service import ContextConstructionService
from app.services.contradiction_service import (
    ContradictionService,
    ContradictionStatus,
)
from app.services.generation_service import LLMGenerationService
from app.services.query_service import QueryProcessingService
from app.services.response_formatting_service import ResponseFormattingService
from app.services.retrieval_service import RetrievalService

logger = get_logger(__name__)


class RAGService:
    """
    Coordinator service orchestrating the multi-stage reliable question answering pipeline:
    Query Processing -> Evidence Retrieval -> Evidence Sufficiency Evaluation
    -> Contradiction Analysis & Document Version Resolution -> Context Construction
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
        contradiction_service: Optional[ContradictionService] = None,
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
        self.contradiction_service = (
            contradiction_service or ContradictionService()
        )

    async def _save_trace(
        self,
        request_id: str,
        question: str,
        cleaned_query: str,
        top_k: int,
        similarity_threshold: float,
        retrieved_chunks: List[Any],
        sufficiency: Any,
        contradiction_result: Optional[Any],
        response: AskResponse,
        decision_action: str,
        refusal_reason: Optional[str],
        trace_details: Dict[str, Any],
        latency_ms: float,
    ) -> None:
        try:
            scores = [c.score for c in retrieved_chunks if c.score is not None]
            top_score = scores[0] if scores else None
            mean_score = sum(scores) / len(scores) if scores else None

            contra_status = (
                contradiction_result.status.value
                if contradiction_result and hasattr(contradiction_result, "status")
                else None
            )

            trace_record = QueryTraceModel(
                request_id=request_id,
                question=question,
                cleaned_query=cleaned_query,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                retrieval_count=len(retrieved_chunks),
                top_similarity_score=round(top_score, 4) if top_score is not None else None,
                mean_similarity_score=round(mean_score, 4) if mean_score is not None else None,
                is_sufficient=sufficiency.is_sufficient if sufficiency else False,
                sufficiency_reason=sufficiency.reason if sufficiency else None,
                contradiction_status=contra_status,
                generated_answer=response.answer,
                faithfulness_score=response.reliability.faithfulness_score if response.reliability else None,
                supported=response.reliability.supported if response.reliability else None,
                unsupported_claims=response.reliability.unsupported_claims if response.reliability else [],
                final_decision=decision_action,
                refusal_reason=refusal_reason,
                citations=[s.model_dump() for s in response.sources],
                trace_details=trace_details,
                latency_ms=latency_ms,
                model_name=self.llm_provider.model_name,
            )
            self.session.add(trace_record)
            await self.session.commit()
        except Exception as exc:
            logger.warning(f"Could not persist query trace record: {exc}")

    async def answer_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filters: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Executes the full RAG pipeline for a given user question with pre-generation
        evidence evaluation, contradiction & version analysis, and post-generation faithfulness verification.
        """
        req_id = request_id or str(uuid.uuid4())
        start_time = time.perf_counter()
        k = top_k or settings.RAG_DEFAULT_TOP_K
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.RAG_DEFAULT_SIMILARITY_THRESHOLD
        )

        logger.info(
            f"Beginning RAG pipeline: [request_id={req_id}] question='{question[:60]}...', top_k={k}, threshold={threshold}"
        )

        trace_details: Dict[str, Any] = {
            "request_id": req_id,
            "pipeline_stages": [],
        }

        # Step 1: Query Processing
        processed_query = self.query_processor.process(question)
        trace_details["pipeline_stages"].append({
            "stage": "query_processing",
            "question": question,
            "cleaned_query": processed_query.cleaned_query,
            "keywords": processed_query.keywords,
        })

        # Step 2: Evidence Retrieval
        retrieved_chunks = await self.retrieval_service.search(
            query=processed_query.cleaned_query,
            top_k=k,
            similarity_threshold=threshold,
            filters=filters,
        )
        trace_details["pipeline_stages"].append({
            "stage": "retrieval",
            "top_k": k,
            "similarity_threshold": threshold,
            "retrieved_count": len(retrieved_chunks),
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "document": c.metadata.get("filename"),
                    "version": c.metadata.get("version"),
                    "status": c.metadata.get("status"),
                    "score": round(c.score, 4) if c.score is not None else None,
                    "excerpt": c.content[:200],
                }
                for c in retrieved_chunks
            ],
        })

        # Step 3: Evidence Sufficiency Evaluation (Pre-generation check)
        sufficiency = self.evidence_evaluator.evaluate_sufficiency(
            retrieved_chunks=retrieved_chunks,
            query_keywords=processed_query.keywords,
            min_similarity_threshold=threshold,
        )
        trace_details["pipeline_stages"].append({
            "stage": "evidence_sufficiency",
            "is_sufficient": sufficiency.is_sufficient,
            "reason": sufficiency.reason,
            "details": sufficiency.details,
        })

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
            response = self.response_formatter.format_no_context_response(
                question=processed_query.cleaned_query,
                model_name=self.llm_provider.model_name,
                custom_message=settings.RAG_FALLBACK_RESPONSE,
                reason=sufficiency.reason or settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                reliability=reliability_report,
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            await self._save_trace(
                request_id=req_id,
                question=question,
                cleaned_query=processed_query.cleaned_query,
                top_k=k,
                similarity_threshold=threshold,
                retrieved_chunks=retrieved_chunks,
                sufficiency=sufficiency,
                contradiction_result=None,
                response=response,
                decision_action="REFUSED",
                refusal_reason=sufficiency.reason or settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                trace_details=trace_details,
                latency_ms=latency_ms,
            )
            return response

        # Step 4: Contradiction Analysis & Document Version Resolution
        contradiction_result = self.contradiction_service.analyze_and_resolve(
            retrieved_chunks=retrieved_chunks,
        )
        trace_details["pipeline_stages"].append({
            "stage": "contradiction_and_versioning",
            "status": contradiction_result.status.value,
            "is_resolved": contradiction_result.is_resolved,
            "resolution_action": contradiction_result.resolution.value if hasattr(contradiction_result.resolution, "value") else str(contradiction_result.resolution),
            "conflicting_sources": contradiction_result.conflicting_sources,
        })

        if (
            contradiction_result.status == ContradictionStatus.CONFLICT
            and not contradiction_result.is_resolved
        ):
            logger.warning(
                f"Unresolved factual conflict detected across sources: {contradiction_result.conflicting_sources}. "
                "Refusing ambiguous answer generation."
            )
            response = self.response_formatter.format_conflict_response(
                question=processed_query.cleaned_query,
                conflicting_chunks=contradiction_result.conflicting_chunks or retrieved_chunks,
                contradiction_report=contradiction_result.to_report(),
                query_keywords=processed_query.keywords,
                model_name=self.llm_provider.model_name,
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            await self._save_trace(
                request_id=req_id,
                question=question,
                cleaned_query=processed_query.cleaned_query,
                top_k=k,
                similarity_threshold=threshold,
                retrieved_chunks=retrieved_chunks,
                sufficiency=sufficiency,
                contradiction_result=contradiction_result,
                response=response,
                decision_action="REFUSED",
                refusal_reason=settings.RAG_CONFLICT_REASON,
                trace_details=trace_details,
                latency_ms=latency_ms,
            )
            return response

        evidence_chunks = contradiction_result.resolved_chunks or retrieved_chunks

        # Step 5: Context Construction
        context = self.context_builder.build_context(evidence_chunks)
        trace_details["pipeline_stages"].append({
            "stage": "context_construction",
            "has_context": context.has_context,
            "included_chunks_count": len(context.included_chunks),
            "total_chars": context.total_chars,
        })

        if not context.has_context:
            logger.info("Context builder produced empty context; returning fallback response.")
            reliability_report = ReliabilityReport(
                status=DecisionAction.REFUSE.value,
                faithfulness_score=0.0,
                supported=False,
                unsupported_claims=[],
                reason="Context construction yielded empty context.",
            )
            response = self.response_formatter.format_no_context_response(
                question=processed_query.cleaned_query,
                model_name=self.llm_provider.model_name,
                custom_message=settings.RAG_FALLBACK_RESPONSE,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                reliability=reliability_report,
                contradiction=contradiction_result.to_report(),
            )
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            await self._save_trace(
                request_id=req_id,
                question=question,
                cleaned_query=processed_query.cleaned_query,
                top_k=k,
                similarity_threshold=threshold,
                retrieved_chunks=retrieved_chunks,
                sufficiency=sufficiency,
                contradiction_result=contradiction_result,
                response=response,
                decision_action="REFUSED",
                refusal_reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                trace_details=trace_details,
                latency_ms=latency_ms,
            )
            return response

        # Step 6: LLM Generation (Initial Attempt)
        llm_response = await self.generation_service.generate_answer(
            question=processed_query.cleaned_query,
            context=context,
        )
        current_answer = llm_response.content
        trace_details["pipeline_stages"].append({
            "stage": "initial_generation",
            "model": llm_response.model_name,
            "raw_answer": current_answer,
        })

        # Step 7: Post-generation Faithfulness Verification & Reliability Decision
        verification = self.answer_verifier.verify_answer(
            answer=current_answer,
            context_text=context.context_text,
        )
        trace_details["pipeline_stages"].append({
            "stage": "faithfulness_verification",
            "score": verification.score,
            "supported": verification.supported,
            "claims_evaluated": verification.claims_evaluated,
            "unsupported_claims": verification.unsupported_claims,
            "reason": verification.reason,
        })

        decision = self.reliability_engine.decide(
            verification=verification,
            current_answer=current_answer,
            regeneration_attempt=0,
        )

        # Step 8: Optional Regeneration if unsupported claims detected
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
                trace_details["pipeline_stages"].append({
                    "stage": "regeneration",
                    "success": True,
                    "regen_answer": regen_answer,
                    "retry_verification_score": verification_retry.score,
                    "final_action": decision.action.value,
                })
            except Exception as exc:
                logger.error(f"Regeneration attempt failed: {exc}. Falling back to REFUSE.")
                decision = self.reliability_engine.decide(
                    verification=verification,
                    current_answer=current_answer,
                    regeneration_attempt=settings.RAG_MAX_REGENERATION_ATTEMPTS,
                )
                trace_details["pipeline_stages"].append({
                    "stage": "regeneration",
                    "success": False,
                    "error": str(exc),
                    "final_action": decision.action.value,
                })

        trace_details["pipeline_stages"].append({
            "stage": "reliability_decision",
            "action": decision.action.value,
            "final_answer": decision.final_answer,
            "faithfulness_score": decision.faithfulness_score,
            "reason": decision.reason,
        })

        reliability_report = ReliabilityReport(
            status=decision.action.value,
            faithfulness_score=decision.faithfulness_score,
            supported=decision.supported,
            unsupported_claims=decision.unsupported_claims,
            reason=decision.reason,
        )

        # Step 9: Response Formatting
        response = self.response_formatter.format_response(
            question=processed_query.cleaned_query,
            answer=decision.final_answer,
            evidence_chunks=context.included_chunks,
            query_keywords=processed_query.keywords,
            model_name=llm_response.model_name,
            reliability=reliability_report,
            contradiction=contradiction_result.to_report(),
        )

        refusal_reason = None
        if decision.action == DecisionAction.REFUSE:
            refusal_reason = "UNVERIFIED_CLAIMS"

        decision_map = {
            DecisionAction.ACCEPT: "ACCEPTED",
            DecisionAction.REGENERATE: "REGENERATED",
            DecisionAction.REFUSE: "REFUSED",
        }
        final_decision_str = decision_map.get(decision.action, decision.action.value)

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        await self._save_trace(
            request_id=req_id,
            question=question,
            cleaned_query=processed_query.cleaned_query,
            top_k=k,
            similarity_threshold=threshold,
            retrieved_chunks=retrieved_chunks,
            sufficiency=sufficiency,
            contradiction_result=contradiction_result,
            response=response,
            decision_action=final_decision_str,
            refusal_reason=refusal_reason,
            trace_details=trace_details,
            latency_ms=latency_ms,
        )

        return response

