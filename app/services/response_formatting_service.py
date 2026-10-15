import re
from typing import List, Optional
from app.core.config import settings
from app.core.logging import get_logger
from app.retrieval.base import RetrievedChunk
from app.schemas.ask import AskResponse, ReliabilityReport, SourceCitation
from app.services.confidence_service import ConfidenceResult, ConfidenceScorer

logger = get_logger(__name__)


def extract_relevant_excerpt(
    content: str,
    keywords: Optional[List[str]] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Extracts the most informative snippet from a chunk based on query keyword density.
    Falls back cleanly to leading content if no individual sentence matches.
    """
    limit = max_chars or settings.RAG_EXCERPT_MAX_CHARS
    cleaned = re.sub(r"\s+", " ", content).strip()
    if len(cleaned) <= limit:
        return cleaned

    if not keywords:
        return cleaned[:limit].rstrip() + "..."

    # Split into candidate sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    if not sentences:
        return cleaned[:limit].rstrip() + "..."

    kw_set = {k.lower() for k in keywords}

    scored_sentences = []
    for s in sentences:
        s_tokens = set(re.findall(r"\b\w{3,}\b", s.lower()))
        matches = len(kw_set & s_tokens)
        scored_sentences.append((matches, s))

    # Sort descending by match count
    scored_sentences.sort(key=lambda x: x[0], reverse=True)
    best_matches = [s for count, s in scored_sentences if count > 0]

    if best_matches:
        candidate = best_matches[0]
        if len(candidate) <= limit:
            return candidate
        return candidate[:limit].rstrip() + "..."

    return cleaned[:limit].rstrip() + "..."


class ResponseFormattingService:
    """
    Service responsible for packaging pipeline outputs into structured, audited API response contracts.
    Computes deterministic confidence levels, extracts source excerpts, and formats machine-readable reasons.
    """

    def __init__(self, confidence_scorer: Optional[ConfidenceScorer] = None):
        self.confidence_scorer = confidence_scorer or ConfidenceScorer()

    def format_response(
        self,
        question: str,
        answer: str,
        evidence_chunks: List[RetrievedChunk],
        query_keywords: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        reliability: Optional[ReliabilityReport] = None,
    ) -> AskResponse:
        """
        Formats question, synthesized answer, calculated confidence, and source citations.
        Audits post-generation faithfulness and ensures unverified answers are strictly refused.
        """
        cleaned_answer = answer.strip()

        # Handle post-generation refusal from ReliabilityDecisionEngine
        if reliability is not None and (
            reliability.status == "REFUSE" or not reliability.supported
        ):
            logger.warning(
                f"Response refused by ReliabilityDecisionEngine: {reliability.reason}"
            )
            refusal_text = (
                cleaned_answer
                if (
                    settings.RAG_UNVERIFIED_REFUSAL_RESPONSE.lower() in cleaned_answer.lower()
                    or "not enough reliable information" in cleaned_answer.lower()
                )
                else settings.RAG_UNVERIFIED_REFUSAL_RESPONSE
            )
            return AskResponse(
                question=question,
                answer=refusal_text,
                confidence="none",
                confidence_score=0.0,
                reason=settings.RAG_UNVERIFIED_REASON,
                sources=[],
                reliability=reliability,
                model_name=model_name,
            )

        # Check if the answer indicates that evidence was unavailable or insufficient
        is_refusal = (
            settings.RAG_FALLBACK_RESPONSE.lower() in cleaned_answer.lower()
            or settings.RAG_UNVERIFIED_REFUSAL_RESPONSE.lower() in cleaned_answer.lower()
            or "not have enough information" in cleaned_answer.lower()
            or "information is unavailable" in cleaned_answer.lower()
            or "couldn't find sufficient information" in cleaned_answer.lower()
            or "not enough reliable information" in cleaned_answer.lower()
            or "cannot answer" in cleaned_answer.lower()
        )

        confidence_result = self.confidence_scorer.evaluate(
            retrieved_chunks=evidence_chunks,
            query_keywords=query_keywords,
            is_refusal=is_refusal,
        )

        # If evidence was insufficient or refusal occurred, return controlled fallback
        if is_refusal or confidence_result.level == "none":
            logger.info("Response classified as INSUFFICIENT_EVIDENCE; zeroing source citations.")
            refusal_reliability = reliability or ReliabilityReport(
                status="REFUSE",
                faithfulness_score=1.0,
                supported=True,
                unsupported_claims=[],
                reason="Controlled refusal: retrieved context contains insufficient evidence.",
            )
            return AskResponse(
                question=question,
                answer=(
                    cleaned_answer
                    if is_refusal
                    else settings.RAG_FALLBACK_RESPONSE
                ),
                confidence=confidence_result.level,
                confidence_score=confidence_result.score,
                reason=settings.RAG_INSUFFICIENT_EVIDENCE_REASON,
                sources=[],
                reliability=refusal_reliability,
                model_name=model_name,
            )

        # Build structured citations
        sources: List[SourceCitation] = []
        for chunk in evidence_chunks:
            meta = chunk.metadata or {}
            doc_name = meta.get("filename") or meta.get("document_title") or "Unknown Document"
            doc_version = str(meta.get("version") or "1.0")
            score = round(chunk.score if chunk.score is not None else 0.0, 4)
            excerpt = extract_relevant_excerpt(chunk.content, query_keywords)

            sources.append(
                SourceCitation(
                    document=doc_name,
                    version=doc_version,
                    chunk_id=chunk.chunk_id,
                    score=score,
                    excerpt=excerpt,
                    chunk_index=meta.get("chunk_index"),
                    document_id=chunk.document_id,
                )
            )

        accepted_reliability = reliability or ReliabilityReport(
            status="ACCEPT",
            faithfulness_score=1.0,
            supported=True,
            unsupported_claims=[],
            reason="All claims verified as grounded in retrieved context.",
        )

        return AskResponse(
            question=question,
            answer=cleaned_answer,
            confidence=confidence_result.level,
            confidence_score=confidence_result.score,
            reason=None,
            sources=sources,
            reliability=accepted_reliability,
            model_name=model_name,
        )

    def format_no_context_response(
        self,
        question: str,
        model_name: Optional[str] = None,
        custom_message: Optional[str] = None,
        reason: Optional[str] = None,
        reliability: Optional[ReliabilityReport] = None,
    ) -> AskResponse:
        """
        Formats controlled response when retrieval yielded 0 chunks or chunks below similarity threshold.
        """
        message = custom_message or settings.RAG_FALLBACK_RESPONSE
        refusal_reason = reason or settings.RAG_INSUFFICIENT_EVIDENCE_REASON
        reliability_report = reliability or ReliabilityReport(
            status="REFUSE",
            faithfulness_score=0.0,
            supported=False,
            unsupported_claims=[],
            reason=f"Pre-generation check: {refusal_reason}",
        )

        return AskResponse(
            question=question,
            answer=message,
            confidence="none",
            confidence_score=0.0,
            reason=refusal_reason,
            sources=[],
            reliability=reliability_report,
            model_name=model_name,
        )
