import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.models.document import DocumentChunkModel, DocumentModel
from app.models.trace import QueryTraceModel, SystemErrorLogModel

logger = get_logger(__name__)

REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "evaluation" / "reports" / "output"


class AdminService:
    """
    Observability and Analytics domain service powering the Admin Dashboard.
    Queries database traces, models, and evaluation artifacts to compute live operational metrics.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_overview(self) -> Dict[str, Any]:
        """Calculates top-level KPI metrics across documents, chunks, queries, and reliability."""
        # Document counts
        doc_count = await self.session.scalar(select(func.count(DocumentModel.id))) or 0
        chunk_count = await self.session.scalar(select(func.count(DocumentChunkModel.id))) or 0

        # Query counts and rates
        total_queries = await self.session.scalar(select(func.count(QueryTraceModel.id))) or 0
        accepted_queries = (
            await self.session.scalar(
                select(func.count(QueryTraceModel.id)).where(QueryTraceModel.final_decision.in_(["ACCEPTED", "ACCEPT"]))
            )
            or 0
        )
        regenerated_queries = (
            await self.session.scalar(
                select(func.count(QueryTraceModel.id)).where(QueryTraceModel.final_decision.in_(["REGENERATED", "REGENERATE"]))
            )
            or 0
        )
        refused_queries = (
            await self.session.scalar(
                select(func.count(QueryTraceModel.id)).where(QueryTraceModel.final_decision.in_(["REFUSED", "REFUSE"]))
            )
            or 0
        )

        acceptance_rate = round(accepted_queries / total_queries, 4) if total_queries > 0 else 1.0
        refusal_rate = round(refused_queries / total_queries, 4) if total_queries > 0 else 0.0

        # Averages
        avg_retrieval_score = (
            await self.session.scalar(
                select(func.avg(QueryTraceModel.top_similarity_score)).where(
                    QueryTraceModel.top_similarity_score.isnot(None)
                )
            )
            or 0.0
        )
        avg_faithfulness = (
            await self.session.scalar(
                select(func.avg(QueryTraceModel.faithfulness_score)).where(
                    QueryTraceModel.faithfulness_score.isnot(None)
                )
            )
            or 0.0
        )

        # Average answer relevancy from latest evaluation benchmark or traces
        latest_eval = self._get_latest_eval_metrics()
        avg_relevancy = latest_eval.get("answer_relevancy", 0.85)

        # Recent system errors
        recent_errors_query = (
            select(SystemErrorLogModel)
            .order_by(desc(SystemErrorLogModel.created_at))
            .limit(10)
        )
        error_records = (await self.session.execute(recent_errors_query)).scalars().all()
        recent_errors = [
            {
                "id": str(err.id),
                "request_id": err.request_id,
                "endpoint": err.endpoint,
                "error_message": err.error_message,
                "timestamp": err.created_at.isoformat() if err.created_at else None,
            }
            for err in error_records
        ]

        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "total_questions": total_queries,
            "accepted_questions": accepted_queries,
            "regenerated_questions": regenerated_queries,
            "refused_questions": refused_queries,
            "acceptance_rate": acceptance_rate,
            "refusal_rate": refusal_rate,
            "average_retrieval_score": round(float(avg_retrieval_score), 4),
            "average_faithfulness_score": round(float(avg_faithfulness), 4),
            "average_answer_relevancy": round(float(avg_relevancy), 4),
            "recent_system_errors": recent_errors,
        }

    async def get_queries(
        self,
        limit: int = 50,
        offset: int = 0,
        decision: Optional[str] = None,
        search: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns paginated query traces with stage metrics and diagnostic decisions."""
        query = select(QueryTraceModel)

        if decision:
            query = query.where(QueryTraceModel.final_decision == decision.upper())

        if search:
            query = query.where(QueryTraceModel.question.ilike(f"%{search}%"))

        # Total count matching filters
        count_query = select(func.count()).select_from(query.subquery())
        total_filtered = (await self.session.scalar(count_query)) or 0

        # Paginated results
        query = query.order_by(desc(QueryTraceModel.created_at)).limit(limit).offset(offset)
        traces = (await self.session.execute(query)).scalars().all()

        items = []
        for t in traces:
            retrieved_docs = []
            if t.trace_details and isinstance(t.trace_details, dict):
                stages = t.trace_details.get("pipeline_stages", [])
                if isinstance(stages, list):
                    for stg in stages:
                        if isinstance(stg, dict) and stg.get("stage") == "retrieval":
                            for chk in stg.get("chunks", []):
                                if isinstance(chk, dict) and chk.get("document"):
                                    retrieved_docs.append(chk.get("document"))
            if not retrieved_docs and t.citations:
                retrieved_docs = [c.get("document") for c in t.citations if isinstance(c, dict) and c.get("document")]
            retrieved_docs = sorted(list(set(retrieved_docs)))

            items.append({
                "request_id": t.request_id,
                "question": t.question,
                "retrieved_documents": retrieved_docs or [c.get("document") for c in t.citations if c.get("document")],
                "top_similarity_score": t.top_similarity_score,
                "mean_similarity_score": t.mean_similarity_score,
                "generated_answer": t.generated_answer,
                "citations": t.citations,
                "faithfulness_score": t.faithfulness_score,
                "supported": t.supported,
                "unsupported_claims": t.unsupported_claims,
                "final_decision": t.final_decision,
                "refusal_reason": t.refusal_reason,
                "latency_ms": t.latency_ms,
                "timestamp": t.created_at.isoformat() if t.created_at else None,
            })

        return {
            "total": total_filtered,
            "limit": limit,
            "offset": offset,
            "items": items,
        }

    async def get_query_trace(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Returns the full end-to-end multi-stage pipeline execution trace for a single request."""
        query = select(QueryTraceModel).where(QueryTraceModel.request_id == request_id)
        trace = (await self.session.execute(query)).scalar_one_or_none()
        if not trace:
            return None

        return {
            "request_id": trace.request_id,
            "question": trace.question,
            "cleaned_query": trace.cleaned_query,
            "top_k": trace.top_k,
            "similarity_threshold": trace.similarity_threshold,
            "retrieval_count": trace.retrieval_count,
            "top_similarity_score": trace.top_similarity_score,
            "mean_similarity_score": trace.mean_similarity_score,
            "is_sufficient": trace.is_sufficient,
            "sufficiency_reason": trace.sufficiency_reason,
            "contradiction_status": trace.contradiction_status,
            "generated_answer": trace.generated_answer,
            "faithfulness_score": trace.faithfulness_score,
            "supported": trace.supported,
            "unsupported_claims": trace.unsupported_claims,
            "final_decision": trace.final_decision,
            "refusal_reason": trace.refusal_reason,
            "citations": trace.citations,
            "pipeline_stages": trace.trace_details.get("pipeline_stages", []),
            "latency_ms": trace.latency_ms,
            "model_name": trace.model_name,
            "timestamp": trace.created_at.isoformat() if trace.created_at else None,
        }

    async def get_retrieval_analysis(self) -> Dict[str, Any]:
        """Analyzes vector retrieval distributions, similarity scores, and threshold effectiveness."""
        query = select(QueryTraceModel).order_by(desc(QueryTraceModel.created_at)).limit(100)
        traces = (await self.session.execute(query)).scalars().all()

        score_buckets = {"<0.40": 0, "0.40-0.55": 0, "0.55-0.70": 0, "0.70-0.85": 0, ">0.85": 0}
        total_retrieved = 0
        relevant_count = 0
        irrelevant_count = 0

        doc_frequency: Dict[str, int] = {}

        for t in traces:
            score = t.top_similarity_score
            if score is not None:
                if score < 0.40:
                    score_buckets["<0.40"] += 1
                elif score < 0.55:
                    score_buckets["0.40-0.55"] += 1
                elif score < 0.70:
                    score_buckets["0.55-0.70"] += 1
                elif score < 0.85:
                    score_buckets["0.70-0.85"] += 1
                else:
                    score_buckets[">0.85"] += 1

            total_retrieved += t.retrieval_count
            if t.is_sufficient:
                relevant_count += len(t.citations) if t.citations else 1
            else:
                irrelevant_count += t.retrieval_count

            for c in t.citations:
                doc = c.get("document", "unknown")
                doc_frequency[doc] = doc_frequency.get(doc, 0) + 1

        top_docs = [
            {"document": doc, "count": count}
            for doc, count in sorted(doc_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        ]

        return {
            "current_retrieval_threshold": settings.DEFAULT_SIMILARITY_THRESHOLD,
            "default_top_k": settings.DEFAULT_SEARCH_TOP_K,
            "total_retrieval_events": len(traces),
            "score_distribution": score_buckets,
            "relevant_retrievals_count": relevant_count,
            "irrelevant_retrievals_count": irrelevant_count,
            "top_retrieved_documents": top_docs,
        }

    def get_evaluation_history(self) -> Dict[str, Any]:
        """Loads historical Ragas evaluation benchmark runs and comparisons from reports store."""
        runs: List[Dict[str, Any]] = []

        if REPORTS_DIR.exists():
            for p in sorted(REPORTS_DIR.glob("eval_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:15]:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        runs.append({
                            "run_id": data.get("run_id"),
                            "timestamp": data.get("timestamp"),
                            "config_name": data.get("config_name"),
                            "model": data.get("model"),
                            "embedding_model": data.get("embedding_model"),
                            "retrieval_config": data.get("retrieval_config", {}),
                            "summary_metrics": data.get("summary_metrics", {}),
                            "total_examples": data.get("total_examples", 0),
                            "average_latency_seconds": data.get("average_latency_seconds", 0.0),
                        })
                except Exception as e:
                    logger.warning(f"Error reading eval report {p.name}: {e}")

        # Load latest comparison if available
        comparison_data = None
        comp_path = REPORTS_DIR / "comparison_config_a_topk_3_vs_config_b_topk_5.json"
        if comp_path.exists():
            try:
                with open(comp_path, "r", encoding="utf-8") as f:
                    comparison_data = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading comparison report: {e}")

        return {
            "total_benchmark_runs": len(runs),
            "runs": runs,
            "latest_comparison": comparison_data,
        }

    async def get_knowledge_base_status(self) -> Dict[str, Any]:
        """Returns comprehensive catalog of all documents, versions, chunks, and embedding health."""
        query = select(DocumentModel).order_by(desc(DocumentModel.created_at))
        docs = (await self.session.execute(query)).scalars().all()

        items = []
        for d in docs:
            # Query chunk count
            c_count = (
                await self.session.scalar(
                    select(func.count(DocumentChunkModel.id)).where(DocumentChunkModel.document_id == d.id)
                )
                or 0
            )

            # Check if embeddings are populated
            has_unembedded = (
                await self.session.scalar(
                    select(func.count(DocumentChunkModel.id)).where(
                        DocumentChunkModel.document_id == d.id,
                        DocumentChunkModel.embedding.is_(None),
                    )
                )
                or 0
            )
            embedding_status = "EMBEDDED" if (c_count > 0 and has_unembedded == 0) else "PENDING"

            items.append({
                "id": str(d.id),
                "filename": d.filename,
                "title": d.title,
                "document_type": d.document_type,
                "version": d.version,
                "status": d.status,
                "effective_date": d.effective_date.isoformat() if d.effective_date else None,
                "expiration_date": d.expiration_date.isoformat() if d.expiration_date else None,
                "chunk_count": c_count,
                "embedding_status": embedding_status,
                "checksum": d.checksum,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            })

        return {
            "total_documents": len(items),
            "documents": items,
        }

    async def get_reliability_metrics(self) -> Dict[str, Any]:
        """Provides deep reliability analytics: hallucinations detected, unsupported claims, and contradictions."""
        total_queries = await self.session.scalar(select(func.count(QueryTraceModel.id))) or 0

        # Insufficient evidence cases
        insufficient_count = (
            await self.session.scalar(
                select(func.count(QueryTraceModel.id)).where(QueryTraceModel.is_sufficient.is_(False))
            )
            or 0
        )

        # Contradiction cases
        contradiction_count = (
            await self.session.scalar(
                select(func.count(QueryTraceModel.id)).where(
                    QueryTraceModel.contradiction_status.in_(["CONFLICT", "PARTIALLY_CONFLICT"])
                )
            )
            or 0
        )

        # Hallucination / Unsupported claims detections
        unsupported_query = select(QueryTraceModel).order_by(desc(QueryTraceModel.created_at)).limit(200)
        traces = (await self.session.execute(unsupported_query)).scalars().all()

        unsupported_claims_log: List[Dict[str, Any]] = []
        for t in traces:
            if t.unsupported_claims and isinstance(t.unsupported_claims, list):
                for claim in t.unsupported_claims:
                    unsupported_claims_log.append({
                        "request_id": t.request_id,
                        "question": t.question,
                        "claim": claim,
                        "final_decision": t.final_decision,
                        "timestamp": t.created_at.isoformat() if t.created_at else None,
                    })
                    if len(unsupported_claims_log) >= 50:
                        break
            if len(unsupported_claims_log) >= 50:
                break

        # Refusal breakdown
        refusal_reasons_query = (
            select(QueryTraceModel.refusal_reason, func.count(QueryTraceModel.id))
            .where(QueryTraceModel.final_decision == "REFUSED")
            .group_by(QueryTraceModel.refusal_reason)
        )
        refusal_breakdown = {
            (r[0] or "UNKNOWN"): r[1] for r in (await self.session.execute(refusal_reasons_query)).all()
        }

        return {
            "total_queries_evaluated": total_queries,
            "insufficient_evidence_cases": insufficient_count,
            "contradiction_cases": contradiction_count,
            "hallucinations_detected_count": len(unsupported_claims_log),
            "unsupported_claims_log": unsupported_claims_log,
            "refusal_reasons_breakdown": refusal_breakdown,
        }

    def _get_latest_eval_metrics(self) -> Dict[str, float]:
        """Reads latest evaluation summary metrics if available."""
        if REPORTS_DIR.exists():
            files = sorted(REPORTS_DIR.glob("eval_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            if files:
                try:
                    with open(files[0], "r", encoding="utf-8") as f:
                        data = json.load(f)
                        return data.get("summary_metrics", {})
                except Exception:
                    pass
        return {}
