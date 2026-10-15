from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.config import settings
from app.core.logging import get_logger
from app.ingestion.base import BaseEmbedder
from app.services.rag_service import RAGService
from evaluation.datasets.schemas import EvalDataset, EvalExample
from evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
from evaluation.metrics.base import BaseMetric, MetricResult
from evaluation.metrics.context_precision import ContextPrecisionMetric
from evaluation.metrics.context_recall import ContextRecallMetric
from evaluation.metrics.faithfulness import FaithfulnessMetric

logger = get_logger(__name__)


class ExampleEvaluationResult(BaseModel):
    """Evaluation output and diagnostic audit for a single benchmark query."""

    example_id: str
    question: str
    category: str
    expected_answer: str
    generated_answer: str
    retrieved_chunk_count: int
    retrieved_sources: List[Dict[str, Any]] = Field(default_factory=list)
    latency_seconds: float
    metrics: Dict[str, MetricResult] = Field(default_factory=dict)
    reliability_status: Optional[str] = None
    contradiction_status: Optional[str] = None


class EvaluationRunResult(BaseModel):
    """Aggregate benchmark run results for a specific RAG configuration."""

    run_id: str
    timestamp: str
    config_name: str
    model: str
    embedding_model: str
    retrieval_config: Dict[str, Any]
    summary_metrics: Dict[str, float]
    category_metrics: Dict[str, Dict[str, float]]
    total_examples: int
    total_latency_seconds: float
    average_latency_seconds: float
    results: List[ExampleEvaluationResult]


class EvaluationRunner:
    """
    Executes automated RAG evaluation against the enterprise benchmark dataset.
    Loads questions, runs them through the actual RAG pipeline, collects answers and context,
    and computes standardized evaluation metrics.
    """

    def __init__(
        self,
        session: AsyncSession,
        rag_service: Optional[RAGService] = None,
        embedder: Optional[BaseEmbedder] = None,
        metrics: Optional[List[BaseMetric]] = None,
    ):
        self.session = session
        self.rag_service = rag_service or RAGService(session=session, embedder=embedder)
        self.embedder = embedder or self.rag_service.retrieval_service.embedder

        self.metrics: List[BaseMetric] = metrics or [
            FaithfulnessMetric(),
            AnswerRelevancyMetric(embedder=self.embedder),
            ContextPrecisionMetric(),
            ContextRecallMetric(),
        ]

    async def run(
        self,
        dataset: EvalDataset,
        config_name: str = "default_config",
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> EvaluationRunResult:
        """
        Runs the benchmark evaluation over all dataset examples for the specified configuration.
        """
        k = top_k or settings.RAG_DEFAULT_TOP_K
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.RAG_DEFAULT_SIMILARITY_THRESHOLD
        )

        logger.info(
            f"Starting evaluation run '{config_name}' on {len(dataset.examples)} examples: "
            f"top_k={k}, threshold={threshold}"
        )

        start_run_time = time.perf_counter()
        example_results: List[ExampleEvaluationResult] = []

        for example in dataset.examples:
            logger.info(f"Evaluating example [{example.id}]: '{example.question[:50]}...'")
            t0 = time.perf_counter()

            # Execute the actual RAG pipeline
            ask_response = await self.rag_service.answer_question(
                question=example.question,
                top_k=k,
                similarity_threshold=threshold,
            )
            latency = round(time.perf_counter() - t0, 3)

            # Extract retrieved contexts and chunks from sources
            retrieved_contexts = [s.excerpt for s in ask_response.sources if s.excerpt]
            retrieved_chunks = ask_response.sources

            # Calculate evaluation metrics
            computed_metrics: Dict[str, MetricResult] = {}
            for metric in self.metrics:
                m_res = await metric.compute(
                    example=example,
                    answer=ask_response.answer,
                    retrieved_contexts=retrieved_contexts,
                    retrieved_chunks=retrieved_chunks,
                    reliability_report=ask_response.reliability,
                    contradiction_report=ask_response.contradiction,
                )
                computed_metrics[metric.name] = m_res

            rel_status = (
                ask_response.reliability.status if ask_response.reliability else None
            )
            contra_status = (
                ask_response.contradiction.status if ask_response.contradiction else None
            )

            res = ExampleEvaluationResult(
                example_id=example.id,
                question=example.question,
                category=example.category.value,
                expected_answer=example.expected_answer,
                generated_answer=ask_response.answer,
                retrieved_chunk_count=len(ask_response.sources),
                retrieved_sources=[s.model_dump() for s in ask_response.sources],
                latency_seconds=latency,
                metrics=computed_metrics,
                reliability_status=rel_status,
                contradiction_status=contra_status,
            )
            example_results.append(res)

        total_latency = round(time.perf_counter() - start_run_time, 3)
        avg_latency = round(total_latency / max(1, len(example_results)), 3)

        # Compute aggregate mean metrics
        summary_metrics: Dict[str, float] = {}
        for metric in self.metrics:
            scores = [r.metrics[metric.name].score for r in example_results if metric.name in r.metrics]
            summary_metrics[metric.name] = round(sum(scores) / len(scores), 4) if scores else 0.0

        # Compute category-level metric breakdowns
        category_metrics: Dict[str, Dict[str, float]] = {}
        for cat in set(r.category for r in example_results):
            cat_results = [r for r in example_results if r.category == cat]
            category_metrics[cat] = {}
            for metric in self.metrics:
                scores = [r.metrics[metric.name].score for r in cat_results if metric.name in r.metrics]
                category_metrics[cat][metric.name] = (
                    round(sum(scores) / len(scores), 4) if scores else 0.0
                )

        run_id = f"eval_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{config_name}"

        return EvaluationRunResult(
            run_id=run_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config_name=config_name,
            model=self.rag_service.llm_provider.model_name,
            embedding_model=self.embedder.model_name,
            retrieval_config={"top_k": k, "similarity_threshold": threshold},
            summary_metrics=summary_metrics,
            category_metrics=category_metrics,
            total_examples=len(example_results),
            total_latency_seconds=total_latency,
            average_latency_seconds=avg_latency,
            results=example_results,
        )
