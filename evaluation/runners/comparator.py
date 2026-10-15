from typing import Any, Dict, List
from pydantic import BaseModel, Field
from evaluation.runners.evaluator import EvaluationRunResult


class MetricDelta(BaseModel):
    """Comparative delta analysis between two configuration runs for a single metric."""

    metric_name: str
    config_a_score: float
    config_b_score: float
    delta: float
    percentage_change: float


class ComparisonResult(BaseModel):
    """Comparative report evaluating trade-offs between two RAG configurations."""

    config_a_name: str
    config_b_name: str
    retrieval_config_a: Dict[str, Any]
    retrieval_config_b: Dict[str, Any]
    metric_deltas: Dict[str, MetricDelta]
    avg_latency_a: float
    avg_latency_b: float
    latency_delta_seconds: float
    key_findings: List[str] = Field(default_factory=list)


class ExperimentComparator:
    """
    Compares two evaluation runs to objectively measure the trade-offs of hyperparameter shifts
    (e.g. top_k = 3 vs. top_k = 5).
    """

    @classmethod
    def compare(
        cls,
        run_a: EvaluationRunResult,
        run_b: EvaluationRunResult,
    ) -> ComparisonResult:
        deltas: Dict[str, MetricDelta] = {}

        all_metric_keys = set(run_a.summary_metrics.keys()).union(run_b.summary_metrics.keys())
        for m in sorted(all_metric_keys):
            val_a = run_a.summary_metrics.get(m, 0.0)
            val_b = run_b.summary_metrics.get(m, 0.0)
            delta = round(val_b - val_a, 4)
            pct = round((delta / val_a * 100) if val_a > 0 else 0.0, 2)
            deltas[m] = MetricDelta(
                metric_name=m,
                config_a_score=val_a,
                config_b_score=val_b,
                delta=delta,
                percentage_change=pct,
            )

        latency_delta = round(run_b.average_latency_seconds - run_a.average_latency_seconds, 3)

        # Generate automated engineering insights based on metric shifts
        findings: List[str] = []

        if "context_recall" in deltas:
            d_recall = deltas["context_recall"].delta
            if d_recall > 0:
                findings.append(
                    f"Increasing retrieval depth to {run_b.retrieval_config.get('top_k')} improved Context Recall by +{round(d_recall * 100, 1)}%, capturing more multi-document evidence."
                )
            elif d_recall == 0:
                findings.append(
                    f"Context Recall remained constant ({run_a.summary_metrics.get('context_recall')}) between configurations."
                )

        if "context_precision" in deltas:
            d_prec = deltas["context_precision"].delta
            if d_prec < 0:
                findings.append(
                    f"Context Precision decreased by {round(d_prec * 100, 1)}% due to lower-ranked distractor chunks being included in top-{run_b.retrieval_config.get('top_k')}."
                )
            elif d_prec > 0:
                findings.append(
                    f"Context Precision improved by +{round(d_prec * 100, 1)}% under {run_b.config_name}."
                )

        if "faithfulness" in deltas:
            d_faith = deltas["faithfulness"].delta
            if abs(d_faith) < 0.02:
                findings.append(
                    "Faithfulness remained robust across both configurations, demonstrating effective post-generation verification."
                )
            elif d_faith < 0:
                findings.append(
                    f"Faithfulness decreased slightly by {round(d_faith * 100, 1)}% under larger context."
                )

        findings.append(
            f"Latency impact: Configuration B average latency changed by {latency_delta:+}s per query."
        )

        return ComparisonResult(
            config_a_name=run_a.config_name,
            config_b_name=run_b.config_name,
            retrieval_config_a=run_a.retrieval_config,
            retrieval_config_b=run_b.retrieval_config,
            metric_deltas=deltas,
            avg_latency_a=run_a.average_latency_seconds,
            avg_latency_b=run_b.average_latency_seconds,
            latency_delta_seconds=latency_delta,
            key_findings=findings,
        )
