from typing import Dict, List
from evaluation.runners.comparator import ComparisonResult
from evaluation.runners.evaluator import EvaluationRunResult


class ReportGenerator:
    """Generates structured Markdown reports from benchmark evaluation results."""

    @classmethod
    def generate_run_markdown(cls, run: EvaluationRunResult) -> str:
        """Renders an individual benchmark run as a comprehensive Markdown report."""
        lines: List[str] = []

        lines.append(f"# Automated RAG Evaluation Report: {run.config_name}")
        lines.append("")
        lines.append(f"- **Run ID**: `{run.run_id}`")
        lines.append(f"- **Timestamp**: `{run.timestamp}`")
        lines.append(f"- **LLM Model**: `{run.model}`")
        lines.append(f"- **Embedding Model**: `{run.embedding_model}`")
        lines.append(
            f"- **Retrieval Configuration**: `top_k={run.retrieval_config.get('top_k')}`, "
            f"`threshold={run.retrieval_config.get('similarity_threshold')}`"
        )
        lines.append(f"- **Total Scenarios Evaluated**: {run.total_examples}")
        lines.append(f"- **Average Latency**: {run.average_latency_seconds}s per query")
        lines.append("")

        # Executive Metric Summary Table
        lines.append("## Executive Metrics Summary")
        lines.append("")
        lines.append("| Metric | Score | Target | Status |")
        lines.append("| :--- | :---: | :---: | :---: |")

        for m_name, score in run.summary_metrics.items():
            formatted_name = m_name.replace("_", " ").title()
            target = 0.80 if "recall" in m_name or "relevancy" in m_name else 0.85
            status = "✅ PASS" if score >= target else "⚠️ ATTENTION"
            lines.append(f"| **{formatted_name}** | **{score:.4f}** | >={target:.2f} | {status} |")

        lines.append("")

        # Category Breakdown Table
        lines.append("## Performance by Question Category")
        lines.append("")
        metrics_list = list(run.summary_metrics.keys())
        header = "| Category | " + " | ".join(m.replace("_", " ").title() for m in metrics_list) + " |"
        divider = "| :--- | " + " | ".join(":---:" for _ in metrics_list) + " |"
        lines.append(header)
        lines.append(divider)

        for cat, cat_scores in sorted(run.category_metrics.items()):
            row_vals = [f"{cat_scores.get(m, 0.0):.4f}" for m in metrics_list]
            lines.append(f"| **{cat}** | " + " | ".join(row_vals) + " |")

        lines.append("")

        # Detailed Query Audit Table
        lines.append("## Individual Query Results")
        lines.append("")
        lines.append("| ID | Category | Question | Faithfulness | Relevancy | Context Prec | Context Rec | Latency | Status |")
        lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |")

        for res in run.results:
            q_trunc = res.question[:35] + ("..." if len(res.question) > 35 else "")
            faith = res.metrics.get("faithfulness")
            rel = res.metrics.get("answer_relevancy")
            prec = res.metrics.get("context_precision")
            rec = res.metrics.get("context_recall")

            faith_str = f"{faith.score:.2f}" if faith else "N/A"
            rel_str = f"{rel.score:.2f}" if rel else "N/A"
            prec_str = f"{prec.score:.2f}" if prec else "N/A"
            rec_str = f"{rec.score:.2f}" if rec else "N/A"
            status = res.reliability_status or "OK"

            lines.append(
                f"| `{res.example_id}` | {res.category} | {q_trunc} | {faith_str} | {rel_str} | {prec_str} | {rec_str} | {res.latency_seconds}s | `{status}` |"
            )

        lines.append("")
        return "\n".join(lines)

    @classmethod
    def generate_comparison_markdown(cls, comparison: ComparisonResult) -> str:
        """Renders an experiment comparison as a detailed Markdown report."""
        lines: List[str] = []

        lines.append(f"# RAG Experiment Comparison: {comparison.config_a_name} vs. {comparison.config_b_name}")
        lines.append("")
        lines.append(
            f"Comparing **{comparison.config_a_name}** (`top_k={comparison.retrieval_config_a.get('top_k')}`) "
            f"against **{comparison.config_b_name}** (`top_k={comparison.retrieval_config_b.get('top_k')}`)."
        )
        lines.append("")

        # Comparison Delta Table
        lines.append("## Metric Comparison & Deltas")
        lines.append("")
        lines.append("| Metric | Configuration A | Configuration B | Delta (B - A) | % Shift |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")

        for m_name, delta in sorted(comparison.metric_deltas.items()):
            formatted_name = m_name.replace("_", " ").title()
            sign = "+" if delta.delta > 0 else ""
            lines.append(
                f"| **{formatted_name}** | {delta.config_a_score:.4f} | {delta.config_b_score:.4f} | "
                f"**{sign}{delta.delta:.4f}** | {sign}{delta.percentage_change:.2f}% |"
            )

        lines.append("")
        lines.append("## Latency Comparison")
        lines.append("")
        lines.append(f"- **{comparison.config_a_name} Latency**: {comparison.avg_latency_a}s")
        lines.append(f"- **{comparison.config_b_name} Latency**: {comparison.avg_latency_b}s")
        sign = "+" if comparison.latency_delta_seconds > 0 else ""
        lines.append(f"- **Net Latency Delta**: {sign}{comparison.latency_delta_seconds}s")
        lines.append("")

        lines.append("## Automated Engineering Findings")
        lines.append("")
        for finding in comparison.key_findings:
            lines.append(f"- {finding}")

        lines.append("")
        return "\n".join(lines)
