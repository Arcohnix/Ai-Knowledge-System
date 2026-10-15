from pathlib import Path
import pytest
from sqlalchemy.ext.asyncio import AsyncSession
from app.ingestion.embeddings.factory import get_embedder
from evaluation.datasets.loader import (
    load_eval_dataset,
    seed_eval_documents_if_needed,
)
from evaluation.datasets.schemas import EvalDataset
from evaluation.reports.generator import ReportGenerator
from evaluation.reports.store import ReportStore
from evaluation.runners.comparator import ExperimentComparator
from evaluation.runners.evaluator import EvaluationRunner


@pytest.mark.asyncio
async def test_evaluation_runner_full_execution(test_db_session: AsyncSession, tmp_path: Path):
    embedder = get_embedder("mock")
    # 1. Seed benchmark documents in test DB
    seeded = await seed_eval_documents_if_needed(session=test_db_session, embedder=embedder)
    assert seeded >= 6

    # 2. Load dataset
    dataset = load_eval_dataset()
    assert len(dataset.examples) >= 10

    # 3. Create runner with mock embedder
    runner = EvaluationRunner(session=test_db_session, embedder=embedder)

    # Run for subset of 4 examples to keep integration test fast
    mini_dataset = EvalDataset(
        name="mini_benchmark",
        version="1.0.0",
        description="Fast test benchmark",
        examples=dataset.examples[:4],
    )

    result_a = await runner.run(
        dataset=mini_dataset,
        config_name="Config_TopK_3",
        top_k=3,
        similarity_threshold=0.2,
    )

    assert result_a.config_name == "Config_TopK_3"
    assert result_a.total_examples == 4
    assert result_a.average_latency_seconds >= 0.0
    assert "faithfulness" in result_a.summary_metrics
    assert "answer_relevancy" in result_a.summary_metrics
    assert "context_precision" in result_a.summary_metrics
    assert "context_recall" in result_a.summary_metrics

    for res in result_a.results:
        assert res.example_id
        assert res.question
        assert res.generated_answer
        assert len(res.metrics) == 4
        for m_name, m_res in res.metrics.items():
            assert 0.0 <= m_res.score <= 1.0


@pytest.mark.asyncio
async def test_comparative_experiment_and_report_storage(
    test_db_session: AsyncSession, tmp_path: Path
):
    embedder = get_embedder("mock")
    await seed_eval_documents_if_needed(session=test_db_session, embedder=embedder)
    dataset = load_eval_dataset()

    mini_dataset = EvalDataset(
        name="mini_benchmark",
        version="1.0.0",
        description="Fast comparative benchmark",
        examples=[dataset.examples[0], dataset.examples[4], dataset.examples[6]],  # Straightforward, multi-doc, unsupported
    )

    runner = EvaluationRunner(session=test_db_session, embedder=embedder)

    run_a = await runner.run(
        dataset=mini_dataset,
        config_name="Config_TopK_3",
        top_k=3,
        similarity_threshold=0.2,
    )
    run_b = await runner.run(
        dataset=mini_dataset,
        config_name="Config_TopK_5",
        top_k=5,
        similarity_threshold=0.2,
    )

    # Compare
    comparison = ExperimentComparator.compare(run_a=run_a, run_b=run_b)
    assert comparison.config_a_name == "Config_TopK_3"
    assert comparison.config_b_name == "Config_TopK_5"
    assert "context_recall" in comparison.metric_deltas
    assert "context_precision" in comparison.metric_deltas

    # Store reports
    store = ReportStore(reports_dir=tmp_path)
    files_a = store.save_run(run_a)
    files_b = store.save_run(run_b)
    files_comp = store.save_comparison(comparison)

    assert files_a["json"].exists()
    assert files_a["markdown"].exists()
    assert files_b["json"].exists()
    assert files_b["markdown"].exists()
    assert files_comp["json"].exists()
    assert files_comp["markdown"].exists()

    md_content = files_comp["markdown"].read_text(encoding="utf-8")
    assert "RAG Experiment Comparison" in md_content
    assert "Metric Comparison & Deltas" in md_content
