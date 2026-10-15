import pytest
from evaluation.datasets.schemas import EvalExample, ExpectedBehavior, QuestionCategory
from evaluation.metrics.answer_relevancy import AnswerRelevancyMetric
from evaluation.metrics.context_precision import ContextPrecisionMetric
from evaluation.metrics.context_recall import ContextRecallMetric
from evaluation.metrics.faithfulness import FaithfulnessMetric
from evaluation.runners.comparator import ExperimentComparator
from evaluation.runners.evaluator import EvaluationRunResult, ExampleEvaluationResult


@pytest.mark.asyncio
async def test_faithfulness_metric_supported_answer():
    metric = FaithfulnessMetric()
    example = EvalExample(
        id="test-1",
        question="How many peer approvals are needed?",
        category=QuestionCategory.STRAIGHTFORWARD,
        is_answerable=True,
        expected_answer="Two peer approvals are required.",
        expected_relevant_documents=["sample.txt"],
        expected_relevant_chunks=["All pull requests must have at least two peer approvals before merging."],
    )

    answer = "All pull requests require two peer approvals before merging."
    contexts = ["All pull requests must have at least two peer approvals before merging."]

    result = await metric.compute(
        example=example,
        answer=answer,
        retrieved_contexts=contexts,
        retrieved_chunks=[],
    )

    assert result.metric_name == "faithfulness"
    assert result.score > 0.8
    assert result.details["supported"] is True


@pytest.mark.asyncio
async def test_faithfulness_metric_unsupported_claim():
    metric = FaithfulnessMetric()
    example = EvalExample(
        id="test-2",
        question="What is the policy?",
        category=QuestionCategory.STRAIGHTFORWARD,
        is_answerable=True,
        expected_answer="Two peer approvals.",
        expected_relevant_documents=["sample.txt"],
    )

    # Introduces a completely fabricated claim
    answer = "All pull requests require two peer approvals. Employees may work from the moon on Sundays."
    contexts = ["All pull requests must have at least two peer approvals before merging."]

    result = await metric.compute(
        example=example,
        answer=answer,
        retrieved_contexts=contexts,
        retrieved_chunks=[],
    )

    assert result.score < 1.0
    assert len(result.details["unsupported_claims"]) > 0


@pytest.mark.asyncio
async def test_faithfulness_metric_expected_refusal():
    metric = FaithfulnessMetric()
    example = EvalExample(
        id="test-3",
        question="What is the pet policy?",
        category=QuestionCategory.UNSUPPORTED,
        is_answerable=False,
        expected_answer="I couldn't find sufficient information in the available knowledge base.",
        expected_behavior=ExpectedBehavior.REFUSE_INSUFFICIENT_EVIDENCE,
    )

    answer = "I couldn't find sufficient information in the available knowledge base to answer this question reliably."
    result = await metric.compute(
        example=example,
        answer=answer,
        retrieved_contexts=[],
        retrieved_chunks=[],
    )

    assert result.score == 1.0
    assert result.details["refused_appropriately"] is True


@pytest.mark.asyncio
async def test_context_precision_ranking():
    metric = ContextPrecisionMetric()
    example = EvalExample(
        id="test-4",
        question="What are deployment windows?",
        category=QuestionCategory.STRAIGHTFORWARD,
        is_answerable=True,
        expected_answer="Tuesday through Thursday.",
        expected_relevant_documents=["sample.txt"],
        expected_relevant_chunks=["Production deployments are permitted Tuesday through Thursday."],
    )

    class MockChunk:
        def __init__(self, filename, content):
            self.metadata = {"filename": filename}
            self.content = content

    # Case A: Relevant chunk at rank 1, irrelevant at rank 2
    chunks_good = [
        MockChunk("sample.txt", "Production deployments are permitted Tuesday through Thursday."),
        MockChunk("other.txt", "Irrelevant corporate text."),
    ]
    res_good = await metric.compute(
        example=example,
        answer="Tuesday through Thursday",
        retrieved_contexts=[],
        retrieved_chunks=chunks_good,
    )
    # Precision at rank 1 = 1/1 = 1.0; precision at rank 2 = 1/2 = 0.5. Average Precision = (1.0 * 1 + 0.5 * 0) / 1 = 1.0
    assert res_good.score == 1.0

    # Case B: Irrelevant chunk at rank 1, relevant at rank 2
    chunks_bad = [
        MockChunk("other.txt", "Irrelevant corporate text."),
        MockChunk("sample.txt", "Production deployments are permitted Tuesday through Thursday."),
    ]
    res_bad = await metric.compute(
        example=example,
        answer="Tuesday through Thursday",
        retrieved_contexts=[],
        retrieved_chunks=chunks_bad,
    )
    # Precision at rank 1 = 0/1 = 0; precision at rank 2 = 1/2 = 0.5. Average Precision = (0 * 0 + 0.5 * 1) / 1 = 0.5
    assert res_bad.score == 0.5
    assert res_good.score > res_bad.score


@pytest.mark.asyncio
async def test_context_recall():
    metric = ContextRecallMetric()
    example = EvalExample(
        id="test-5",
        question="Deployments and TLS?",
        category=QuestionCategory.MULTI_DOCUMENT,
        is_answerable=True,
        expected_answer="Tuesday through Thursday and mutual TLS.",
        expected_relevant_documents=["sample.txt", "sample.md"],
    )

    class MockChunk:
        def __init__(self, filename):
            self.metadata = {"filename": filename}
            self.content = "text"

    # Full recall: both documents present
    chunks_full = [MockChunk("sample.txt"), MockChunk("sample.md")]
    res_full = await metric.compute(
        example=example,
        answer="Both",
        retrieved_contexts=[],
        retrieved_chunks=chunks_full,
    )
    assert res_full.score == 1.0

    # Partial recall: only 1 of 2 documents retrieved
    chunks_partial = [MockChunk("sample.txt"), MockChunk("other.txt")]
    res_partial = await metric.compute(
        example=example,
        answer="Partial",
        retrieved_contexts=[],
        retrieved_chunks=chunks_partial,
    )
    assert res_partial.score == 0.5
    assert "sample.md" in res_partial.details["missed_documents"]


def test_experiment_comparator():
    run_a = EvaluationRunResult(
        run_id="run_a",
        timestamp="2026-10-15T12:00:00Z",
        config_name="Config_TopK_3",
        model="mock-llm",
        embedding_model="fastembed",
        retrieval_config={"top_k": 3},
        summary_metrics={"faithfulness": 0.95, "context_recall": 0.70, "context_precision": 0.90},
        category_metrics={},
        total_examples=10,
        total_latency_seconds=10.0,
        average_latency_seconds=1.0,
        results=[],
    )

    run_b = EvaluationRunResult(
        run_id="run_b",
        timestamp="2026-10-15T12:05:00Z",
        config_name="Config_TopK_5",
        model="mock-llm",
        embedding_model="fastembed",
        retrieval_config={"top_k": 5},
        summary_metrics={"faithfulness": 0.95, "context_recall": 0.85, "context_precision": 0.82},
        category_metrics={},
        total_examples=10,
        total_latency_seconds=12.0,
        average_latency_seconds=1.2,
        results=[],
    )

    comp = ExperimentComparator.compare(run_a, run_b)
    assert comp.config_a_name == "Config_TopK_3"
    assert comp.config_b_name == "Config_TopK_5"
    assert comp.metric_deltas["context_recall"].delta == 0.15
    assert comp.metric_deltas["context_precision"].delta == -0.08
    assert comp.latency_delta_seconds == 0.2
    assert len(comp.key_findings) > 0
