"""
Automated RAG Evaluation System using Ragas metrics framework.
Provides datasets, metric evaluators, runners, and structured reports.
"""

from evaluation.datasets import (
    EvalDataset,
    EvalExample,
    ExpectedBehavior,
    QuestionCategory,
    load_eval_dataset,
)
from evaluation.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextPrecisionMetric,
    ContextRecallMetric,
    FaithfulnessMetric,
    MetricResult,
)
from evaluation.reports import ReportGenerator, ReportStore
from evaluation.runners import (
    ComparisonResult,
    EvaluationRunResult,
    EvaluationRunner,
    ExampleEvaluationResult,
    ExperimentComparator,
)

__all__ = [
    "EvalDataset",
    "EvalExample",
    "QuestionCategory",
    "ExpectedBehavior",
    "load_eval_dataset",
    "BaseMetric",
    "MetricResult",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    "EvaluationRunner",
    "EvaluationRunResult",
    "ExampleEvaluationResult",
    "ExperimentComparator",
    "ComparisonResult",
    "ReportGenerator",
    "ReportStore",
]
