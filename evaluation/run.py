import argparse
import asyncio
from pathlib import Path
import sys
from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger
from app.ingestion.embeddings.factory import get_embedder
from evaluation.datasets.loader import (
    load_eval_dataset,
    seed_eval_documents_if_needed,
)
from evaluation.reports.generator import ReportGenerator
from evaluation.reports.store import ReportStore
from evaluation.runners.comparator import ExperimentComparator
from evaluation.runners.evaluator import EvaluationRunner

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Automated RAG Evaluation Pipeline (Phase 8)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        default=True,
        help="Run comparative experiment between Configuration A and B (default: True)",
    )
    parser.add_argument(
        "--no-compare",
        dest="compare",
        action="store_false",
        help="Run only single configuration",
    )
    parser.add_argument(
        "--config-a-name",
        type=str,
        default="Config_A_TopK_3",
        help="Name for Configuration A",
    )
    parser.add_argument(
        "--top-k-a",
        type=int,
        default=3,
        help="Retrieval top_k for Configuration A (default: 3)",
    )
    parser.add_argument(
        "--config-b-name",
        type=str,
        default="Config_B_TopK_5",
        help="Name for Configuration B",
    )
    parser.add_argument(
        "--top-k-b",
        type=int,
        default=5,
        help="Retrieval top_k for Configuration B (default: 5)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Retrieval similarity threshold (default: 0.35)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to custom evaluation dataset JSON",
    )
    parser.add_argument(
        "--reports-dir",
        type=str,
        default=None,
        help="Directory to save evaluation reports",
    )
    return parser.parse_args()


async def main_async():
    args = parse_args()
    print("=" * 70)
    print("🚀 AUTOMATED RAG EVALUATION HARNESS (PHASE 8)")
    print("=" * 70)
    print(f"Model: {settings.LLM_MODEL} | Embedding: {settings.EMBEDDING_MODEL_NAME}")
    print(f"Database: {settings.POSTGRES_DB} on {settings.POSTGRES_SERVER}:{settings.POSTGRES_PORT}")
    print("=" * 70)

    dataset_path = Path(args.dataset) if args.dataset else None
    dataset = load_eval_dataset(dataset_path)

    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    store = ReportStore(reports_dir=reports_dir)

    embedder = get_embedder()

    # Step 1: Ensure benchmark documents are seeded in the database
    async with AsyncSessionLocal() as session:
        print("\n[1/4] Verifying benchmark document fixtures in knowledge base...")
        seeded = await seed_eval_documents_if_needed(session=session, embedder=embedder)
        print(f"      Seeded {seeded} new document(s) into database.")

    # Step 2: Run Configuration A
    print(f"\n[2/4] Running Configuration A: '{args.config_a_name}' (top_k={args.top_k_a})...")
    async with AsyncSessionLocal() as session:
        runner_a = EvaluationRunner(session=session, embedder=embedder)
        run_a = await runner_a.run(
            dataset=dataset,
            config_name=args.config_a_name,
            top_k=args.top_k_a,
            similarity_threshold=args.similarity_threshold,
        )
    files_a = store.save_run(run_a)
    print(f"      Configuration A completed in {run_a.total_latency_seconds}s.")
    print(f"      Report saved: {files_a['markdown']}")

    if not args.compare:
        print("\n" + "=" * 70)
        print(ReportGenerator.generate_run_markdown(run_a))
        return

    # Step 3: Run Configuration B
    print(f"\n[3/4] Running Configuration B: '{args.config_b_name}' (top_k={args.top_k_b})...")
    async with AsyncSessionLocal() as session:
        runner_b = EvaluationRunner(session=session, embedder=embedder)
        run_b = await runner_b.run(
            dataset=dataset,
            config_name=args.config_b_name,
            top_k=args.top_k_b,
            similarity_threshold=args.similarity_threshold,
        )
    files_b = store.save_run(run_b)
    print(f"      Configuration B completed in {run_b.total_latency_seconds}s.")
    print(f"      Report saved: {files_b['markdown']}")

    # Step 4: Compare Configurations
    print("\n[4/4] Comparing Configuration A vs. Configuration B...")
    comparison = ExperimentComparator.compare(run_a=run_a, run_b=run_b)
    files_comp = store.save_comparison(comparison)
    print(f"      Comparison report saved: {files_comp['markdown']}")

    # Print summary to console
    print("\n" + "=" * 70)
    print(ReportGenerator.generate_comparison_markdown(comparison))
    print("=" * 70)
    print(f"🎉 Evaluation and comparative experiment finished successfully!")
    print(f"   Artifacts saved to: {store.reports_dir}")


def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nEvaluation canceled by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
