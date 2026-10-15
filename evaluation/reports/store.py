import json
from pathlib import Path
from typing import Dict, List, Optional
from app.core.logging import get_logger
from evaluation.reports.generator import ReportGenerator
from evaluation.runners.comparator import ComparisonResult
from evaluation.runners.evaluator import EvaluationRunResult

logger = get_logger(__name__)

DEFAULT_REPORTS_DIR = Path(__file__).parent / "output"


class ReportStore:
    """Persists evaluation results and benchmark comparisons to JSON and Markdown files."""

    def __init__(self, reports_dir: Optional[Path] = None):
        self.reports_dir = reports_dir or DEFAULT_REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def save_run(self, run: EvaluationRunResult) -> Dict[str, Path]:
        """Saves evaluation run as both JSON metadata and Markdown report."""
        json_path = self.reports_dir / f"{run.run_id}.json"
        md_path = self.reports_dir / f"{run.run_id}.md"

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(run.model_dump(), f, indent=2)

        # Save Markdown
        md_content = ReportGenerator.generate_run_markdown(run)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Saved evaluation run reports: {json_path.name}, {md_path.name}")
        return {"json": json_path, "markdown": md_path}

    def save_comparison(self, comparison: ComparisonResult) -> Dict[str, Path]:
        """Saves comparison result as both JSON and Markdown."""
        base_name = f"comparison_{comparison.config_a_name}_vs_{comparison.config_b_name}".lower()
        json_path = self.reports_dir / f"{base_name}.json"
        md_path = self.reports_dir / f"{base_name}.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(comparison.model_dump(), f, indent=2)

        md_content = ReportGenerator.generate_comparison_markdown(comparison)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Saved comparison reports: {json_path.name}, {md_path.name}")
        return {"json": json_path, "markdown": md_path}

    def list_saved_runs(self) -> List[Path]:
        """Returns all saved JSON evaluation runs sorted by modification time."""
        return sorted(self.reports_dir.glob("eval_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
