import json
from pathlib import Path
from typing import Any, Dict, List

DATASET_PATH = Path(__file__).parent.parent.parent / "tests" / "data" / "qa_eval_dataset.json"


def load_qa_eval_dataset() -> List[Dict[str, Any]]:
    """Loads the benchmark question-answering evaluation dataset."""
    if not DATASET_PATH.exists():
        return []
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
