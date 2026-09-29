"""Report writers for evaluation runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from models.evaluation_result import EvaluationReport


def write_json_report(reports: list[EvaluationReport], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump([asdict(report) for report in reports], file, indent=2)
