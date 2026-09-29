"""Dataset loading utilities."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from models.test_case import GenAITestCase


def load_json_cases(path: str | Path) -> list[GenAITestCase]:
    with Path(path).open(encoding="utf-8") as file:
        payload = json.load(file)
    rows = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    return [GenAITestCase.from_dict(row) for row in rows]


def load_csv_cases(path: str | Path) -> list[GenAITestCase]:
    """Example extension point for CSV-backed datasets."""
    with Path(path).open(newline="", encoding="utf-8") as file:
        return [GenAITestCase.from_dict(dict(row)) for row in csv.DictReader(file)]
