"""Evaluation result models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    threshold: float
    passed: bool
    reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    test_id: str
    results: list[EvaluationResult]
    diagnosis: str | None = None

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)
