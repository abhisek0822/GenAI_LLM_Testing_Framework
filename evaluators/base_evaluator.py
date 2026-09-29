"""Evaluator interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


class BaseEvaluator(ABC):
    metric_name: str

    @abstractmethod
    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        """Evaluate a populated test case."""


def threshold_for(test_case: GenAITestCase, metric_name: str, default: float) -> float:
    return float(test_case.thresholds.get(metric_name, default))
