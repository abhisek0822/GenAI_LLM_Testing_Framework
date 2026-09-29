"""Lightweight offline semantic similarity based on token overlap."""

from __future__ import annotations

import re

from evaluators.base_evaluator import BaseEvaluator, threshold_for
from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


def _tokens(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 2}


class TokenSemanticSimilarityEvaluator(BaseEvaluator):
    """A deterministic stand-in for embeddings in local demo mode."""

    metric_name = "semantic_similarity"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        expected = _tokens(test_case.expected_output or "")
        actual = _tokens(test_case.actual_output or "")
        score = len(expected & actual) / len(expected | actual) if expected or actual else 1.0
        threshold = threshold_for(test_case, self.metric_name, 0.72)
        return EvaluationResult(self.metric_name, score, threshold, score >= threshold)
