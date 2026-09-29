"""Deterministic evaluators for assertions that do not need an LLM judge."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

from evaluators.base_evaluator import BaseEvaluator, threshold_for
from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


class ExactMatchEvaluator(BaseEvaluator):
    metric_name = "exact_match"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        score = float((test_case.actual_output or "").strip() == (test_case.expected_output or "").strip())
        threshold = threshold_for(test_case, self.metric_name, 1.0)
        return EvaluationResult(self.metric_name, score, threshold, score >= threshold)


class ContainsEvaluator(BaseEvaluator):
    metric_name = "contains"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        expected = (test_case.expected_output or "").lower()
        actual = (test_case.actual_output or "").lower()
        score = float(expected in actual) if expected else 0.0
        threshold = threshold_for(test_case, self.metric_name, 1.0)
        return EvaluationResult(self.metric_name, score, threshold, score >= threshold)


class RegexEvaluator(BaseEvaluator):
    metric_name = "regex"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        pattern = str(test_case.metadata.get("regex", ""))
        score = float(bool(pattern and re.search(pattern, test_case.actual_output or "", re.IGNORECASE)))
        threshold = threshold_for(test_case, self.metric_name, 1.0)
        return EvaluationResult(self.metric_name, score, threshold, score >= threshold)


class RequiredKeywordsEvaluator(BaseEvaluator):
    metric_name = "required_keywords"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        keywords: Iterable[str] = test_case.metadata.get("required_keywords", [])
        values = list(keywords)
        actual = (test_case.actual_output or "").lower()
        matches = sum(1 for keyword in values if keyword.lower() in actual)
        score = matches / len(values) if values else 1.0
        threshold = threshold_for(test_case, self.metric_name, 0.8)
        return EvaluationResult(self.metric_name, score, threshold, score >= threshold)


class ProhibitedStringsEvaluator(BaseEvaluator):
    metric_name = "prohibited_strings"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        prohibited: Iterable[str] = test_case.metadata.get("prohibited_strings", [])
        actual = (test_case.actual_output or "").lower()
        found = [term for term in prohibited if term.lower() in actual]
        score = 0.0 if found else 1.0
        threshold = threshold_for(test_case, self.metric_name, 1.0)
        return EvaluationResult(
            self.metric_name,
            score,
            threshold,
            score >= threshold,
            reason=f"Found prohibited strings: {found}" if found else "",
        )


class JsonFieldEvaluator(BaseEvaluator):
    metric_name = "json_field_validation"

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        required_fields: list[str] = list(test_case.metadata.get("required_json_fields", []))
        try:
            payload = json.loads(test_case.actual_output or "{}")
        except json.JSONDecodeError as exc:
            return EvaluationResult(self.metric_name, 0.0, 1.0, False, reason=str(exc))
        missing = [field for field in required_fields if field not in payload]
        score = 1.0 if not missing else 0.0
        return EvaluationResult(
            self.metric_name,
            score,
            1.0,
            score == 1.0,
            reason=f"Missing fields: {missing}" if missing else "",
        )
