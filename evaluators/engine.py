"""Evaluation engine and evaluator registry."""

from __future__ import annotations

from evaluators.base_evaluator import BaseEvaluator
from evaluators.custom_judge import COMPLETENESS_RUBRIC, SUMMARIZATION_RUBRIC, RubricEvaluator
from evaluators.deterministic import (
    ContainsEvaluator,
    ExactMatchEvaluator,
    JsonFieldEvaluator,
    ProhibitedStringsEvaluator,
    RegexEvaluator,
    RequiredKeywordsEvaluator,
)
from evaluators.diagnostics import diagnose_failure
from evaluators.safety_evaluator import AbstentionEvaluator, ToxicityEvaluator
from evaluators.semantic_similarity import TokenSemanticSimilarityEvaluator
from models.evaluation_result import EvaluationReport
from models.test_case import GenAITestCase


class EvaluationEngine:
    def __init__(self, evaluators: dict[str, BaseEvaluator] | None = None) -> None:
        self.evaluators = evaluators or default_evaluators()

    def evaluate(self, test_case: GenAITestCase) -> EvaluationReport:
        results = [self.evaluators[name].evaluate(test_case) for name in test_case.metrics]
        diagnosis = None if all(result.passed for result in results) else diagnose_failure(results)
        return EvaluationReport(test_id=test_case.test_id, results=results, diagnosis=diagnosis)


def default_evaluators() -> dict[str, BaseEvaluator]:
    return {
        "exact_match": ExactMatchEvaluator(),
        "contains": ContainsEvaluator(),
        "regex": RegexEvaluator(),
        "required_keywords": RequiredKeywordsEvaluator(),
        "prohibited_strings": ProhibitedStringsEvaluator(),
        "json_field_validation": JsonFieldEvaluator(),
        "semantic_similarity": TokenSemanticSimilarityEvaluator(),
        "toxicity": ToxicityEvaluator(),
        "abstention_correctness": AbstentionEvaluator(),
        "completeness": RubricEvaluator(COMPLETENESS_RUBRIC),
        "summarization_quality": RubricEvaluator(SUMMARIZATION_RUBRIC),
    }
