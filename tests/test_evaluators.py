from __future__ import annotations

from evaluators.deterministic import RequiredKeywordsEvaluator
from evaluators.diagnostics import diagnose_failure
from evaluators.safety_evaluator import ToxicityEvaluator
from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


def test_required_keywords_scores_partial_matches() -> None:
    case = GenAITestCase(
        test_id="TC_UNIT_001",
        category="unit",
        description="partial keyword match",
        input="",
        actual_output="Refunds require a receipt.",
        metadata={"required_keywords": ["30 days", "receipt"]},
        metrics=["required_keywords"],
    )
    result = RequiredKeywordsEvaluator().evaluate(case)
    assert result.score == 0.5
    assert not result.passed


def test_toxicity_uses_upper_bound_threshold() -> None:
    case = GenAITestCase(
        test_id="TC_UNIT_002",
        category="unit",
        description="toxicity",
        input="",
        actual_output="That is stupid.",
        metrics=["toxicity"],
    )
    result = ToxicityEvaluator().evaluate(case)
    assert result.score > result.threshold
    assert not result.passed


def test_diagnosis_identifies_generation_issue() -> None:
    diagnosis = diagnose_failure(
        [
            EvaluationResult("context_precision", 0.9, 0.8, True),
            EvaluationResult("context_recall", 0.9, 0.8, True),
            EvaluationResult("faithfulness", 0.4, 0.9, False),
        ]
    )
    assert "Generation issue" in diagnosis
