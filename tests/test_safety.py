from __future__ import annotations

import pytest

from evaluators import EvaluationEngine
from models.test_case import GenAITestCase


@pytest.mark.dataset("datasets/safety_cases.json")
def test_safety_quality_gates(case: GenAITestCase, populated_case) -> None:
    report = EvaluationEngine().evaluate(populated_case(case))
    expected_failure = "toxicity" in case.tags
    assert report.passed is not expected_failure
