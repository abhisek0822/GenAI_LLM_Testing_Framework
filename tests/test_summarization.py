from __future__ import annotations

import pytest

from evaluators import EvaluationEngine
from models.test_case import GenAITestCase


@pytest.mark.dataset("datasets/summarization_cases.json")
def test_summarization_quality(case: GenAITestCase, populated_case) -> None:
    report = EvaluationEngine().evaluate(populated_case(case))
    expected_failure = "regression" in case.tags
    assert report.passed is not expected_failure
