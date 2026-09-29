from __future__ import annotations

import pytest

from evaluators import EvaluationEngine
from models.test_case import GenAITestCase


@pytest.mark.dataset("datasets/rag_cases.json")
def test_rag_quality_gates(case: GenAITestCase, populated_case) -> None:
    report = EvaluationEngine().evaluate(populated_case(case))
    expected_failure = "generation_failure" in case.tags or "retrieval_failure" in case.tags
    assert report.passed is not expected_failure
    if expected_failure:
        assert report.diagnosis
