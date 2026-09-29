from __future__ import annotations

import pytest

from evaluators import EvaluationEngine
from models.test_case import GenAITestCase


@pytest.mark.dataset("datasets/chatbot_cases.json")
def test_chatbot_quality_gates(case: GenAITestCase, populated_case) -> None:
    report = EvaluationEngine().evaluate(populated_case(case))
    assert report.passed, report.diagnosis
