"""Compare current mock outputs against the regression baseline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clients.mock_agent_client import MockAgentClient
from evaluators.semantic_similarity import TokenSemanticSimilarityEvaluator
from models.api_models import AgentRequest
from models.test_case import GenAITestCase


def main() -> int:
    baseline = json.loads(Path("datasets/regression_baseline.json").read_text(encoding="utf-8"))
    client = MockAgentClient()
    evaluator = TokenSemanticSimilarityEvaluator()
    failures = []

    prompts = {
        "TC_CHAT_001": "What is your refund policy?",
        "TC_SUM_001": "Summarize the meeting notes: The team decided to ship the beta on Friday. Priya owns QA sign-off.",
    }
    for row in baseline["cases"]:
        response = client.generate(AgentRequest(prompt=prompts[row["test_id"]]))
        case = GenAITestCase(
            test_id=row["test_id"],
            category="regression",
            description="Regression baseline comparison",
            input=prompts[row["test_id"]],
            expected_output=row["expected_output"],
            actual_output=response.output,
            metrics=["semantic_similarity"],
            thresholds={"semantic_similarity": row["minimum_semantic_similarity"]},
        )
        result = evaluator.evaluate(case)
        if not result.passed:
            failures.append((row["test_id"], result.score))

    if failures:
        print(f"Baseline failures: {failures}")
        return 1
    print("Regression baseline comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
