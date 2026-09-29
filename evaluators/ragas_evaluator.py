"""Optional RAGAS adapter using modern sample/dataset field names."""

from __future__ import annotations

from evaluators.base_evaluator import BaseEvaluator
from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


class RagasEvaluator(BaseEvaluator):
    metric_name = "ragas"

    def __init__(self, metric_names: list[str] | None = None) -> None:
        self.metric_names = metric_names or [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        try:
            from ragas import evaluate
            from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithoutReference,
                LLMContextRecall,
                ResponseRelevancy,
            )
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install ragas to use RagasEvaluator") from exc

        metric_map = {
            "faithfulness": Faithfulness(),
            "answer_relevancy": ResponseRelevancy(),
            "context_precision": LLMContextPrecisionWithoutReference(),
            "context_recall": LLMContextRecall(),
        }
        sample = SingleTurnSample(
            user_input=test_case.input,
            response=test_case.actual_output or "",
            reference=test_case.expected_output,
            retrieved_contexts=test_case.retrieved_context,
        )
        dataset = EvaluationDataset(samples=[sample])
        result = evaluate(dataset, metrics=[metric_map[name] for name in self.metric_names])
        scores = result.to_pandas().iloc[0].to_dict()
        failing = {
            name: float(scores[name])
            for name in self.metric_names
            if float(scores[name]) < float(test_case.thresholds.get(name, 0.8))
        }
        score = 1.0 if not failing else min(failing.values())
        return EvaluationResult(
            self.metric_name,
            score,
            1.0,
            not failing,
            reason=f"Failing RAGAS metrics: {failing}" if failing else "",
            metadata=scores,
        )
