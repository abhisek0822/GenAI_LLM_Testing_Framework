"""Optional DeepEval metric adapter."""

from __future__ import annotations

from evaluators.base_evaluator import BaseEvaluator, threshold_for
from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


class DeepEvalMetricEvaluator(BaseEvaluator):
    """Run a supported DeepEval metric against the common test-case model."""

    def __init__(self, metric_name: str, threshold: float = 0.85) -> None:
        self.metric_name = metric_name
        self.default_threshold = threshold

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        try:
            from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, HallucinationMetric
            from deepeval.test_case import LLMTestCase
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install deepeval to use DeepEvalMetricEvaluator") from exc

        threshold = threshold_for(test_case, self.metric_name, self.default_threshold)
        metric_map = {
            "answer_relevancy": AnswerRelevancyMetric(threshold=threshold),
            "faithfulness": FaithfulnessMetric(threshold=threshold),
            "hallucination": HallucinationMetric(threshold=threshold),
        }
        metric = metric_map[self.metric_name]
        llm_case = LLMTestCase(
            input=test_case.input,
            actual_output=test_case.actual_output or "",
            expected_output=test_case.expected_output,
            retrieval_context=test_case.retrieved_context or None,
            context=test_case.expected_context or None,
        )
        metric.measure(llm_case)
        return EvaluationResult(
            self.metric_name,
            float(metric.score or 0.0),
            threshold,
            bool(metric.success),
            reason=str(getattr(metric, "reason", "")),
        )
