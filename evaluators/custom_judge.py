"""Custom rubric evaluators and DeepEval G-Eval adapter."""

from __future__ import annotations

from dataclasses import dataclass

from evaluators.base_evaluator import BaseEvaluator, threshold_for
from models.evaluation_result import EvaluationResult
from models.test_case import GenAITestCase


@dataclass(frozen=True)
class Rubric:
    name: str
    threshold: float
    required_terms: tuple[str, ...]
    description: str


class RubricEvaluator(BaseEvaluator):
    def __init__(self, rubric: Rubric) -> None:
        self.rubric = rubric
        self.metric_name = rubric.name

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        actual = (test_case.actual_output or "").lower()
        matched = [term for term in self.rubric.required_terms if term.lower() in actual]
        score = len(matched) / len(self.rubric.required_terms) if self.rubric.required_terms else 1.0
        threshold = threshold_for(test_case, self.metric_name, self.rubric.threshold)
        return EvaluationResult(
            self.metric_name,
            score,
            threshold,
            score >= threshold,
            reason=f"Matched rubric terms: {matched}. {self.rubric.description}",
        )


class DeepEvalGEvalAdapter(BaseEvaluator):
    """Optional G-Eval adapter using DeepEval's current GEval/LLMTestCase APIs."""

    def __init__(self, name: str, evaluation_steps: list[str], threshold: float = 0.8) -> None:
        self.metric_name = name
        self.evaluation_steps = evaluation_steps
        self.default_threshold = threshold

    def evaluate(self, test_case: GenAITestCase) -> EvaluationResult:
        try:
            from deepeval.metrics import GEval
            from deepeval.test_case import LLMTestCase, LLMTestCaseParams
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Install deepeval to use DeepEvalGEvalAdapter") from exc

        metric = GEval(
            name=self.metric_name,
            evaluation_steps=self.evaluation_steps,
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=threshold_for(test_case, self.metric_name, self.default_threshold),
        )
        llm_test_case = LLMTestCase(
            input=test_case.input,
            actual_output=test_case.actual_output or "",
            expected_output=test_case.expected_output,
            retrieval_context=test_case.retrieved_context or None,
        )
        metric.measure(llm_test_case)
        return EvaluationResult(
            self.metric_name,
            float(metric.score or 0.0),
            float(metric.threshold),
            bool(metric.success),
            reason=str(getattr(metric, "reason", "")),
        )


COMPLETENESS_RUBRIC = Rubric(
    name="completeness",
    threshold=0.8,
    required_terms=("30 days", "receipt"),
    description="5=all required info; 1=fails to address the request.",
)

SUMMARIZATION_RUBRIC = Rubric(
    name="summarization_quality",
    threshold=0.8,
    required_terms=("Friday", "Priya", "QA"),
    description="Checks whether key meeting decisions and owners are preserved.",
)
