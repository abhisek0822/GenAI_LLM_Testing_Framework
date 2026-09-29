"""Failure diagnosis for RAG and GenAI quality reports."""

from __future__ import annotations

from models.evaluation_result import EvaluationResult


def diagnose_failure(results: list[EvaluationResult]) -> str:
    scores = {result.metric_name: result.score for result in results}
    passed = {result.metric_name: result.passed for result in results}

    precision = scores.get("context_precision")
    recall = scores.get("context_recall")
    faithfulness = scores.get("faithfulness")
    correctness = scores.get("correctness") or scores.get("semantic_similarity")
    toxicity_passed = passed.get("toxicity", True)

    if not toxicity_passed:
        return "Safety issue: factual metrics may be acceptable, but toxicity checks failed."
    if precision is not None and recall is not None and (precision < 0.8 or recall < 0.8):
        if precision >= 0.8 > recall:
            return (
                "Retrieval issue: retrieved chunks are relevant, but recall is low. "
                "Investigate top-k, chunking, embeddings, query rewriting, filters, and reranking."
            )
        return "Retrieval issue: context precision or recall is low; inspect retrieval ranking and corpus coverage."
    if faithfulness is not None and faithfulness < 0.9:
        return "Generation issue: retrieved context appears adequate, but the answer is not faithful to it."
    if faithfulness is not None and correctness is not None and faithfulness >= 0.9 > correctness:
        return "Reference or source-data issue: the answer follows context but disagrees with the golden answer."
    return "General quality issue: inspect deterministic failures, prompt behavior, and golden data."
