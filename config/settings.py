"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    """Configuration for system-under-test and evaluator integrations."""

    genai_mode: str = os.getenv("GENAI_MODE", "mock")
    genai_base_url: str = os.getenv("GENAI_BASE_URL", "http://localhost:8000")
    genai_api_token: str | None = os.getenv("GENAI_API_TOKEN") or None
    genai_timeout_seconds: float = float(os.getenv("GENAI_TIMEOUT_SECONDS", "10"))
    evaluator_api_key: str | None = os.getenv("EVALUATOR_API_KEY") or None
    evaluator_model: str = os.getenv("EVALUATOR_MODEL", "gpt-4o-mini")

    faithfulness_threshold: float = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.90"))
    relevancy_threshold: float = float(os.getenv("RELEVANCY_THRESHOLD", "0.85"))
    context_precision_threshold: float = float(os.getenv("CONTEXT_PRECISION_THRESHOLD", "0.80"))
    context_recall_threshold: float = float(os.getenv("CONTEXT_RECALL_THRESHOLD", "0.80"))
    toxicity_threshold: float = float(os.getenv("TOXICITY_THRESHOLD", "0.20"))


settings = Settings()
