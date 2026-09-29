"""Common internal model for GenAI test scenarios."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GenAITestCase:
    test_id: str
    category: str
    description: str
    input: str
    expected_output: str | None = None
    expected_context: list[str] = field(default_factory=list)
    retrieved_context: list[str] = field(default_factory=list)
    actual_output: str | None = None
    metrics: list[str] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    risk_level: str = "medium"
    expected_behavior: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenAITestCase":
        return cls(**payload)
