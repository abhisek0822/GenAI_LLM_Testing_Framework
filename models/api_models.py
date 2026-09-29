"""API payload and response models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRequest:
    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    output: str
    retrieved_context: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    status_code: int = 200
    latency_ms: float = 0.0
