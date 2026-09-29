"""Client abstraction for GenAI applications under test."""

from __future__ import annotations

from abc import ABC, abstractmethod

from models.api_models import AgentRequest, AgentResponse


class BaseAgentClient(ABC):
    @abstractmethod
    def generate(self, request: AgentRequest) -> AgentResponse:
        """Send a prompt to the system under test."""
