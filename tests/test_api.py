from __future__ import annotations

from clients.mock_agent_client import MockAgentClient
from models.api_models import AgentRequest


def test_mock_client_returns_contract() -> None:
    response = MockAgentClient().generate(AgentRequest(prompt="What is your refund policy?"))
    assert response.status_code == 200
    assert response.output
    assert isinstance(response.retrieved_context, list)
    assert response.latency_ms >= 0
