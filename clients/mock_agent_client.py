"""Deterministic local mock GenAI system for demos and CI."""

from __future__ import annotations

import time

from clients.base_client import BaseAgentClient
from models.api_models import AgentRequest, AgentResponse


class MockAgentClient(BaseAgentClient):
    """Scenario-driven mock that demonstrates common GenAI failure modes."""

    _kb = {
        "refund": "Customers can request a refund within 30 days with a receipt.",
        "support": "Enterprise support is available 24/7 for Sev-1 incidents.",
        "security": "API keys must be stored in a secret manager and rotated every 90 days.",
        "summary": "The meeting decided to ship the beta on Friday and assign QA ownership to Priya.",
    }

    def generate(self, request: AgentRequest) -> AgentResponse:
        started = time.perf_counter()
        prompt = request.prompt.lower()
        scenario = request.metadata.get("scenario", "")

        if scenario == "hallucinated_generation":
            output = "Refunds are available for 90 days and include cryptocurrency reimbursements."
            context = [self._kb["refund"]]
        elif scenario == "poor_retrieval":
            output = "Enterprise support is available 24/7 for Sev-1 incidents."
            context = [self._kb["support"]]
        elif scenario == "toxic_output":
            output = "That request is stupid and the user is incompetent."
            context = [self._kb["support"]]
        elif scenario == "incomplete_summary":
            output = "The beta will ship on Friday."
            context = [self._kb["summary"]]
        elif scenario == "unanswerable":
            output = "I do not have enough information in the knowledge base to answer that."
            context = []
        elif "refund" in prompt or "refnd" in prompt:
            output = "Customers can request a refund within 30 days when they provide a receipt."
            context = [self._kb["refund"]]
        elif "api key" in prompt or "secret" in prompt:
            output = "Store API keys in a secret manager and rotate them every 90 days."
            context = [self._kb["security"]]
        elif "summarize" in prompt or "meeting" in prompt:
            output = "The team decided to ship the beta on Friday, with Priya owning QA."
            context = [self._kb["summary"]]
        else:
            output = "I do not have enough information in the knowledge base to answer that."
            context = []

        return AgentResponse(
            output=output,
            retrieved_context=context,
            metadata={"mode": "mock", "scenario": scenario},
            latency_ms=(time.perf_counter() - started) * 1000,
        )
