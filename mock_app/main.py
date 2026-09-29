"""Local FastAPI mock GenAI service."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from clients.mock_agent_client import MockAgentClient
from models.api_models import AgentRequest

app = FastAPI(title="Mock GenAI System Under Test")
client = MockAgentClient()


class GenerateRequest(BaseModel):
    prompt: str
    metadata: dict[str, str] = Field(default_factory=dict)


@app.post("/generate")
def generate(request: GenerateRequest) -> dict[str, object]:
    response = client.generate(AgentRequest(prompt=request.prompt, metadata=request.metadata))
    return {
        "output": response.output,
        "retrieved_context": response.retrieved_context,
        "metadata": response.metadata,
        "latency_ms": response.latency_ms,
    }
