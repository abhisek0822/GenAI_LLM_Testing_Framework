"""HTTP client for real GenAI APIs."""

from __future__ import annotations

import time
from typing import Any

import requests

from clients.base_client import BaseAgentClient
from config.settings import settings
from models.api_models import AgentRequest, AgentResponse
from utils.logger import get_logger

logger = get_logger(__name__)


class AgentClientError(RuntimeError):
    """Raised when the configured GenAI API cannot be called successfully."""


class HttpAgentClient(BaseAgentClient):
    def __init__(
        self,
        base_url: str = settings.genai_base_url,
        token: str | None = settings.genai_api_token,
        timeout_seconds: float = settings.genai_timeout_seconds,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout_seconds = timeout_seconds

    def generate(self, request: AgentRequest) -> AgentResponse:
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        started = time.perf_counter()
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={"prompt": request.prompt, "metadata": request.metadata},
                headers=headers,
                timeout=self.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except requests.Timeout as exc:
            raise AgentClientError("GenAI API request timed out") from exc
        except requests.ConnectionError as exc:
            raise AgentClientError("Could not connect to GenAI API") from exc
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else "unknown"
            raise AgentClientError(f"GenAI API returned HTTP {status}") from exc
        except ValueError as exc:
            raise AgentClientError("GenAI API returned invalid JSON") from exc

        logger.info("genai_api_call_complete")
        return AgentResponse(
            output=str(payload.get("output", "")),
            retrieved_context=list(payload.get("retrieved_context", [])),
            metadata=dict(payload.get("metadata", {})),
            status_code=response.status_code,
            latency_ms=latency_ms,
        )
