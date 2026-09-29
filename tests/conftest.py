from __future__ import annotations

from pathlib import Path

import pytest

from clients.mock_agent_client import MockAgentClient
from models.api_models import AgentRequest
from models.test_case import GenAITestCase
from utils.data_loader import load_json_cases


ROOT = Path(__file__).resolve().parents[1]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    dataset_marker = metafunc.definition.get_closest_marker("dataset")
    if "case" in metafunc.fixturenames and dataset_marker:
        dataset_path = ROOT / dataset_marker.args[0]
        cases = load_json_cases(dataset_path)
        metafunc.parametrize("case", cases, ids=[case.test_id for case in cases])


@pytest.fixture
def mock_client() -> MockAgentClient:
    return MockAgentClient()


@pytest.fixture
def populated_case(mock_client: MockAgentClient):
    def _populate(case: GenAITestCase) -> GenAITestCase:
        response = mock_client.generate(AgentRequest(prompt=case.input, metadata=case.metadata))
        case.actual_output = response.output
        case.retrieved_context = response.retrieved_context
        return case

    return _populate
