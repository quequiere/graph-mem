"""Shared test fixtures."""

import pytest

from graph_mem.client import GraphitiClient


@pytest.fixture
def mock_client(monkeypatch):
    """A GraphitiClient that doesn't make real HTTP calls.
    Tests using this fixture should mock specific methods with monkeypatch.
    """
    return GraphitiClient(base_url="http://test:8000")
