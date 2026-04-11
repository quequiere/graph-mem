import os

import pytest

from graph_mem.config import Settings, get_settings


def test_default_settings():
    settings = get_settings()
    assert settings.graphiti_url == "http://127.0.0.1:8000"
    assert settings.graphiti_api_key is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://remote:9000")
    monkeypatch.setenv("GRAPHITI_API_KEY", "secret-key")
    settings = get_settings()
    assert settings.graphiti_url == "http://remote:9000"
    assert settings.graphiti_api_key == "secret-key"
