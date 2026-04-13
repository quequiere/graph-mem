

from graph_mem.config import get_settings


def test_default_settings():
    settings = get_settings()
    # 127.0.0.1 (not localhost) — on Windows, 'localhost' resolves to IPv6 ::1
    # but the Graphiti container only listens on IPv4.
    assert settings.graphiti_url == "http://127.0.0.1:8000"
    assert settings.graphiti_api_key is None
    assert settings.graphiti_timeout == 60.0


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://remote:9000")
    monkeypatch.setenv("GRAPHITI_API_KEY", "secret-key")
    monkeypatch.setenv("GRAPHITI_TIMEOUT", "120.0")
    settings = get_settings()
    assert settings.graphiti_url == "http://remote:9000"
    assert settings.graphiti_api_key == "secret-key"
    assert settings.graphiti_timeout == 120.0
