"""Unit tests for the SessionStart hook."""

import pytest

from graph_mem.hooks.session_start import _build_system_message, run


# --- _build_system_message ---


def test_build_system_message_with_facts():
    sections = {
        "profile": [{"fact": "developer uses Python"}],
        "project": [{"fact": "project uses FastAPI"}],
        "reminders": [{"fact": "Reminder: bump deps"}],
    }
    msg = _build_system_message(sections)
    assert "1 profile" in msg
    assert "1 project" in msg
    assert "1 reminder" in msg
    assert "developer uses Python" in msg
    assert "project uses FastAPI" in msg


def test_build_system_message_no_context():
    sections = {"profile": [], "project": [], "reminders": []}
    msg = _build_system_message(sections)
    assert "no prior context" in msg


def test_build_system_message_truncates_long_facts():
    long_fact = "x" * 150
    sections = {"profile": [{"fact": long_fact}], "project": [], "reminders": []}
    msg = _build_system_message(sections)
    assert "..." in msg
    # Should be truncated to 97 + "..."
    for line in msg.splitlines():
        if line.strip().startswith("- "):
            assert len(line.strip()) <= 102  # "  - " + 97 + "..."


def test_build_system_message_string_facts():
    """Facts can be plain strings (not dicts)."""
    sections = {"profile": ["I prefer dark mode"], "project": [], "reminders": []}
    msg = _build_system_message(sections)
    assert "I prefer dark mode" in msg


# --- run() ---


@pytest.mark.asyncio
async def test_run_with_context(monkeypatch):
    """run() returns context when Graphiti is reachable."""
    async def mock_get_context_sections(client, project_id):
        return {
            "profile": [{"fact": "developer uses TDD"}],
            "project": [],
            "reminders": [],
            "formatted": "## Developer Profile\n- developer uses TDD",
        }

    monkeypatch.setattr(
        "graph_mem.hooks.session_start.get_context_sections",
        mock_get_context_sections,
    )
    monkeypatch.setattr(
        "graph_mem.hooks.session_start.get_project_id",
        lambda: "project_test",
    )

    result = await run()

    assert "hookSpecificOutput" in result
    additional = result["hookSpecificOutput"]["additionalContext"]
    assert "Persistent knowledge graph memory" in additional
    assert "developer uses TDD" in additional

    assert "systemMessage" in result
    assert "1 profile" in result["systemMessage"]


@pytest.mark.asyncio
async def test_run_graphiti_down(monkeypatch):
    """run() returns instructions even when Graphiti is unreachable."""
    async def mock_get_context_sections(client, project_id):
        raise ConnectionError("Graphiti is down")

    monkeypatch.setattr(
        "graph_mem.hooks.session_start.get_context_sections",
        mock_get_context_sections,
    )
    monkeypatch.setattr(
        "graph_mem.hooks.session_start.get_project_id",
        lambda: "project_test",
    )

    result = await run()

    additional = result["hookSpecificOutput"]["additionalContext"]
    assert "Persistent knowledge graph memory" in additional
    # Should not contain recalled context
    assert "Recalled context" not in additional
    assert "Could not connect" in result["systemMessage"]


@pytest.mark.asyncio
async def test_run_no_context(monkeypatch):
    """run() handles empty context gracefully."""
    async def mock_get_context_sections(client, project_id):
        return {
            "profile": [],
            "project": [],
            "reminders": [],
            "formatted": "No context available yet for this session.",
        }

    monkeypatch.setattr(
        "graph_mem.hooks.session_start.get_context_sections",
        mock_get_context_sections,
    )
    monkeypatch.setattr(
        "graph_mem.hooks.session_start.get_project_id",
        lambda: "project_test",
    )

    result = await run()

    additional = result["hookSpecificOutput"]["additionalContext"]
    assert "Recalled context" not in additional
    assert "no prior context" in result["systemMessage"]
