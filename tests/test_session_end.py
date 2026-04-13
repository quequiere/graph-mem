"""Unit tests for the SessionEnd (Stop) hook."""

import pytest

from graph_mem.hooks.session_end import run


@pytest.mark.asyncio
async def test_run_saves_summary(monkeypatch):
    """run() calls save_session with the summary and project_id."""
    calls = []

    async def mock_save_session(client, summary, project_id):
        calls.append({"summary": summary, "project_id": project_id})
        return "ok"

    monkeypatch.setattr("graph_mem.hooks.session_end.save_session", mock_save_session)
    monkeypatch.setattr("graph_mem.hooks.session_end.get_project_id", lambda: "project_test")

    await run("Implemented auth module and fixed 3 bugs")

    assert len(calls) == 1
    assert "auth module" in calls[0]["summary"]
    assert calls[0]["project_id"] == "project_test"


@pytest.mark.asyncio
async def test_run_empty_summary(monkeypatch):
    """run() does nothing when summary is empty."""
    calls = []

    async def mock_save_session(client, summary, project_id):
        calls.append(True)

    monkeypatch.setattr("graph_mem.hooks.session_end.save_session", mock_save_session)

    await run("")
    await run("   ")

    assert len(calls) == 0


@pytest.mark.asyncio
async def test_run_whitespace_only_summary(monkeypatch):
    """run() does nothing when summary is whitespace only."""
    calls = []

    async def mock_save_session(client, summary, project_id):
        calls.append(True)

    monkeypatch.setattr("graph_mem.hooks.session_end.save_session", mock_save_session)

    await run("\n\t  \n")

    assert len(calls) == 0
