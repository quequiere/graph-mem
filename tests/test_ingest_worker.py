"""Unit tests for the background ingest worker."""

import pytest

from graph_mem.config import USER_PROFILE
from graph_mem.hooks._ingest_worker import ingest


@pytest.mark.asyncio
async def test_ingest_sends_message(monkeypatch):
    """ingest() calls add_messages with the correct group_id and content."""
    calls = []

    async def mock_add_messages(self, group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True}

    monkeypatch.setattr(
        "graph_mem.client.GraphitiClient.add_messages",
        mock_add_messages,
    )

    await ingest(USER_PROFILE, "I prefer dark mode")

    assert len(calls) == 1
    assert calls[0]["group_id"] == USER_PROFILE
    assert "dark mode" in calls[0]["messages"][0]["content"]
    assert calls[0]["messages"][0]["name"] == "auto-capture"


@pytest.mark.asyncio
async def test_ingest_project_scope(monkeypatch):
    """ingest() sends to the project group_id."""
    calls = []

    async def mock_add_messages(self, group_id, messages):
        calls.append({"group_id": group_id})
        return {"success": True}

    monkeypatch.setattr(
        "graph_mem.client.GraphitiClient.add_messages",
        mock_add_messages,
    )

    await ingest("project_github_com_user_repo", "We use PostgreSQL")

    assert len(calls) == 1
    assert calls[0]["group_id"] == "project_github_com_user_repo"
