import pytest
from graph_mem.client import GraphitiClient
from graph_mem.config import USER_PROFILE
from graph_mem.tools.reminders import add_reminder, get_reminders


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_add_reminder(client, monkeypatch):
    calls = []
    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    await add_reminder(client, content="Renew API key before April 15", group_id=USER_PROFILE)
    assert len(calls) == 1
    assert "Renew API key" in calls[0]["messages"][0]["content"]
    assert "REMINDER" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_add_reminder_project(client, monkeypatch):
    calls = []
    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    await add_reminder(client, content="Update deps before release", group_id="project_myapp")
    assert calls[0]["group_id"] == "project_myapp"


@pytest.mark.asyncio
async def test_get_reminders(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {
            "facts": [
                {"uuid": "1", "name": "reminder", "fact": "Reminder: renew API key before April 15",
                 "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None}
            ]
        }
    monkeypatch.setattr(client, "search", mock_search)
    result = await get_reminders(client, group_ids=[USER_PROFILE])
    assert "API key" in result


@pytest.mark.asyncio
async def test_get_reminders_all(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": []}
    monkeypatch.setattr(client, "search", mock_search)
    result = await get_reminders(client)
    assert "no" in result.lower()
