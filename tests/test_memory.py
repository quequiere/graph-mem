import pytest
from graph_mem.client import GraphitiClient
from graph_mem.config import USER_PROFILE
from graph_mem.tools.memory import save_memory, save_session


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_save_memory(client, monkeypatch):
    calls = []
    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    await save_memory(client, content="I prefer dark mode", group_id=USER_PROFILE)
    assert len(calls) == 1
    assert calls[0]["group_id"] == USER_PROFILE
    assert "dark mode" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_save_session(client, monkeypatch):
    calls = []
    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    await save_session(
        client,
        summary="Implemented the auth module and fixed 3 bugs",
        project_id="project_github.com/user/repo",
    )
    assert len(calls) == 1
    assert calls[0]["group_id"] == "project_github.com/user/repo"
    assert "auth module" in calls[0]["messages"][0]["content"]
