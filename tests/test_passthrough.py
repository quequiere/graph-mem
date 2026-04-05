import pytest

from graph_mem.tools.passthrough import status, add_raw_memory, search_facts, search_entities, reset_memory
from graph_mem.client import GraphitiClient


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_status(client, monkeypatch):
    async def mock_healthcheck():
        return {"status": "healthy"}
    monkeypatch.setattr(client, "healthcheck", mock_healthcheck)
    result = await status(client)
    assert "healthy" in result


@pytest.mark.asyncio
async def test_add_raw_memory(client, monkeypatch):
    async def mock_add_messages(group_id, messages):
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    result = await add_raw_memory(
        client,
        content="test memory",
        group_id="user_profile",
        source_description="test",
    )
    assert "success" in result.lower() or "added" in result.lower()


@pytest.mark.asyncio
async def test_search_facts(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {
            "facts": [
                {
                    "uuid": "abc",
                    "name": "pref",
                    "fact": "prefers dark mode",
                    "valid_at": None,
                    "invalid_at": None,
                    "created_at": "2026-04-05T10:00:00Z",
                    "expired_at": None,
                }
            ]
        }
    monkeypatch.setattr(client, "search", mock_search)
    result = await search_facts(client, query="dark mode", group_ids=["user_profile"])
    assert "dark mode" in result


@pytest.mark.asyncio
async def test_search_entities(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": []}
    monkeypatch.setattr(client, "search", mock_search)
    result = await search_entities(client, query="Python", group_ids=["user_profile"])
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_reset_memory(client, monkeypatch):
    deleted = []
    async def mock_delete_group(group_id):
        deleted.append(group_id)
        return {"success": True, "message": "Group deleted"}
    monkeypatch.setattr(client, "delete_group", mock_delete_group)
    result = await reset_memory(client, group_ids=["user_profile", "project_test"])
    assert len(deleted) == 2
    assert "deleted" in result.lower() or "reset" in result.lower()
