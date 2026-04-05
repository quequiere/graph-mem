import pytest
from graph_mem.client import GraphitiClient
from graph_mem.tools.context import get_context


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_get_context(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        if group_ids == ["user_profile"]:
            if "reminder" in query.lower():
                return {"facts": [
                    {"uuid": "r1", "name": "reminder", "fact": "Reminder: renew API key before April 15",
                     "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None}
                ]}
            return {"facts": [
                {"uuid": "1", "name": "expertise", "fact": "developer has 10 years TypeScript",
                 "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None}
            ]}
        if "project_" in (group_ids or [""])[0]:
            if "reminder" in query.lower():
                return {"facts": []}
            return {"facts": [
                {"uuid": "2", "name": "stack", "fact": "project uses Python and FastMCP",
                 "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None}
            ]}
        return {"facts": []}
    monkeypatch.setattr(client, "search", mock_search)
    result = await get_context(client, project_id="project_graph-mem")
    assert "TypeScript" in result
    assert "Python" in result or "FastMCP" in result
    assert "API key" in result


@pytest.mark.asyncio
async def test_get_context_no_data(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": []}
    monkeypatch.setattr(client, "search", mock_search)
    result = await get_context(client, project_id="project_new")
    assert isinstance(result, str)
