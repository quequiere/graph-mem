import pytest
from graph_mem.client import GraphitiClient
from graph_mem.tools.profile import get_profile


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_get_profile(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {
            "facts": [
                {"uuid": "1", "name": "expertise", "fact": "developer has 10 years of TypeScript experience",
                 "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None},
                {"uuid": "2", "name": "preference", "fact": "developer prefers pnpm over npm",
                 "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None},
            ]
        }
    monkeypatch.setattr(client, "search", mock_search)
    result = await get_profile(client)
    assert "TypeScript" in result
    assert "pnpm" in result


@pytest.mark.asyncio
async def test_get_profile_empty(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": []}
    monkeypatch.setattr(client, "search", mock_search)
    result = await get_profile(client)
    assert "no profile" in result.lower() or "empty" in result.lower()
