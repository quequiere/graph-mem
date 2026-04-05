import pytest
from graph_mem.client import GraphitiClient
from graph_mem.tools.onboard import check_project, onboard_project


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_check_project_known(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": [
            {"uuid": "1", "name": "project", "fact": "graph-mem is a Claude Code memory plugin",
             "valid_at": None, "invalid_at": None, "created_at": "2026-04-05T10:00:00Z", "expired_at": None}
        ]}
    monkeypatch.setattr(client, "search", mock_search)
    result = await check_project(client, project_id="project_graph-mem")
    assert result["known"] is True
    assert len(result["facts"]) > 0


@pytest.mark.asyncio
async def test_check_project_unknown(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": []}
    monkeypatch.setattr(client, "search", mock_search)
    result = await check_project(client, project_id="project_unknown")
    assert result["known"] is False


@pytest.mark.asyncio
async def test_onboard_project(client, monkeypatch, tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# My Project\nA test project for unit tests.")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "my-project"\ndependencies = ["fastapi"]')

    calls = []
    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)

    result = await onboard_project(client, project_id="project_my-project", project_path=str(tmp_path), description="A test project")
    project_calls = [c for c in calls if c["group_id"] == "project_my-project"]
    assert len(project_calls) >= 1
    profile_calls = [c for c in calls if c["group_id"] == "user_profile"]
    assert len(profile_calls) >= 1


@pytest.mark.asyncio
async def test_onboard_project_no_readme(client, monkeypatch, tmp_path):
    calls = []
    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}
    monkeypatch.setattr(client, "add_messages", mock_add_messages)

    result = await onboard_project(client, project_id="project_bare", project_path=str(tmp_path))
    assert len(calls) >= 1
