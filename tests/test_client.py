import httpx
import pytest
import respx

from graph_mem.client import GraphitiClient


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://localhost:8000")


@respx.mock
@pytest.mark.asyncio
async def test_healthcheck(client):
    respx.get("http://localhost:8000/healthcheck").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )
    result = await client.healthcheck()
    assert result == {"status": "healthy"}


@respx.mock
@pytest.mark.asyncio
async def test_add_messages(client):
    route = respx.post("http://localhost:8000/messages").mock(
        return_value=httpx.Response(202, json={"message": "ok", "success": True})
    )
    result = await client.add_messages(
        group_id="user_profile",
        messages=[
            {
                "content": "I prefer dark mode",
                "role_type": "user",
                "role": "developer",
                "name": "preference",
            }
        ],
    )
    assert result["success"] is True
    assert route.calls[0].request.url == "http://localhost:8000/messages"
    body = route.calls[0].request.content
    import json
    parsed = json.loads(body)
    assert parsed["group_id"] == "user_profile"
    assert len(parsed["messages"]) == 1
    assert parsed["messages"][0]["content"] == "I prefer dark mode"


@respx.mock
@pytest.mark.asyncio
async def test_search(client):
    respx.post("http://localhost:8000/search").mock(
        return_value=httpx.Response(
            200,
            json={
                "facts": [
                    {
                        "uuid": "abc",
                        "name": "preference",
                        "fact": "developer prefers dark mode",
                        "valid_at": None,
                        "invalid_at": None,
                        "created_at": "2026-04-05T10:00:00Z",
                        "expired_at": None,
                    }
                ]
            },
        )
    )
    result = await client.search(query="dark mode", group_ids=["user_profile"])
    assert len(result["facts"]) == 1
    assert result["facts"][0]["fact"] == "developer prefers dark mode"


@respx.mock
@pytest.mark.asyncio
async def test_get_memory(client):
    respx.post("http://localhost:8000/get-memory").mock(
        return_value=httpx.Response(200, json={"facts": []})
    )
    result = await client.get_memory(
        group_id="project_graph-mem",
        messages=[{"content": "what was decided?", "role_type": "user", "role": None}],
    )
    assert result["facts"] == []


@respx.mock
@pytest.mark.asyncio
async def test_delete_group(client):
    respx.delete("http://localhost:8000/group/user_profile").mock(
        return_value=httpx.Response(200, json={"message": "Group deleted", "success": True})
    )
    result = await client.delete_group("user_profile")
    assert result["success"] is True


@respx.mock
@pytest.mark.asyncio
async def test_add_entity_node(client):
    respx.post("http://localhost:8000/entity-node").mock(
        return_value=httpx.Response(201, json={"uuid": "abc", "name": "test"})
    )
    result = await client.add_entity_node(
        uuid="abc",
        group_id="user_profile",
        name="test",
        summary="A test entity",
    )
    assert result["uuid"] == "abc"


@respx.mock
@pytest.mark.asyncio
async def test_get_episodes(client):
    respx.get("http://localhost:8000/episodes/user_profile").mock(
        return_value=httpx.Response(200, json=[])
    )
    result = await client.get_episodes(group_id="user_profile", last_n=5)
    assert result == []


@respx.mock
@pytest.mark.asyncio
async def test_client_with_api_key():
    client = GraphitiClient(base_url="http://localhost:8000", api_key="my-key")
    respx.get("http://localhost:8000/healthcheck").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )
    with respx.mock:
        result = await client.healthcheck()
    assert result == {"status": "healthy"}


@pytest.mark.asyncio
async def test_custom_timeout():
    client = GraphitiClient(base_url="http://localhost:8000", timeout=30.0)
    assert client._timeout == 30.0
    http = await client._get_client()
    assert http.timeout.read == 30.0
    await client.close()


@pytest.mark.asyncio
async def test_default_timeout():
    client = GraphitiClient(base_url="http://localhost:8000")
    assert client._timeout == 60.0
    http = await client._get_client()
    assert http.timeout.read == 60.0
    await client.close()


@pytest.mark.asyncio
async def test_close():
    client = GraphitiClient(base_url="http://localhost:8000")
    http = await client._get_client()
    assert not http.is_closed
    await client.close()
    assert client._http is None


@pytest.mark.asyncio
async def test_close_idempotent():
    client = GraphitiClient(base_url="http://localhost:8000")
    await client.close()  # no client opened yet — should not raise
    await client.close()  # still safe


@respx.mock
@pytest.mark.asyncio
async def test_async_context_manager():
    respx.get("http://localhost:8000/healthcheck").mock(
        return_value=httpx.Response(200, json={"status": "healthy"})
    )
    async with GraphitiClient(base_url="http://localhost:8000") as client:
        result = await client.healthcheck()
        assert result == {"status": "healthy"}
        http = client._http
    # after exiting, client should be closed
    assert client._http is None
    assert http.is_closed
