# graph-mem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the graph-mem MCP server that provides persistent developer memory via Graphiti's knowledge graph.

**Architecture:** Local Python MCP server (FastMCP, stdio transport) that calls Graphiti's REST API over HTTP using httpx. Installed via pip/uvx, configured with a single `GRAPHITI_URL` env var. Custom tools implement business logic (context injection, project onboarding, session capture); passthrough tools re-expose Graphiti's search and admin capabilities.

**Tech Stack:** Python 3.11+, FastMCP (`mcp` package), httpx, hatchling (build)

**Design spec:** `.doc/specs/2026-04-05-graph-mem-design.md`

---

## Graphiti REST API Reference

Gathered from Graphiti source code (`server/graph_service/`). All requests are JSON.

### POST /messages (add episodes)

```python
# Request
{
    "group_id": str,           # required
    "messages": [              # required, list of Message
        {
            "content": str,            # required
            "role_type": "user"|"assistant"|"system",  # required
            "role": str | None,        # custom role name
            "name": str,               # episode label, default ""
            "uuid": str | None,        # optional client UUID
            "timestamp": datetime,     # default: now
            "source_description": str  # default ""
        }
    ]
}
# Response: 202 {"message": "Messages added to processing queue", "success": true}
```

### POST /search (search facts)

```python
# Request
{
    "query": str,                      # required
    "group_ids": list[str] | None,     # optional filter
    "max_facts": int                   # default 10
}
# Response: 200
{
    "facts": [
        {
            "uuid": str,
            "name": str,
            "fact": str,
            "valid_at": datetime | None,
            "invalid_at": datetime | None,
            "created_at": datetime,
            "expired_at": datetime | None
        }
    ]
}
```

### POST /get-memory (search with message context)

```python
# Request
{
    "group_id": str,                   # required (single, not list)
    "max_facts": int,                  # default 10
    "center_node_uuid": str | None,    # required field (can be null)
    "messages": list[Message]          # required
}
# Response: 200 {"facts": [FactResult...]}
```

### POST /entity-node (create entity)

```python
# Request
{
    "uuid": str,       # required
    "group_id": str,   # required
    "name": str,       # required
    "summary": str     # default ""
}
# Response: 201 (node object)
```

### DELETE /group/{group_id}

```python
# Response: 200 {"message": "Group deleted", "success": true}
```

### GET /healthcheck

```python
# Response: 200 {"status": "healthy"}
```

### GET /episodes/{group_id}?last_n=N

```python
# Response: 200 (list of episodes)
```

---

## File Structure

```
graph-mem/
├── src/
│   └── graph_mem/
│       ├── __init__.py            # Package init, version
│       ├── server.py              # FastMCP server, tool registration, main()
│       ├── client.py              # GraphitiClient: async httpx wrapper for REST API
│       ├── config.py              # Settings from env vars
│       ├── project_id.py          # Git remote URL -> project identifier
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── context.py         # get_context
│       │   ├── onboard.py         # onboard_project, check_project
│       │   ├── memory.py          # save_session, save_memory
│       │   ├── profile.py         # get_profile
│       │   ├── reminders.py       # add_reminder, get_reminders
│       │   └── passthrough.py     # add_raw_memory, search_entities, search_facts, reset_memory, status
│       └── hooks/
│           ├── __init__.py
│           ├── session_start.py   # SessionStart hook script
│           └── session_end.py     # SessionEnd (Stop) hook script
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures (mock GraphitiClient, etc.)
│   ├── test_client.py             # GraphitiClient unit tests
│   ├── test_config.py             # Config loading tests
│   ├── test_project_id.py         # Project identifier tests
│   ├── test_passthrough.py        # Passthrough tools tests
│   ├── test_memory.py             # save_session, save_memory tests
│   ├── test_profile.py            # get_profile tests
│   ├── test_reminders.py          # add_reminder, get_reminders tests
│   ├── test_context.py            # get_context tests
│   └── test_onboard.py            # onboard_project, check_project tests
├── skills/
│   └── graph-mem/
│       └── SKILL.md               # Skill for guiding CLI agents
├── pyproject.toml
├── docker-compose.yml             # Graphiti + Neo4j for local dev
└── README.md                      # Already exists
```

---

## Task 1: Project scaffolding and config

**Files:**
- Create: `pyproject.toml`
- Create: `src/graph_mem/__init__.py`
- Create: `src/graph_mem/config.py`
- Create: `tests/__init__.py`
- Create: `tests/test_config.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "graph-mem"
version = "0.1.0"
description = "Persistent intelligent memory for AI coding assistants using knowledge graphs"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.0.0",
    "httpx>=0.27.0",
]

[project.scripts]
graph-mem = "graph_mem.server:main"

[tool.hatch.build.targets.wheel]
packages = ["src/graph_mem"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "respx>=0.22",
]
```

- [ ] **Step 2: Create package init**

`src/graph_mem/__init__.py`:

```python
"""graph-mem: Persistent intelligent memory for AI coding assistants."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Write the failing test for config**

`tests/__init__.py`: empty file.

`tests/conftest.py`:

```python
"""Shared test fixtures."""
```

`tests/test_config.py`:

```python
import os

import pytest

from graph_mem.config import Settings, get_settings


def test_default_settings():
    settings = get_settings()
    assert settings.graphiti_url == "http://localhost:8000"
    assert settings.graphiti_api_key is None


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("GRAPHITI_URL", "http://remote:9000")
    monkeypatch.setenv("GRAPHITI_API_KEY", "secret-key")
    settings = get_settings()
    assert settings.graphiti_url == "http://remote:9000"
    assert settings.graphiti_api_key == "secret-key"
```

- [ ] **Step 4: Run test to verify it fails**

Run: `cd /c/src/graph-mem && pip install -e ".[dev]" && pytest tests/test_config.py -v`
Expected: FAIL (cannot import `graph_mem.config`)

- [ ] **Step 5: Implement config**

`src/graph_mem/config.py`:

```python
"""Configuration from environment variables."""

import os
from dataclasses import dataclass


@dataclass
class Settings:
    graphiti_url: str = "http://localhost:8000"
    graphiti_api_key: str | None = None


def get_settings() -> Settings:
    return Settings(
        graphiti_url=os.environ.get("GRAPHITI_URL", "http://localhost:8000"),
        graphiti_api_key=os.environ.get("GRAPHITI_API_KEY"),
    )
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml src/ tests/
git commit -m "feat: project scaffolding with config module"
```

---

## Task 2: Graphiti HTTP client

**Files:**
- Create: `src/graph_mem/client.py`
- Create: `tests/test_client.py`

The client wraps all Graphiti REST API calls. All tools will use this client. We use `respx` to mock httpx in tests.

- [ ] **Step 1: Write the failing tests for GraphitiClient**

`tests/test_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_client.py -v`
Expected: FAIL (cannot import `graph_mem.client`)

- [ ] **Step 3: Implement GraphitiClient**

`src/graph_mem/client.py`:

```python
"""Async HTTP client for the Graphiti REST API."""

from typing import Any

import httpx


class GraphitiClient:
    """Wraps all Graphiti REST API calls."""

    def __init__(self, base_url: str = "http://localhost:8000", api_key: str | None = None):
        self.base_url = base_url.rstrip("/")
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._headers = headers

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, headers=self._headers, timeout=30.0)

    async def healthcheck(self) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.get("/healthcheck")
            resp.raise_for_status()
            return resp.json()

    async def add_messages(
        self,
        group_id: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/messages",
                json={"group_id": group_id, "messages": messages},
            )
            resp.raise_for_status()
            return resp.json()

    async def search(
        self,
        query: str,
        group_ids: list[str] | None = None,
        max_facts: int = 10,
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/search",
                json={"query": query, "group_ids": group_ids, "max_facts": max_facts},
            )
            resp.raise_for_status()
            return resp.json()

    async def get_memory(
        self,
        group_id: str,
        messages: list[dict[str, Any]],
        max_facts: int = 10,
        center_node_uuid: str | None = None,
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/get-memory",
                json={
                    "group_id": group_id,
                    "max_facts": max_facts,
                    "center_node_uuid": center_node_uuid,
                    "messages": messages,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def add_entity_node(
        self,
        uuid: str,
        group_id: str,
        name: str,
        summary: str = "",
    ) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.post(
                "/entity-node",
                json={"uuid": uuid, "group_id": group_id, "name": name, "summary": summary},
            )
            resp.raise_for_status()
            return resp.json()

    async def delete_group(self, group_id: str) -> dict[str, Any]:
        async with self._client() as client:
            resp = await client.delete(f"/group/{group_id}")
            resp.raise_for_status()
            return resp.json()

    async def get_episodes(
        self,
        group_id: str,
        last_n: int = 10,
    ) -> Any:
        async with self._client() as client:
            resp = await client.get(f"/episodes/{group_id}", params={"last_n": last_n})
            resp.raise_for_status()
            return resp.json()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_client.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph_mem/client.py tests/test_client.py
git commit -m "feat: GraphitiClient HTTP wrapper for REST API"
```

---

## Task 3: Project identifier utility

**Files:**
- Create: `src/graph_mem/project_id.py`
- Create: `tests/test_project_id.py`

Derives a stable project identifier from git remote URL or directory name fallback.

- [ ] **Step 1: Write the failing tests**

`tests/test_project_id.py`:

```python
import subprocess
from unittest.mock import patch

import pytest

from graph_mem.project_id import get_project_id, normalize_git_url


def test_normalize_https_url():
    assert normalize_git_url("https://github.com/user/repo.git") == "github.com/user/repo"


def test_normalize_ssh_url():
    assert normalize_git_url("git@github.com:user/repo.git") == "github.com/user/repo"


def test_normalize_https_no_git_suffix():
    assert normalize_git_url("https://github.com/user/repo") == "github.com/user/repo"


def test_normalize_ssh_no_git_suffix():
    assert normalize_git_url("git@gitlab.com:org/project") == "gitlab.com/org/project"


def test_get_project_id_with_remote(tmp_path):
    # Set up a git repo with a remote
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    subprocess.run(
        ["git", "remote", "add", "origin", "https://github.com/user/my-project.git"],
        cwd=tmp_path,
        capture_output=True,
    )
    project_id = get_project_id(str(tmp_path))
    assert project_id == "project_github.com/user/my-project"


def test_get_project_id_no_remote(tmp_path):
    subprocess.run(["git", "init"], cwd=tmp_path, capture_output=True)
    project_id = get_project_id(str(tmp_path))
    assert project_id == f"project_{tmp_path.name}"


def test_get_project_id_no_git(tmp_path):
    project_id = get_project_id(str(tmp_path))
    assert project_id == f"project_{tmp_path.name}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_project_id.py -v`
Expected: FAIL (cannot import `graph_mem.project_id`)

- [ ] **Step 3: Implement project_id**

`src/graph_mem/project_id.py`:

```python
"""Derive a stable project identifier from git remote or directory name."""

import os
import re
import subprocess


def normalize_git_url(url: str) -> str:
    """Normalize a git remote URL to 'host/owner/repo' form."""
    # SSH: git@github.com:user/repo.git
    ssh_match = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"
    # HTTPS: https://github.com/user/repo.git
    https_match = re.match(r"https?://([^/]+)/(.+?)(?:\.git)?$", url)
    if https_match:
        return f"{https_match.group(1)}/{https_match.group(2)}"
    return url


def get_project_id(project_path: str | None = None) -> str:
    """Get the project group_id for Graphiti.

    Uses git remote origin URL if available, falls back to directory name.
    Returns: 'project_{identifier}'
    """
    path = project_path or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            normalized = normalize_git_url(result.stdout.strip())
            return f"project_{normalized}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return f"project_{os.path.basename(os.path.abspath(path))}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_project_id.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/graph_mem/project_id.py tests/test_project_id.py
git commit -m "feat: project identifier from git remote URL"
```

---

## Task 4: MCP server skeleton + passthrough tools

**Files:**
- Create: `src/graph_mem/server.py`
- Create: `src/graph_mem/tools/__init__.py`
- Create: `src/graph_mem/tools/passthrough.py`
- Create: `tests/test_passthrough.py`

The server skeleton creates the FastMCP instance, initializes the GraphitiClient, and registers tools. Passthrough tools are thin wrappers over the client.

- [ ] **Step 1: Write the failing tests for passthrough tools**

`tests/conftest.py` (update):

```python
"""Shared test fixtures."""

import pytest

from graph_mem.client import GraphitiClient


@pytest.fixture
def mock_client(monkeypatch):
    """A GraphitiClient that doesn't make real HTTP calls.

    Tests using this fixture should mock specific methods with monkeypatch.
    """
    return GraphitiClient(base_url="http://test:8000")
```

`tests/test_passthrough.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_passthrough.py -v`
Expected: FAIL (cannot import `graph_mem.tools.passthrough`)

- [ ] **Step 3: Implement passthrough tools**

`src/graph_mem/tools/__init__.py`:

```python
"""graph-mem MCP tools."""
```

`src/graph_mem/tools/passthrough.py`:

```python
"""Passthrough tools: thin wrappers over Graphiti REST API."""

import json

from graph_mem.client import GraphitiClient


def _format_facts(facts: list[dict]) -> str:
    """Format a list of facts into readable text."""
    if not facts:
        return "No results found."
    lines = []
    for f in facts:
        status = ""
        if f.get("invalid_at"):
            status = " [SUPERSEDED]"
        elif f.get("expired_at"):
            status = " [EXPIRED]"
        lines.append(f"- {f['fact']}{status}")
    return "\n".join(lines)


async def status(client: GraphitiClient) -> str:
    """Check if Graphiti is running and healthy."""
    try:
        result = await client.healthcheck()
        return f"Graphiti status: {result.get('status', 'unknown')}"
    except Exception as e:
        return f"Graphiti is not reachable: {e}"


async def add_raw_memory(
    client: GraphitiClient,
    content: str,
    group_id: str,
    name: str = "",
    source_description: str = "",
) -> str:
    """Add an episode directly to the knowledge graph."""
    await client.add_messages(
        group_id=group_id,
        messages=[
            {
                "content": content,
                "role_type": "system",
                "role": "graph-mem",
                "name": name,
                "source_description": source_description,
            }
        ],
    )
    return "Memory added to processing queue."


async def search_facts(
    client: GraphitiClient,
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> str:
    """Search for relationships between entities in the knowledge graph."""
    result = await client.search(query=query, group_ids=group_ids, max_facts=max_facts)
    return _format_facts(result.get("facts", []))


async def search_entities(
    client: GraphitiClient,
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> str:
    """Search for entities by semantic query.

    Note: Uses fact search under the hood since the Graphiti REST API
    does not expose a dedicated node search endpoint.
    """
    result = await client.search(query=query, group_ids=group_ids, max_facts=max_facts)
    return _format_facts(result.get("facts", []))


async def reset_memory(
    client: GraphitiClient,
    group_ids: list[str],
) -> str:
    """Purge all data for the given group IDs. DANGEROUS - use with caution."""
    for gid in group_ids:
        await client.delete_group(gid)
    return f"Deleted data for groups: {', '.join(group_ids)}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_passthrough.py -v`
Expected: 5 passed

- [ ] **Step 5: Create the MCP server skeleton**

`src/graph_mem/server.py`:

```python
"""graph-mem MCP server."""

from mcp.server.fastmcp import FastMCP

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.tools import passthrough

mcp = FastMCP("graph-mem")

# Initialize client at module level - tools will use this
_settings = get_settings()
_client = GraphitiClient(
    base_url=_settings.graphiti_url,
    api_key=_settings.graphiti_api_key,
)


# --- Passthrough tools ---


@mcp.tool()
async def status() -> str:
    """Check if Graphiti is running and healthy."""
    return await passthrough.status(_client)


@mcp.tool()
async def add_raw_memory(
    content: str,
    group_id: str,
    name: str = "",
    source_description: str = "",
) -> str:
    """Add an episode directly to the knowledge graph.

    Args:
        content: The content to store.
        group_id: Target group (e.g. 'user_profile' or 'project_{id}').
        name: Label for the episode.
        source_description: Description of the content source.
    """
    return await passthrough.add_raw_memory(
        _client, content=content, group_id=group_id, name=name, source_description=source_description
    )


@mcp.tool()
async def search_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> str:
    """Search for relationships between entities in the knowledge graph.

    Args:
        query: Natural language search query.
        group_ids: Filter results to specific groups.
        max_facts: Maximum number of facts to return.
    """
    return await passthrough.search_facts(_client, query=query, group_ids=group_ids, max_facts=max_facts)


@mcp.tool()
async def search_entities(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> str:
    """Search for entities by semantic query.

    Args:
        query: Natural language search query.
        group_ids: Filter results to specific groups.
        max_facts: Maximum number of results.
    """
    return await passthrough.search_entities(_client, query=query, group_ids=group_ids, max_facts=max_facts)


@mcp.tool()
async def reset_memory(group_ids: list[str]) -> str:
    """Purge all data for the given group IDs. DANGEROUS - use with caution.

    Args:
        group_ids: List of group IDs to delete (e.g. ['user_profile', 'project_myapp']).
    """
    return await passthrough.reset_memory(_client, group_ids=group_ids)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit**

```bash
git add src/graph_mem/server.py src/graph_mem/tools/
git commit -m "feat: MCP server skeleton with passthrough tools"
```

---

## Task 5: save_memory and save_session tools

**Files:**
- Create: `src/graph_mem/tools/memory.py`
- Create: `tests/test_memory.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_memory.py`:

```python
import pytest

from graph_mem.client import GraphitiClient
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
    result = await save_memory(client, content="I prefer dark mode", group_id="user_profile")
    assert len(calls) == 1
    assert calls[0]["group_id"] == "user_profile"
    assert "dark mode" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_save_session(client, monkeypatch):
    calls = []

    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    result = await save_session(
        client,
        summary="Implemented the auth module and fixed 3 bugs",
        project_id="project_github.com/user/repo",
    )
    assert len(calls) == 1
    assert calls[0]["group_id"] == "project_github.com/user/repo"
    assert "auth module" in calls[0]["messages"][0]["content"]


@pytest.mark.asyncio
async def test_save_session_updates_user_profile(client, monkeypatch):
    calls = []

    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    result = await save_session(
        client,
        summary="Started learning Rust for the CLI rewrite",
        project_id="project_myapp",
    )
    # Session always goes to project group
    assert calls[0]["group_id"] == "project_myapp"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL (cannot import `graph_mem.tools.memory`)

- [ ] **Step 3: Implement memory tools**

`src/graph_mem/tools/memory.py`:

```python
"""Memory storage tools: save_memory, save_session."""

from graph_mem.client import GraphitiClient


async def save_memory(
    client: GraphitiClient,
    content: str,
    group_id: str,
) -> str:
    """Store a specific piece of information in the knowledge graph.

    The agent decides which group_id to use:
    - 'user_profile' for personal info (preferences, expertise, habits)
    - 'project_{id}' for project-specific info
    """
    await client.add_messages(
        group_id=group_id,
        messages=[
            {
                "content": content,
                "role_type": "user",
                "role": "developer",
                "name": "memory",
                "source_description": "Developer explicitly asked to remember this.",
            }
        ],
    )
    return f"Saved to {group_id}."


async def save_session(
    client: GraphitiClient,
    summary: str,
    project_id: str,
) -> str:
    """Send a session summary to Graphiti for entity/relation extraction.

    Stored in the project group. Graphiti will extract entities (decisions,
    technologies, blockers, etc.) and update the knowledge graph.
    """
    await client.add_messages(
        group_id=project_id,
        messages=[
            {
                "content": summary,
                "role_type": "system",
                "role": "graph-mem",
                "name": "session-summary",
                "source_description": "Automatic session summary from Claude Code.",
            }
        ],
    )
    return f"Session summary saved to {project_id}."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_memory.py -v`
Expected: 3 passed

- [ ] **Step 5: Register tools in server.py**

Add to `src/graph_mem/server.py` after the passthrough tools section:

```python
from graph_mem.tools import memory as memory_tools
from graph_mem.project_id import get_project_id

# --- Memory tools ---


@mcp.tool()
async def save_memory(content: str, group_id: str) -> str:
    """Store a specific piece of information. Use 'user_profile' for personal info, 'project_{id}' for project-specific.

    Args:
        content: The information to store.
        group_id: Target group ('user_profile' or 'project_{id}').
    """
    return await memory_tools.save_memory(_client, content=content, group_id=group_id)


@mcp.tool()
async def save_session(summary: str, project_path: str | None = None) -> str:
    """Send a session summary to the knowledge graph for entity extraction.

    Args:
        summary: Session summary text.
        project_path: Path to project root. Defaults to cwd.
    """
    project_id = get_project_id(project_path)
    return await memory_tools.save_session(_client, summary=summary, project_id=project_id)
```

- [ ] **Step 6: Commit**

```bash
git add src/graph_mem/tools/memory.py tests/test_memory.py src/graph_mem/server.py
git commit -m "feat: save_memory and save_session tools"
```

---

## Task 6: get_profile tool

**Files:**
- Create: `src/graph_mem/tools/profile.py`
- Create: `tests/test_profile.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_profile.py`:

```python
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
                {
                    "uuid": "1",
                    "name": "expertise",
                    "fact": "developer has 10 years of TypeScript experience",
                    "valid_at": None,
                    "invalid_at": None,
                    "created_at": "2026-04-05T10:00:00Z",
                    "expired_at": None,
                },
                {
                    "uuid": "2",
                    "name": "preference",
                    "fact": "developer prefers pnpm over npm",
                    "valid_at": None,
                    "invalid_at": None,
                    "created_at": "2026-04-05T10:00:00Z",
                    "expired_at": None,
                },
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_profile.py -v`
Expected: FAIL

- [ ] **Step 3: Implement get_profile**

`src/graph_mem/tools/profile.py`:

```python
"""Profile tool: retrieve developer profile from user_profile group."""

from graph_mem.client import GraphitiClient
from graph_mem.tools.passthrough import _format_facts

USER_PROFILE = "user_profile"


async def get_profile(client: GraphitiClient) -> str:
    """Retrieve the complete developer profile."""
    result = await client.search(
        query="developer profile preferences expertise habits projects principles",
        group_ids=[USER_PROFILE],
        max_facts=30,
    )
    facts = result.get("facts", [])
    if not facts:
        return "No profile data yet. The developer profile is empty."
    return f"Developer profile:\n{_format_facts(facts)}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_profile.py -v`
Expected: 2 passed

- [ ] **Step 5: Register in server.py**

Add to `src/graph_mem/server.py`:

```python
from graph_mem.tools import profile as profile_tools

# --- Profile tools ---


@mcp.tool()
async def get_profile() -> str:
    """Retrieve the complete developer profile (preferences, expertise, active projects, principles)."""
    return await profile_tools.get_profile(_client)
```

- [ ] **Step 6: Commit**

```bash
git add src/graph_mem/tools/profile.py tests/test_profile.py src/graph_mem/server.py
git commit -m "feat: get_profile tool"
```

---

## Task 7: add_reminder and get_reminders tools

**Files:**
- Create: `src/graph_mem/tools/reminders.py`
- Create: `tests/test_reminders.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_reminders.py`:

```python
import pytest

from graph_mem.client import GraphitiClient
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
    result = await add_reminder(client, content="Renew API key before April 15", group_id="user_profile")
    assert len(calls) == 1
    assert "Renew API key" in calls[0]["messages"][0]["content"]
    assert "REMINDER" in calls[0]["messages"][0]["content"] or "reminder" in calls[0]["messages"][0]["content"].lower()


@pytest.mark.asyncio
async def test_add_reminder_project(client, monkeypatch):
    calls = []

    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    result = await add_reminder(
        client,
        content="Update deps before release",
        group_id="project_myapp",
    )
    assert calls[0]["group_id"] == "project_myapp"


@pytest.mark.asyncio
async def test_get_reminders(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {
            "facts": [
                {
                    "uuid": "1",
                    "name": "reminder",
                    "fact": "Reminder: renew API key before April 15",
                    "valid_at": None,
                    "invalid_at": None,
                    "created_at": "2026-04-05T10:00:00Z",
                    "expired_at": None,
                }
            ]
        }

    monkeypatch.setattr(client, "search", mock_search)
    result = await get_reminders(client, group_ids=["user_profile"])
    assert "API key" in result


@pytest.mark.asyncio
async def test_get_reminders_all(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {"facts": []}

    monkeypatch.setattr(client, "search", mock_search)
    result = await get_reminders(client)
    assert "no" in result.lower() or "empty" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_reminders.py -v`
Expected: FAIL

- [ ] **Step 3: Implement reminders tools**

`src/graph_mem/tools/reminders.py`:

```python
"""Reminder tools: add_reminder, get_reminders."""

from graph_mem.client import GraphitiClient
from graph_mem.tools.passthrough import _format_facts


async def add_reminder(
    client: GraphitiClient,
    content: str,
    group_id: str,
) -> str:
    """Create a reminder entity in the knowledge graph."""
    await client.add_messages(
        group_id=group_id,
        messages=[
            {
                "content": f"REMINDER: {content}",
                "role_type": "system",
                "role": "graph-mem",
                "name": "reminder",
                "source_description": "Developer-created reminder.",
            }
        ],
    )
    return f"Reminder added to {group_id}."


async def get_reminders(
    client: GraphitiClient,
    group_ids: list[str] | None = None,
) -> str:
    """List active reminders."""
    result = await client.search(
        query="active reminders things to remember",
        group_ids=group_ids,
        max_facts=20,
    )
    facts = result.get("facts", [])
    if not facts:
        return "No active reminders."
    return f"Active reminders:\n{_format_facts(facts)}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_reminders.py -v`
Expected: 4 passed

- [ ] **Step 5: Register in server.py**

Add to `src/graph_mem/server.py`:

```python
from graph_mem.tools import reminders as reminder_tools

# --- Reminder tools ---


@mcp.tool()
async def add_reminder(content: str, group_id: str) -> str:
    """Create a reminder. Use 'user_profile' for personal, 'project_{id}' for project-specific.

    Args:
        content: What to remember.
        group_id: Target group.
    """
    return await reminder_tools.add_reminder(_client, content=content, group_id=group_id)


@mcp.tool()
async def get_reminders(group_ids: list[str] | None = None) -> str:
    """List active reminders.

    Args:
        group_ids: Filter by groups. If omitted, returns all reminders.
    """
    return await reminder_tools.get_reminders(_client, group_ids=group_ids)
```

- [ ] **Step 6: Commit**

```bash
git add src/graph_mem/tools/reminders.py tests/test_reminders.py src/graph_mem/server.py
git commit -m "feat: add_reminder and get_reminders tools"
```

---

## Task 8: check_project and onboard_project tools

**Files:**
- Create: `src/graph_mem/tools/onboard.py`
- Create: `tests/test_onboard.py`

- [ ] **Step 1: Write the failing tests**

`tests/test_onboard.py`:

```python
import os

import pytest

from graph_mem.client import GraphitiClient
from graph_mem.tools.onboard import check_project, onboard_project


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_check_project_known(client, monkeypatch):
    async def mock_search(query, group_ids=None, max_facts=10):
        return {
            "facts": [
                {
                    "uuid": "1",
                    "name": "project",
                    "fact": "graph-mem is a Claude Code memory plugin",
                    "valid_at": None,
                    "invalid_at": None,
                    "created_at": "2026-04-05T10:00:00Z",
                    "expired_at": None,
                }
            ]
        }

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
    # Create a minimal project
    readme = tmp_path / "README.md"
    readme.write_text("# My Project\nA test project for unit tests.")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "my-project"\ndependencies = ["fastapi"]')

    calls = []

    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    result = await onboard_project(
        client,
        project_id="project_my-project",
        project_path=str(tmp_path),
        description="A test project",
    )
    # Should have at least one call to the project group
    project_calls = [c for c in calls if c["group_id"] == "project_my-project"]
    assert len(project_calls) >= 1
    # Should also register in user_profile
    profile_calls = [c for c in calls if c["group_id"] == "user_profile"]
    assert len(profile_calls) >= 1


@pytest.mark.asyncio
async def test_onboard_project_no_readme(client, monkeypatch, tmp_path):
    calls = []

    async def mock_add_messages(group_id, messages):
        calls.append({"group_id": group_id, "messages": messages})
        return {"success": True, "message": "ok"}

    monkeypatch.setattr(client, "add_messages", mock_add_messages)
    result = await onboard_project(
        client,
        project_id="project_bare",
        project_path=str(tmp_path),
    )
    # Should still work even without README
    assert len(calls) >= 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_onboard.py -v`
Expected: FAIL

- [ ] **Step 3: Implement onboard tools**

`src/graph_mem/tools/onboard.py`:

```python
"""Onboarding tools: check_project, onboard_project."""

import os
from pathlib import Path
from typing import Any

from graph_mem.client import GraphitiClient

USER_PROFILE = "user_profile"

# Files to look for when analyzing a project
MANIFEST_FILES = [
    "package.json",
    "pyproject.toml",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "Gemfile",
    "composer.json",
]

MAX_FILE_READ = 2000  # characters


async def check_project(
    client: GraphitiClient,
    project_id: str,
) -> dict[str, Any]:
    """Check if a project is known in the knowledge graph."""
    result = await client.search(
        query="project description objectives stack team",
        group_ids=[project_id],
        max_facts=5,
    )
    facts = result.get("facts", [])
    return {"known": len(facts) > 0, "facts": facts}


async def onboard_project(
    client: GraphitiClient,
    project_id: str,
    project_path: str,
    description: str | None = None,
) -> str:
    """Analyze a project and store its essence in the knowledge graph."""
    parts = []

    # Developer's description takes priority
    if description:
        parts.append(f"Developer description: {description}")

    # Read README if present
    readme_path = _find_readme(project_path)
    if readme_path:
        content = _read_truncated(readme_path)
        parts.append(f"README:\n{content}")

    # Read manifest files
    for manifest in MANIFEST_FILES:
        manifest_path = os.path.join(project_path, manifest)
        if os.path.isfile(manifest_path):
            content = _read_truncated(manifest_path)
            parts.append(f"{manifest}:\n{content}")

    # Directory structure (top-level only)
    try:
        entries = sorted(os.listdir(project_path))
        dirs = [e for e in entries if os.path.isdir(os.path.join(project_path, e)) and not e.startswith(".")]
        files = [e for e in entries if os.path.isfile(os.path.join(project_path, e)) and not e.startswith(".")]
        parts.append(f"Top-level directories: {', '.join(dirs) if dirs else 'none'}")
        parts.append(f"Top-level files: {', '.join(files) if files else 'none'}")
    except OSError:
        pass

    project_name = project_id.removeprefix("project_").split("/")[-1]
    combined = "\n\n".join(parts) if parts else f"New project: {project_name} (no details available)"

    # Store detailed info in project group
    await client.add_messages(
        group_id=project_id,
        messages=[
            {
                "content": f"Project onboarding for {project_name}:\n\n{combined}",
                "role_type": "system",
                "role": "graph-mem",
                "name": "project-onboarding",
                "source_description": "Automatic project analysis during first encounter.",
            }
        ],
    )

    # Store lightweight reference in user_profile
    summary = description or f"Project {project_name}"
    await client.add_messages(
        group_id=USER_PROFILE,
        messages=[
            {
                "content": f"Developer works on project: {project_name}. {summary}",
                "role_type": "system",
                "role": "graph-mem",
                "name": "project-reference",
                "source_description": "Project registered in developer profile.",
            }
        ],
    )

    return f"Project '{project_name}' onboarded and registered in profile."


def _find_readme(project_path: str) -> str | None:
    """Find a README file (case-insensitive)."""
    try:
        for entry in os.listdir(project_path):
            if entry.lower().startswith("readme"):
                return os.path.join(project_path, entry)
    except OSError:
        pass
    return None


def _read_truncated(path: str) -> str:
    """Read a file, truncated to MAX_FILE_READ characters."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_FILE_READ)
            if len(content) == MAX_FILE_READ:
                content += "\n... (truncated)"
            return content
    except OSError:
        return "(could not read file)"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_onboard.py -v`
Expected: 4 passed

- [ ] **Step 5: Register in server.py**

Add to `src/graph_mem/server.py`:

```python
from graph_mem.tools import onboard as onboard_tools

# --- Onboarding tools ---


@mcp.tool()
async def check_project(project_path: str | None = None) -> str:
    """Check if the current project is known in the knowledge graph.

    Args:
        project_path: Path to project root. Defaults to cwd.
    """
    project_id = get_project_id(project_path)
    result = await onboard_tools.check_project(_client, project_id=project_id)
    if result["known"]:
        facts = "\n".join(f"- {f['fact']}" for f in result["facts"])
        return f"Project is known.\n{facts}"
    return "Project is not known. Use onboard_project to set it up."


@mcp.tool()
async def onboard_project(project_path: str | None = None, description: str | None = None) -> str:
    """Analyze a project and store its essence in the knowledge graph.

    Reads README, manifests, and directory structure. Stores results in both
    the project group and the developer profile.

    Args:
        project_path: Path to project root. Defaults to cwd.
        description: Developer's own description (supplements auto-analysis).
    """
    path = project_path or os.getcwd()
    project_id = get_project_id(path)
    return await onboard_tools.onboard_project(
        _client, project_id=project_id, project_path=path, description=description
    )
```

Add `import os` at the top of server.py if not already present.

- [ ] **Step 6: Commit**

```bash
git add src/graph_mem/tools/onboard.py tests/test_onboard.py src/graph_mem/server.py
git commit -m "feat: check_project and onboard_project tools"
```

---

## Task 9: get_context tool

**Files:**
- Create: `src/graph_mem/tools/context.py`
- Create: `tests/test_context.py`

This is the main tool that merges user_profile + project context + reminders into a single context injection.

- [ ] **Step 1: Write the failing tests**

`tests/test_context.py`:

```python
import pytest

from graph_mem.client import GraphitiClient
from graph_mem.tools.context import get_context


@pytest.fixture
def client():
    return GraphitiClient(base_url="http://test:8000")


@pytest.mark.asyncio
async def test_get_context(client, monkeypatch):
    search_calls = []

    async def mock_search(query, group_ids=None, max_facts=10):
        search_calls.append({"query": query, "group_ids": group_ids})
        if group_ids == ["user_profile"]:
            if "reminder" in query.lower():
                return {
                    "facts": [
                        {
                            "uuid": "r1",
                            "name": "reminder",
                            "fact": "Reminder: renew API key before April 15",
                            "valid_at": None,
                            "invalid_at": None,
                            "created_at": "2026-04-05T10:00:00Z",
                            "expired_at": None,
                        }
                    ]
                }
            return {
                "facts": [
                    {
                        "uuid": "1",
                        "name": "expertise",
                        "fact": "developer has 10 years TypeScript",
                        "valid_at": None,
                        "invalid_at": None,
                        "created_at": "2026-04-05T10:00:00Z",
                        "expired_at": None,
                    }
                ]
            }
        if "project_" in (group_ids or [""])[0]:
            if "reminder" in query.lower():
                return {"facts": []}
            return {
                "facts": [
                    {
                        "uuid": "2",
                        "name": "stack",
                        "fact": "project uses Python and FastMCP",
                        "valid_at": None,
                        "invalid_at": None,
                        "created_at": "2026-04-05T10:00:00Z",
                        "expired_at": None,
                    }
                ]
            }
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_context.py -v`
Expected: FAIL

- [ ] **Step 3: Implement get_context**

`src/graph_mem/tools/context.py`:

```python
"""Context tool: merge user_profile + project context + reminders."""

from graph_mem.client import GraphitiClient
from graph_mem.tools.passthrough import _format_facts

USER_PROFILE = "user_profile"


async def get_context(
    client: GraphitiClient,
    project_id: str,
) -> str:
    """Retrieve merged context for the current session.

    Queries:
    1. user_profile for developer preferences, expertise, principles
    2. project_{id} for project-specific context
    3. Both scopes for active reminders
    """
    sections = []

    # 1. Developer profile
    profile_result = await client.search(
        query="developer profile preferences expertise habits principles active projects",
        group_ids=[USER_PROFILE],
        max_facts=20,
    )
    profile_facts = profile_result.get("facts", [])
    if profile_facts:
        sections.append(f"## Developer Profile\n{_format_facts(profile_facts)}")

    # 2. Project context
    project_result = await client.search(
        query="project stack team decisions goals blockers recent sessions context",
        group_ids=[project_id],
        max_facts=20,
    )
    project_facts = project_result.get("facts", [])
    if project_facts:
        sections.append(f"## Project Context\n{_format_facts(project_facts)}")

    # 3. Reminders (both scopes)
    reminder_facts = []
    for gid in [USER_PROFILE, project_id]:
        reminder_result = await client.search(
            query="active reminders things to remember",
            group_ids=[gid],
            max_facts=10,
        )
        reminder_facts.extend(reminder_result.get("facts", []))

    if reminder_facts:
        sections.append(f"## Reminders\n{_format_facts(reminder_facts)}")

    if not sections:
        return "No context available yet for this session."

    return "\n\n".join(sections)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_context.py -v`
Expected: 2 passed

- [ ] **Step 5: Register in server.py**

Add to `src/graph_mem/server.py`:

```python
from graph_mem.tools import context as context_tools

# --- Context tools ---


@mcp.tool()
async def get_context(project_path: str | None = None) -> str:
    """Retrieve the full context for the current session.

    Merges developer profile + project context + active reminders.

    Args:
        project_path: Path to project root. Defaults to cwd.
    """
    project_id = get_project_id(project_path)
    return await context_tools.get_context(_client, project_id=project_id)
```

- [ ] **Step 6: Commit**

```bash
git add src/graph_mem/tools/context.py tests/test_context.py src/graph_mem/server.py
git commit -m "feat: get_context tool with merged profile + project + reminders"
```

---

## Task 10: Claude Code hooks

**Files:**
- Create: `src/graph_mem/hooks/__init__.py`
- Create: `src/graph_mem/hooks/session_start.py`
- Create: `src/graph_mem/hooks/session_end.py`

Hooks are standalone Python scripts that Claude Code runs at session lifecycle events. They call our MCP tools via the GraphitiClient directly (not through MCP protocol).

- [ ] **Step 1: Create hooks package**

`src/graph_mem/hooks/__init__.py`:

```python
"""Claude Code lifecycle hooks for graph-mem."""
```

- [ ] **Step 2: Implement SessionStart hook**

`src/graph_mem/hooks/session_start.py`:

```python
"""SessionStart hook: inject context at the beginning of a Claude Code session.

This script is called by Claude Code's hook system. It outputs text that gets
injected into the agent's context via hookSpecificOutput.additionalContext.

Flow:
1. Detect project from cwd
2. check_project() - is this project known?
3. get_context() - query user_profile + project_{id}
4. get_reminders() - inject active reminders
"""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.context import get_context
from graph_mem.tools.onboard import check_project
from graph_mem.tools.reminders import get_reminders


async def run() -> str:
    settings = get_settings()
    client = GraphitiClient(base_url=settings.graphiti_url, api_key=settings.graphiti_api_key)

    project_id = get_project_id()
    parts = []

    # Check if project is known
    project_status = await check_project(client, project_id=project_id)
    if not project_status["known"]:
        parts.append(
            "New project detected. Use `onboard_project` to set it up in the knowledge graph."
        )

    # Get full context
    context = await get_context(client, project_id=project_id)
    if context and "no context available" not in context.lower():
        parts.append(context)

    if not parts:
        parts.append("graph-mem: No context available yet. Start by onboarding this project.")

    return "\n\n".join(parts)


def main():
    try:
        output = asyncio.run(run())
        print(output)
    except Exception as e:
        print(f"graph-mem SessionStart hook error: {e}", file=sys.stderr)
        sys.exit(0)  # Don't block the session on hook failure


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Implement SessionEnd hook**

`src/graph_mem/hooks/session_end.py`:

```python
"""SessionEnd (Stop) hook: save session summary when a Claude Code session ends.

This script is called by Claude Code's hook system. The session transcript
is available via stdin. The hook generates a summary prompt and saves it.

Note: The actual LLM summary generation is handled by Claude Code's hook system.
This hook receives the summary as input and stores it in Graphiti.
"""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.memory import save_session


async def run(summary: str) -> None:
    if not summary.strip():
        return

    settings = get_settings()
    client = GraphitiClient(base_url=settings.graphiti_url, api_key=settings.graphiti_api_key)
    project_id = get_project_id()

    await save_session(client, summary=summary, project_id=project_id)


def main():
    summary = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not summary.strip():
        return
    try:
        asyncio.run(run(summary))
    except Exception as e:
        print(f"graph-mem SessionEnd hook error: {e}", file=sys.stderr)
        sys.exit(0)  # Don't block on hook failure


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add hook entry points to pyproject.toml**

Add these to the `[project.scripts]` section in `pyproject.toml`:

```toml
[project.scripts]
graph-mem = "graph_mem.server:main"
graph-mem-session-start = "graph_mem.hooks.session_start:main"
graph-mem-session-end = "graph_mem.hooks.session_end:main"
```

- [ ] **Step 5: Commit**

```bash
git add src/graph_mem/hooks/ pyproject.toml
git commit -m "feat: Claude Code SessionStart and SessionEnd hooks"
```

---

## Task 11: docker-compose.yml for local Graphiti

**Files:**
- Create: `docker-compose.yml`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
services:
  graphiti:
    image: zepai/graphiti:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphiti}
    depends_on:
      neo4j:
        condition: service_healthy

  neo4j:
    image: neo4j:5.22.0
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-graphiti}
    healthcheck:
      test: ["CMD-SHELL", "neo4j status || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  neo4j_data:
```

- [ ] **Step 2: Add .env.example**

Create `.env.example`:

```
OPENAI_API_KEY=sk-...
NEO4J_PASSWORD=graphiti
```

- [ ] **Step 3: Update .gitignore**

Add `.env` if not already present (it is).

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml .env.example
git commit -m "feat: docker-compose for local Graphiti + Neo4j"
```

---

## Task 12: SKILL.md for CLI agents

**Files:**
- Create: `skills/graph-mem/SKILL.md`

- [ ] **Step 1: Create skill directory and file**

`skills/graph-mem/SKILL.md`:

````markdown
---
name: graph-mem
description: Persistent developer memory using knowledge graphs. Guides when and how to store and retrieve memories.
---

# graph-mem Memory Skill

You have access to a persistent knowledge graph that remembers information about the developer and their projects across sessions. Context is automatically injected at session start.

## When to Store Memory

**STORE (use `save_memory`):**
- Developer expresses a preference ("I prefer X", "I don't like Y")
- Developer shares personal info (expertise, role, habits)
- Developer briefs a project (stack, team, objectives, problems)
- Developer makes a technical decision
- Developer says "remember that...", "remind me..."

**DO NOT STORE:**
- Source code or file contents
- Secrets, tokens, passwords
- Ephemeral commands ("run npm install")
- Info derivable from code (file structure, imports, etc.)

## How to Choose group_id

- Info about the developer (preference, expertise, habit, personal) -> `user_profile`
- Info about a specific project -> `project_{id}` (derived from cwd, provided by tools)
- When in doubt -> `user_profile`

## Available Tools

| Tool | Purpose |
|---|---|
| `get_context` | Get merged developer profile + project context + reminders |
| `save_memory` | Store a specific piece of information |
| `save_session` | Save a session summary (usually automatic) |
| `get_profile` | Get the developer's profile |
| `check_project` | Check if a project is known |
| `onboard_project` | Analyze and register a new project |
| `add_reminder` | Create a reminder |
| `get_reminders` | List active reminders |
| `search_facts` | Search for relationships in the graph |
| `search_entities` | Search for entities by query |
| `add_raw_memory` | Low-level: add an episode directly |
| `reset_memory` | DANGEROUS: purge all data for given groups |
| `status` | Check if Graphiti is running |

## How to Use Injected Context

At session start, context is injected containing:
- **Developer profile**: Adapt tone and technicality level
- **Project context**: Know what was done, decisions made, current state
- **Reminders**: Mention them naturally when relevant
- **Principles**: Respect them in your suggestions (e.g., if "no mocks in integration tests", don't suggest mocks)

## Reminders

When the developer says "remind me to..." or "don't forget to...":
1. Use `add_reminder` with the appropriate group_id
2. Reminders surface automatically at session start via `get_reminders`
````

- [ ] **Step 2: Commit**

```bash
git add skills/
git commit -m "feat: SKILL.md for guiding CLI agents"
```

---

## Task 13: Full server assembly and smoke test

**Files:**
- Modify: `src/graph_mem/server.py` (final assembly)

This task brings together all the imports in server.py and verifies the full MCP server starts correctly.

- [ ] **Step 1: Write the final server.py**

Assemble all tool registrations into the final `src/graph_mem/server.py`. The complete file:

```python
"""graph-mem MCP server."""

import os

from mcp.server.fastmcp import FastMCP

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools import context as context_tools
from graph_mem.tools import memory as memory_tools
from graph_mem.tools import onboard as onboard_tools
from graph_mem.tools import passthrough
from graph_mem.tools import profile as profile_tools
from graph_mem.tools import reminders as reminder_tools

mcp = FastMCP("graph-mem")

_settings = get_settings()
_client = GraphitiClient(
    base_url=_settings.graphiti_url,
    api_key=_settings.graphiti_api_key,
)


# --- Passthrough tools ---


@mcp.tool()
async def status() -> str:
    """Check if Graphiti is running and healthy."""
    return await passthrough.status(_client)


@mcp.tool()
async def add_raw_memory(
    content: str,
    group_id: str,
    name: str = "",
    source_description: str = "",
) -> str:
    """Add an episode directly to the knowledge graph.

    Args:
        content: The content to store.
        group_id: Target group (e.g. 'user_profile' or 'project_{id}').
        name: Label for the episode.
        source_description: Description of the content source.
    """
    return await passthrough.add_raw_memory(
        _client, content=content, group_id=group_id, name=name, source_description=source_description
    )


@mcp.tool()
async def search_facts(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> str:
    """Search for relationships between entities in the knowledge graph.

    Args:
        query: Natural language search query.
        group_ids: Filter results to specific groups.
        max_facts: Maximum number of facts to return.
    """
    return await passthrough.search_facts(_client, query=query, group_ids=group_ids, max_facts=max_facts)


@mcp.tool()
async def search_entities(
    query: str,
    group_ids: list[str] | None = None,
    max_facts: int = 10,
) -> str:
    """Search for entities by semantic query.

    Args:
        query: Natural language search query.
        group_ids: Filter results to specific groups.
        max_facts: Maximum number of results.
    """
    return await passthrough.search_entities(_client, query=query, group_ids=group_ids, max_facts=max_facts)


@mcp.tool()
async def reset_memory(group_ids: list[str]) -> str:
    """Purge all data for the given group IDs. DANGEROUS - use with caution.

    Args:
        group_ids: List of group IDs to delete (e.g. ['user_profile', 'project_myapp']).
    """
    return await passthrough.reset_memory(_client, group_ids=group_ids)


# --- Memory tools ---


@mcp.tool()
async def save_memory(content: str, group_id: str) -> str:
    """Store a specific piece of information. Use 'user_profile' for personal info, 'project_{id}' for project-specific.

    Args:
        content: The information to store.
        group_id: Target group ('user_profile' or 'project_{id}').
    """
    return await memory_tools.save_memory(_client, content=content, group_id=group_id)


@mcp.tool()
async def save_session(summary: str, project_path: str | None = None) -> str:
    """Send a session summary to the knowledge graph for entity extraction.

    Args:
        summary: Session summary text.
        project_path: Path to project root. Defaults to cwd.
    """
    project_id = get_project_id(project_path)
    return await memory_tools.save_session(_client, summary=summary, project_id=project_id)


# --- Profile tools ---


@mcp.tool()
async def get_profile() -> str:
    """Retrieve the complete developer profile (preferences, expertise, active projects, principles)."""
    return await profile_tools.get_profile(_client)


# --- Reminder tools ---


@mcp.tool()
async def add_reminder(content: str, group_id: str) -> str:
    """Create a reminder. Use 'user_profile' for personal, 'project_{id}' for project-specific.

    Args:
        content: What to remember.
        group_id: Target group.
    """
    return await reminder_tools.add_reminder(_client, content=content, group_id=group_id)


@mcp.tool()
async def get_reminders(group_ids: list[str] | None = None) -> str:
    """List active reminders.

    Args:
        group_ids: Filter by groups. If omitted, returns all reminders.
    """
    return await reminder_tools.get_reminders(_client, group_ids=group_ids)


# --- Onboarding tools ---


@mcp.tool()
async def check_project(project_path: str | None = None) -> str:
    """Check if the current project is known in the knowledge graph.

    Args:
        project_path: Path to project root. Defaults to cwd.
    """
    project_id = get_project_id(project_path)
    result = await onboard_tools.check_project(_client, project_id=project_id)
    if result["known"]:
        facts = "\n".join(f"- {f['fact']}" for f in result["facts"])
        return f"Project is known.\n{facts}"
    return "Project is not known. Use onboard_project to set it up."


@mcp.tool()
async def onboard_project(project_path: str | None = None, description: str | None = None) -> str:
    """Analyze a project and store its essence in the knowledge graph.

    Reads README, manifests, and directory structure. Stores results in both
    the project group and the developer profile.

    Args:
        project_path: Path to project root. Defaults to cwd.
        description: Developer's own description (supplements auto-analysis).
    """
    path = project_path or os.getcwd()
    project_id = get_project_id(path)
    return await onboard_tools.onboard_project(
        _client, project_id=project_id, project_path=path, description=description
    )


# --- Context tools ---


@mcp.tool()
async def get_context(project_path: str | None = None) -> str:
    """Retrieve the full context for the current session.

    Merges developer profile + project context + active reminders.

    Args:
        project_path: Path to project root. Defaults to cwd.
    """
    project_id = get_project_id(project_path)
    return await context_tools.get_context(_client, project_id=project_id)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 3: Verify server starts**

Run: `echo '{}' | timeout 3 python -m graph_mem.server 2>&1 || true`

This should start without import errors. It will exit because stdin closes, which is expected.

- [ ] **Step 4: Commit**

```bash
git add src/graph_mem/server.py
git commit -m "feat: complete MCP server assembly with all 13 tools"
```

---

## Summary

| Task | What it delivers | Tools added |
|---|---|---|
| 1 | Project scaffolding, config | - |
| 2 | GraphitiClient HTTP wrapper | - |
| 3 | Project identifier utility | - |
| 4 | MCP server skeleton + passthrough | status, add_raw_memory, search_facts, search_entities, reset_memory |
| 5 | Memory storage | save_memory, save_session |
| 6 | Profile retrieval | get_profile |
| 7 | Reminders | add_reminder, get_reminders |
| 8 | Project onboarding | check_project, onboard_project |
| 9 | Context injection | get_context |
| 10 | Claude Code hooks | SessionStart, SessionEnd |
| 11 | Docker compose | Local Graphiti + Neo4j |
| 12 | Skill file | SKILL.md for CLI agents |
| 13 | Final assembly + smoke test | Full server verification |

Total: 13 MCP tools (8 custom + 5 passthrough), 2 hooks, 1 skill.
