"""End-to-end integration tests for graph-mem MCP tools.

These tests launch the full stack (Ollama + Neo4j + Graphiti) via Docker
and communicate with graph-mem through a real MCP stdio session.
"""

import asyncio

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# Generous timeout for Ollama-backed operations (qwen2.5:7b can take up to 3min)
TOOL_TIMEOUT = 300

# Max time to poll for entity extraction results
EXTRACTION_TIMEOUT = 300

# Poll interval when waiting for extraction
POLL_INTERVAL = 10


def _text(result) -> str:
    """Extract text from a CallToolResult."""
    assert result.content, "Empty tool result"
    return result.content[0].text


async def _poll_search(mcp_session, tool: str, args: dict, expect: list[str], timeout: float = EXTRACTION_TIMEOUT) -> str:
    """Poll a search tool until expected keywords appear or timeout.

    Returns the last result text (pass or fail). Caller asserts on keywords.
    """
    deadline = asyncio.get_event_loop().time() + timeout
    last_text = ""
    while asyncio.get_event_loop().time() < deadline:
        result = await asyncio.wait_for(
            mcp_session.call_tool(tool, arguments=args),
            timeout=TOOL_TIMEOUT,
        )
        if result.isError:
            await asyncio.sleep(POLL_INTERVAL)
            continue
        text = _text(result)
        last_text = text
        if any(kw in text.lower() for kw in expect):
            return text
        await asyncio.sleep(POLL_INTERVAL)
    return last_text


async def test_status(mcp_session):
    """Graphiti responds healthy through MCP."""
    result = await asyncio.wait_for(
        mcp_session.call_tool("status", arguments={}),
        timeout=TOOL_TIMEOUT,
    )
    assert not result.isError
    text = _text(result)
    assert "healthy" in text.lower() or "status" in text.lower()


async def test_save_and_search_facts(mcp_session):
    """Save a memory and retrieve it via semantic search."""
    save_result = await asyncio.wait_for(
        mcp_session.call_tool(
            "save_memory",
            arguments={
                "content": "The developer prefers using Vim keybindings in all editors",
                "group_id": "user_profile",
            },
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not save_result.isError
    assert "saved" in _text(save_result).lower()

    text = await _poll_search(
        mcp_session,
        "search_facts",
        {"query": "editor keybindings preferences", "group_ids": ["user_profile"]},
        ["vim", "keybinding", "editor"],
    )
    assert "vim" in text.lower() or "keybinding" in text.lower() or "editor" in text.lower(), \
        f"Expected vim/keybinding/editor, got: {text[:200]}"


async def test_add_raw_and_search_entities(mcp_session):
    """Add raw memory and search for extracted entities."""
    add_result = await asyncio.wait_for(
        mcp_session.call_tool(
            "add_raw_memory",
            arguments={
                "content": "Alice is the tech lead of the backend team. She specializes in distributed systems.",
                "group_id": "project_test-project",
                "name": "team-info",
                "source_description": "integration test",
            },
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not add_result.isError

    text = await _poll_search(
        mcp_session,
        "search_entities",
        {"query": "Alice tech lead backend", "group_ids": ["project_test-project"]},
        ["alice", "tech lead", "backend"],
    )
    assert "alice" in text.lower() or "tech lead" in text.lower() or "backend" in text.lower(), \
        f"Expected alice/tech lead/backend, got: {text[:200]}"


async def test_save_session(mcp_session, tmp_path):
    """Session summary is ingested without error."""
    result = await asyncio.wait_for(
        mcp_session.call_tool(
            "save_session",
            arguments={
                "summary": "Worked on adding integration tests with pytest-docker. "
                "Set up Ollama as LLM backend. Resolved Docker networking issues.",
                "project_path": str(tmp_path),
            },
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not result.isError
    assert "saved" in _text(result).lower()


async def test_profile_empty_then_populated(mcp_session):
    """Profile starts empty, then returns data after saving memory."""
    # Empty profile
    empty_result = await asyncio.wait_for(
        mcp_session.call_tool("get_profile", arguments={}),
        timeout=TOOL_TIMEOUT,
    )
    assert not empty_result.isError
    empty_text = _text(empty_result)
    assert "no profile" in empty_text.lower() or "empty" in empty_text.lower()

    # Add profile data
    await asyncio.wait_for(
        mcp_session.call_tool(
            "save_memory",
            arguments={
                "content": "The developer is a senior Python engineer who loves type hints",
                "group_id": "user_profile",
            },
        ),
        timeout=TOOL_TIMEOUT,
    )

    # Poll until profile has data
    deadline = asyncio.get_event_loop().time() + EXTRACTION_TIMEOUT
    last_text = ""
    while asyncio.get_event_loop().time() < deadline:
        populated_result = await asyncio.wait_for(
            mcp_session.call_tool("get_profile", arguments={}),
            timeout=TOOL_TIMEOUT,
        )
        if not populated_result.isError:
            text = _text(populated_result)
            last_text = text
            if "python" in text.lower() or "type hint" in text.lower() or "senior" in text.lower():
                break
        await asyncio.sleep(POLL_INTERVAL)

    assert "python" in last_text.lower() or "type hint" in last_text.lower() or "senior" in last_text.lower(), \
        f"Expected python/type hint/senior, got: {last_text[:200]}"


async def test_reminders(mcp_session):
    """Add a reminder and retrieve it."""
    add_result = await asyncio.wait_for(
        mcp_session.call_tool(
            "add_reminder",
            arguments={
                "content": "Review the pull request for authentication module",
                "group_id": "user_profile",
            },
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not add_result.isError
    assert "reminder" in _text(add_result).lower()

    text = await _poll_search(
        mcp_session,
        "get_reminders",
        {"group_ids": ["user_profile"]},
        ["pull request", "authentication", "review"],
    )
    assert "pull request" in text.lower() or "authentication" in text.lower() or "review" in text.lower(), \
        f"Expected pull request/authentication/review, got: {text[:200]}"


async def test_onboard_and_check(mcp_session, tmp_path):
    """Onboard a project, then check it's recognized."""
    readme = tmp_path / "README.md"
    readme.write_text("# Test Project\nA test project for integration tests.")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "test-project"\nversion = "0.1.0"')

    onboard_result = await asyncio.wait_for(
        mcp_session.call_tool(
            "onboard_project",
            arguments={
                "project_path": str(tmp_path),
                "description": "A test project used for integration testing",
            },
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not onboard_result.isError
    assert "onboarded" in _text(onboard_result).lower()

    # Poll until project is known
    deadline = asyncio.get_event_loop().time() + EXTRACTION_TIMEOUT
    last_text = ""
    while asyncio.get_event_loop().time() < deadline:
        check_result = await asyncio.wait_for(
            mcp_session.call_tool(
                "check_project",
                arguments={"project_path": str(tmp_path)},
            ),
            timeout=TOOL_TIMEOUT,
        )
        if not check_result.isError:
            text = _text(check_result)
            last_text = text
            if "known" in text.lower() or "project" in text.lower():
                break
        await asyncio.sleep(POLL_INTERVAL)

    assert "known" in last_text.lower() or "project" in last_text.lower(), \
        f"Expected known/project, got: {last_text[:200]}"


async def test_get_context(mcp_session, tmp_path):
    """get_context returns without error and produces expected structure."""
    project_dir = str(tmp_path)

    # With a clean graph, get_context should return gracefully
    result = await asyncio.wait_for(
        mcp_session.call_tool(
            "get_context",
            arguments={"project_path": project_dir},
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not result.isError
    text = _text(result)
    # Empty graph → no context yet (valid response)
    assert "no context" in text.lower() or "profile" in text.lower() or "project" in text.lower()


async def test_reset_memory(mcp_session):
    """After reset, previously saved data is gone."""
    await asyncio.wait_for(
        mcp_session.call_tool(
            "save_memory",
            arguments={
                "content": "Developer favorite color is blue for syntax highlighting",
                "group_id": "user_profile",
            },
        ),
        timeout=TOOL_TIMEOUT,
    )

    # Wait for extraction before resetting
    await asyncio.sleep(30)

    reset_result = await asyncio.wait_for(
        mcp_session.call_tool(
            "reset_memory",
            arguments={"group_ids": ["user_profile"]},
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not reset_result.isError
    assert "deleted" in _text(reset_result).lower()

    # Search should find nothing
    search_result = await asyncio.wait_for(
        mcp_session.call_tool(
            "search_facts",
            arguments={
                "query": "favorite color syntax highlighting",
                "group_ids": ["user_profile"],
            },
        ),
        timeout=TOOL_TIMEOUT,
    )
    assert not search_result.isError
    text = _text(search_result)
    assert "no results" in text.lower() or "blue" not in text.lower()
