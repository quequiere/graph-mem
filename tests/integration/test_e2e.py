"""End-to-end integration tests for graph-mem.

Exercises the minimal MCP tool surface (save_memory / search_memory /
__IMPORTANT__graph_mem) against the full stack: Neo4j + patched Graphiti
+ Ollama with the recommended local models (gemma3:4b + qwen3-embedding:4b).

Extraction is slow with local models — we save a memory then poll
search_memory until entities appear in the knowledge graph.
"""

import asyncio

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.asyncio]

# Generous timeouts — gemma3:4b extraction can take minutes per message.
TOOL_TIMEOUT = 300
EXTRACTION_TIMEOUT = 600
POLL_INTERVAL = 10


def _text(result) -> str:
    """Extract text from a CallToolResult."""
    assert result.content, "Empty tool result"
    return result.content[0].text


async def _call(mcp_session, tool: str, args: dict | None = None) -> str:
    """Call a tool, assert no error, return text."""
    result = await asyncio.wait_for(
        mcp_session.call_tool(tool, arguments=args or {}),
        timeout=TOOL_TIMEOUT,
    )
    assert not result.isError, f"{tool} returned error: {result}"
    return _text(result)


async def _poll_search(
    mcp_session,
    query: str,
    scope: str,
    expected_keywords: list[str],
    timeout: float = EXTRACTION_TIMEOUT,
) -> str:
    """Poll search_memory until any keyword appears or timeout."""
    deadline = asyncio.get_event_loop().time() + timeout
    last_text = ""
    while asyncio.get_event_loop().time() < deadline:
        result = await asyncio.wait_for(
            mcp_session.call_tool(
                "search_memory",
                arguments={"query": query, "scope": scope},
            ),
            timeout=TOOL_TIMEOUT,
        )
        if not result.isError:
            last_text = _text(result)
            if any(kw.lower() in last_text.lower() for kw in expected_keywords):
                return last_text
        await asyncio.sleep(POLL_INTERVAL)
    return last_text


async def test_important_tool_returns_workflow(mcp_session):
    """__IMPORTANT__graph_mem is a pure string tool — fastest smoke test."""
    text = await _call(mcp_session, "__IMPORTANT__graph_mem")
    assert "graph-mem" in text.lower()
    assert "persistent" in text.lower() or "memory" in text.lower()


async def test_save_user_memory_and_search(mcp_session):
    """Save a user fact and retrieve it via user-scoped search."""
    save_text = await _call(
        mcp_session,
        "save_memory",
        {
            "content": "The developer prefers Vim keybindings in all editors.",
            "scope": "user",
        },
    )
    assert "saved" in save_text.lower()

    text = await _poll_search(
        mcp_session,
        query="editor keybindings preferences",
        scope="user",
        expected_keywords=["vim", "keybinding", "editor"],
    )
    assert any(kw in text.lower() for kw in ("vim", "keybinding", "editor")), (
        f"Expected vim/keybinding/editor in recall, got: {text[:300]}"
    )


async def test_save_project_memory_and_search(mcp_session):
    """Save a project fact and retrieve it via project-scoped search."""
    save_text = await _call(
        mcp_session,
        "save_memory",
        {
            "content": "Alice is the tech lead of the backend team and owns the payments service.",
            "scope": "project",
        },
    )
    assert "saved" in save_text.lower()

    text = await _poll_search(
        mcp_session,
        query="tech lead backend payments",
        scope="project",
        expected_keywords=["alice", "tech lead", "backend", "payments"],
    )
    assert any(
        kw in text.lower() for kw in ("alice", "tech lead", "backend", "payments")
    ), f"Expected alice/tech lead/backend/payments, got: {text[:300]}"


# Note: a "scope=all merges both groups" test was intentionally removed.
# It required saving two facts (user + project) in a single test, which is
# intrinsically flaky under local-model extraction: two sequential entity-
# extraction runs (~1-2 min each with gemma3:4b) race with the autouse
# _clean fixture that runs between tests. The scope=all routing itself is
# 3 lines in server.py (group_ids = [USER_PROFILE, _project_id]) and is
# more reliably exercised by unit tests than by a doubly-async e2e test.
