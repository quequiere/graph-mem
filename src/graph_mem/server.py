"""graph-mem MCP server.

Minimal tool surface: save, search, and a workflow reminder.
Other operations (onboard, context, reminders, etc.) are used internally
by hooks but not exposed as MCP tools to keep the tool list small and
increase the chance that the model actually uses them.
"""

import os

from mcp.server.fastmcp import FastMCP

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools import memory as memory_tools
from graph_mem.tools import passthrough

mcp = FastMCP("graph-mem")


def _error(action: str, e: Exception) -> str:
    return f"graph-mem error ({action}): {e}"


_settings = get_settings()
_client = GraphitiClient(
    base_url=_settings.graphiti_url,
    api_key=_settings.graphiti_api_key,
)


# --- Workflow reminder (always visible in tool list) ---


@mcp.tool(name="__IMPORTANT__graph_mem")
async def important() -> str:
    """READ THIS FIRST: graph-mem is your persistent memory system.

    Call this tool if you are unsure how to use graph-mem.
    """
    return """graph-mem — Persistent Knowledge Graph Memory
=================================================

You have a knowledge graph that persists across sessions.

WHEN TO SAVE (use save_memory):
- User shares preferences, personal info, expertise, role
- User says "remember this", "don't forget", "note that"
- Important project decisions or context worth keeping
- group_id: "user_profile" for personal, "project_{id}" for project-specific

WHEN TO SEARCH (use search_memory):
- User asks "what do you know about me?", "do you remember?"
- You need context about the user or project
- Before making assumptions — check memory first

DO NOT STORE: source code, secrets, ephemeral commands, info derivable from code."""


# --- Core tools ---


@mcp.tool()
async def save_memory(content: str, group_id: str = "user_profile") -> str:
    """Save information to persistent memory.

    Use this when the user shares preferences, personal info, project decisions,
    or explicitly asks you to remember something.

    Args:
        content: The information to store.
        group_id: 'user_profile' for personal info, 'project_{id}' for project-specific.
    """
    try:
        return await memory_tools.save_memory(_client, content=content, group_id=group_id)
    except Exception as e:
        return _error("save_memory", e)


@mcp.tool()
async def search_memory(query: str, group_ids: list[str] | None = None) -> str:
    """Search persistent memory for facts about the user or project.

    Use this when the user asks what you remember, or when you need context.

    Args:
        query: Natural language search query.
        group_ids: Filter to specific groups (e.g. ['user_profile']). Omit for all.
    """
    try:
        # Search both facts and entities, merge results
        facts_result = await passthrough.search_facts(
            _client, query=query, group_ids=group_ids, max_facts=10
        )
        entities_result = await passthrough.search_entities(
            _client, query=query, group_ids=group_ids, max_facts=5
        )

        parts = []
        if facts_result and "no facts found" not in facts_result.lower():
            parts.append(f"Facts:\n{facts_result}")
        if entities_result and "no entities found" not in entities_result.lower():
            parts.append(f"Entities:\n{entities_result}")

        if not parts:
            return "No memories found for this query."
        return "\n\n".join(parts)
    except Exception as e:
        return _error("search_memory", e)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
