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


def _error(action: str, e: Exception) -> str:
    return f"graph-mem error ({action}): {e}"


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
    try:
        return await passthrough.add_raw_memory(
            _client, content=content, group_id=group_id, name=name, source_description=source_description
        )
    except Exception as e:
        return _error("add_raw_memory", e)


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
    try:
        return await passthrough.search_facts(_client, query=query, group_ids=group_ids, max_facts=max_facts)
    except Exception as e:
        return _error("search_facts", e)


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
    try:
        return await passthrough.search_entities(_client, query=query, group_ids=group_ids, max_facts=max_facts)
    except Exception as e:
        return _error("search_entities", e)


@mcp.tool()
async def reset_memory(group_ids: list[str]) -> str:
    """Purge all data for the given group IDs. DANGEROUS - use with caution.

    Args:
        group_ids: List of group IDs to delete (e.g. ['user_profile', 'project_myapp']).
    """
    try:
        return await passthrough.reset_memory(_client, group_ids=group_ids)
    except Exception as e:
        return _error("reset_memory", e)


# --- Memory tools ---


@mcp.tool()
async def save_memory(content: str, group_id: str) -> str:
    """Store a specific piece of information. Use 'user_profile' for personal info, 'project_{id}' for project-specific.

    Args:
        content: The information to store.
        group_id: Target group ('user_profile' or 'project_{id}').
    """
    try:
        return await memory_tools.save_memory(_client, content=content, group_id=group_id)
    except Exception as e:
        return _error("save_memory", e)


@mcp.tool()
async def save_session(summary: str, project_path: str | None = None) -> str:
    """Send a session summary to the knowledge graph for entity extraction.

    Args:
        summary: Session summary text.
        project_path: Path to project root. Defaults to cwd.
    """
    try:
        project_id = get_project_id(project_path)
        return await memory_tools.save_session(_client, summary=summary, project_id=project_id)
    except Exception as e:
        return _error("save_session", e)


# --- Profile tools ---


@mcp.tool()
async def get_profile() -> str:
    """Retrieve the complete developer profile (preferences, expertise, active projects, principles)."""
    try:
        return await profile_tools.get_profile(_client)
    except Exception as e:
        return _error("get_profile", e)


# --- Reminder tools ---


@mcp.tool()
async def add_reminder(content: str, group_id: str) -> str:
    """Create a reminder. Use 'user_profile' for personal, 'project_{id}' for project-specific.

    Args:
        content: What to remember.
        group_id: Target group.
    """
    try:
        return await reminder_tools.add_reminder(_client, content=content, group_id=group_id)
    except Exception as e:
        return _error("add_reminder", e)


@mcp.tool()
async def get_reminders(group_ids: list[str] | None = None) -> str:
    """List active reminders.

    Args:
        group_ids: Filter by groups. If omitted, returns all reminders.
    """
    try:
        return await reminder_tools.get_reminders(_client, group_ids=group_ids)
    except Exception as e:
        return _error("get_reminders", e)


# --- Onboarding tools ---


@mcp.tool()
async def check_project(project_path: str | None = None) -> str:
    """Check if the current project is known in the knowledge graph.

    Args:
        project_path: Path to project root. Defaults to cwd.
    """
    try:
        project_id = get_project_id(project_path)
        result = await onboard_tools.check_project(_client, project_id=project_id)
        if result["known"]:
            facts = "\n".join(f"- {f['fact']}" for f in result["facts"])
            return f"Project is known.\n{facts}"
        return "Project is not known. Use onboard_project to set it up."
    except Exception as e:
        return _error("check_project", e)


@mcp.tool()
async def onboard_project(project_path: str | None = None, description: str | None = None) -> str:
    """Analyze a project and store its essence in the knowledge graph.

    Reads README, manifests, and directory structure. Stores results in both
    the project group and the developer profile.

    Args:
        project_path: Path to project root. Defaults to cwd.
        description: Developer's own description (supplements auto-analysis).
    """
    try:
        path = project_path or os.getcwd()
        project_id = get_project_id(path)
        return await onboard_tools.onboard_project(
            _client, project_id=project_id, project_path=path, description=description
        )
    except Exception as e:
        return _error("onboard_project", e)


# --- Context tools ---


@mcp.tool()
async def get_context(project_path: str | None = None) -> str:
    """Retrieve the full context for the current session.

    Merges developer profile + project context + active reminders.

    Args:
        project_path: Path to project root. Defaults to cwd.
    """
    try:
        project_id = get_project_id(project_path)
        return await context_tools.get_context(_client, project_id=project_id)
    except Exception as e:
        return _error("get_context", e)


def main():
    mcp.run()


if __name__ == "__main__":
    main()
