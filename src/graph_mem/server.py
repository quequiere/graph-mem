"""graph-mem MCP server."""

from mcp.server.fastmcp import FastMCP

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.tools import passthrough

mcp = FastMCP("graph-mem")

_settings = get_settings()
_client = GraphitiClient(
    base_url=_settings.graphiti_url,
    api_key=_settings.graphiti_api_key,
)


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
