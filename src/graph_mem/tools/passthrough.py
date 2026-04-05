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
