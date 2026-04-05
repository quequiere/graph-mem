"""Memory storage tools: save_memory, save_session."""

from graph_mem.client import GraphitiClient


async def save_memory(
    client: GraphitiClient,
    content: str,
    group_id: str,
) -> str:
    """Store a specific piece of information in the knowledge graph."""
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
    """Send a session summary to Graphiti for entity/relation extraction."""
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
