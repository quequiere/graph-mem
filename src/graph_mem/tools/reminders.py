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
