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
