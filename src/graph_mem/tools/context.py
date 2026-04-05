"""Context tool: merge user_profile + project context + reminders."""

from graph_mem.client import GraphitiClient
from graph_mem.tools.passthrough import _format_facts

USER_PROFILE = "user_profile"


async def get_context(
    client: GraphitiClient,
    project_id: str,
) -> str:
    """Retrieve merged context for the current session."""
    sections = []

    profile_result = await client.search(
        query="developer profile preferences expertise habits principles active projects",
        group_ids=[USER_PROFILE],
        max_facts=20,
    )
    profile_facts = profile_result.get("facts", [])
    if profile_facts:
        sections.append(f"## Developer Profile\n{_format_facts(profile_facts)}")

    project_result = await client.search(
        query="project stack team decisions goals blockers recent sessions context",
        group_ids=[project_id],
        max_facts=20,
    )
    project_facts = project_result.get("facts", [])
    if project_facts:
        sections.append(f"## Project Context\n{_format_facts(project_facts)}")

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
