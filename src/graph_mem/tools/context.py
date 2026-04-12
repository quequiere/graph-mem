"""Context tool: merge user_profile + project context + reminders."""

import asyncio

from graph_mem.client import GraphitiClient
from graph_mem.config import USER_PROFILE
from graph_mem.tools.passthrough import _format_facts


async def _fetch_sections(
    client: GraphitiClient,
    project_id: str,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Fetch raw fact lists from Graphiti in parallel. Returns (profile, project, reminders)."""
    profile_task = client.search(
        query="developer profile preferences expertise habits principles active projects",
        group_ids=[USER_PROFILE],
        max_facts=20,
    )
    project_task = client.search(
        query="project stack team decisions goals blockers recent sessions context",
        group_ids=[project_id],
        max_facts=20,
    )
    reminder_user_task = client.search(
        query="active reminders things to remember",
        group_ids=[USER_PROFILE],
        max_facts=10,
    )
    reminder_project_task = client.search(
        query="active reminders things to remember",
        group_ids=[project_id],
        max_facts=10,
    )

    profile_result, project_result, reminder_user, reminder_project = await asyncio.gather(
        profile_task, project_task, reminder_user_task, reminder_project_task,
        return_exceptions=True,
    )

    profile_facts = (
        profile_result.get("facts", [])
        if not isinstance(profile_result, Exception) else []
    )
    project_facts = (
        project_result.get("facts", [])
        if not isinstance(project_result, Exception) else []
    )
    reminder_facts = []
    for r in [reminder_user, reminder_project]:
        if not isinstance(r, Exception):
            reminder_facts.extend(r.get("facts", []))

    return profile_facts, project_facts, reminder_facts


async def get_context_sections(
    client: GraphitiClient,
    project_id: str,
) -> dict:
    """Retrieve context sections with both raw facts and formatted text.

    Returns a dict with keys: profile, project, reminders, formatted.
    """
    profile_facts, project_facts, reminder_facts = await _fetch_sections(client, project_id)

    sections = []
    if profile_facts:
        sections.append(f"## Developer Profile\n{_format_facts(profile_facts)}")
    if project_facts:
        sections.append(f"## Project Context\n{_format_facts(project_facts)}")
    if reminder_facts:
        sections.append(f"## Reminders\n{_format_facts(reminder_facts)}")

    formatted = "\n\n".join(sections) if sections else "No context available yet for this session."

    return {
        "profile": profile_facts,
        "project": project_facts,
        "reminders": reminder_facts,
        "formatted": formatted,
    }


async def get_context(
    client: GraphitiClient,
    project_id: str,
) -> str:
    """Retrieve merged context for the current session."""
    result = await get_context_sections(client, project_id)
    return result["formatted"]
