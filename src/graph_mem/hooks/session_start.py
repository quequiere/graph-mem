"""SessionStart hook: inject context at the beginning of a Claude Code session."""

import asyncio
import json
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.context import get_context, get_context_sections


INSTRUCTIONS = """[graph-mem] Persistent knowledge graph memory is active.
- User messages are AUTO-CAPTURED by a background hook — no need to call `save_memory`
- To RECALL info (what you know about user/project): use `search_memory` tool
- ALWAYS search graph-mem when the user asks what you remember about them"""


def _build_system_message(sections: dict) -> str:
    """Build a user-visible summary from context sections."""
    lines = ["[graph-mem] Session context loaded"]

    profile_facts = sections.get("profile", [])
    project_facts = sections.get("project", [])
    reminder_facts = sections.get("reminders", [])

    counts = []
    if profile_facts:
        counts.append(f"{len(profile_facts)} profile")
    if project_facts:
        counts.append(f"{len(project_facts)} project")
    if reminder_facts:
        counts.append(f"{len(reminder_facts)} reminder")

    if counts:
        lines[0] += f" ({', '.join(counts)} facts recalled)"
    else:
        lines[0] += " (no prior context)"

    # Show a few key facts as preview
    preview_facts = (profile_facts[:2] + project_facts[:2] + reminder_facts[:1])
    for f in preview_facts:
        fact_text = f if isinstance(f, str) else f.get("fact", "")
        if fact_text:
            # Truncate long facts for terminal display
            if len(fact_text) > 100:
                fact_text = fact_text[:97] + "..."
            lines.append(f"  - {fact_text}")

    return "\n".join(lines)


async def run() -> dict:
    """Return a dict with additionalContext (for Claude) and systemMessage (for user)."""
    settings = get_settings()

    project_id = get_project_id()
    parts = [INSTRUCTIONS]
    system_message = "[graph-mem] Could not connect to Graphiti"

    # Inject any existing context (profile + project + reminders)
    try:
        async with GraphitiClient(
            base_url=settings.graphiti_url,
            api_key=settings.graphiti_api_key,
            timeout=settings.graphiti_timeout,
        ) as client:
            sections = await get_context_sections(client, project_id=project_id)
            context = sections.get("formatted")
            if context and "no context available" not in context.lower():
                parts.append(f"--- Recalled context ---\n{context}")
            system_message = _build_system_message(sections)
    except Exception:
        pass  # Graphiti may be down; don't block session start

    return {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": "\n\n".join(parts),
        },
        "systemMessage": system_message,
    }


def main():
    try:
        result = asyncio.run(run())
        print(json.dumps(result))
    except Exception as e:
        print(f"graph-mem SessionStart hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
