"""SessionStart hook: inject context at the beginning of a Claude Code session."""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.context import get_context
from graph_mem.tools.onboard import check_project
from graph_mem.tools.reminders import get_reminders


async def run() -> str:
    settings = get_settings()
    client = GraphitiClient(base_url=settings.graphiti_url, api_key=settings.graphiti_api_key)

    project_id = get_project_id()
    parts = []

    project_status = await check_project(client, project_id=project_id)
    if not project_status["known"]:
        parts.append(
            "New project detected. Use `onboard_project` to set it up in the knowledge graph."
        )

    context = await get_context(client, project_id=project_id)
    if context and "no context available" not in context.lower():
        parts.append(context)

    if not parts:
        parts.append("graph-mem: No context available yet. Start by onboarding this project.")

    return "\n\n".join(parts)


def main():
    try:
        output = asyncio.run(run())
        print(output)
    except Exception as e:
        print(f"graph-mem SessionStart hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
