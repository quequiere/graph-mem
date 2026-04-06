"""SessionStart hook: inject context at the beginning of a Claude Code session."""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.context import get_context
from graph_mem.tools.onboard import check_project
from graph_mem.tools.reminders import get_reminders


INSTRUCTIONS = """[graph-mem] You have a persistent knowledge graph memory via MCP tools (graph-mem server).
- Use `save_memory` to store developer preferences, project decisions, personal info
- Use `get_context` or `search_facts` to recall information about the developer or project
- When the user asks what you remember, ALWAYS use graph-mem tools to search, not just built-in memory
- group_id: "user_profile" for personal info, "project_{id}" for project-specific info"""


async def run() -> str:
    settings = get_settings()
    client = GraphitiClient(base_url=settings.graphiti_url, api_key=settings.graphiti_api_key)

    project_id = get_project_id()
    parts = [INSTRUCTIONS]

    project_status = await check_project(client, project_id=project_id)
    if not project_status["known"]:
        parts.append(
            "New project detected. Use `onboard_project` to set it up in the knowledge graph."
        )

    context = await get_context(client, project_id=project_id)
    if context and "no context available" not in context.lower():
        parts.append(context)

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
