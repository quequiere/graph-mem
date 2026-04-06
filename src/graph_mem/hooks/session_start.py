"""SessionStart hook: inject context at the beginning of a Claude Code session."""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.context import get_context


INSTRUCTIONS = """[graph-mem] Persistent knowledge graph memory is active.
- To SAVE info (preferences, decisions, personal facts): use `save_memory` tool
- To RECALL info (what you know about user/project): use `search_memory` tool
- ALWAYS search graph-mem when the user asks what you remember about them
- group_id: "user_profile" for personal info, "project_{id}" for project-specific"""


async def run() -> str:
    settings = get_settings()
    client = GraphitiClient(base_url=settings.graphiti_url, api_key=settings.graphiti_api_key)

    project_id = get_project_id()
    parts = [INSTRUCTIONS]

    # Inject any existing context (profile + project + reminders)
    try:
        context = await get_context(client, project_id=project_id)
        if context and "no context available" not in context.lower():
            parts.append(f"--- Recalled context ---\n{context}")
    except Exception:
        pass  # Graphiti may be down; don't block session start

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
