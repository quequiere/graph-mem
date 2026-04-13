"""SessionEnd (Stop) hook: save session summary when a Claude Code session ends."""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.project_id import get_project_id
from graph_mem.tools.memory import save_session


async def run(summary: str) -> None:
    if not summary.strip():
        return

    settings = get_settings()
    project_id = get_project_id()

    async with GraphitiClient(
        base_url=settings.graphiti_url,
        api_key=settings.graphiti_api_key,
        timeout=settings.graphiti_timeout,
    ) as client:
        await save_session(client, summary=summary, project_id=project_id)


def main():
    summary = sys.stdin.read() if not sys.stdin.isatty() else ""
    if not summary.strip():
        return
    try:
        asyncio.run(run(summary))
    except Exception as e:
        print(f"graph-mem SessionEnd hook error: {e}", file=sys.stderr)
        sys.exit(0)


if __name__ == "__main__":
    main()
