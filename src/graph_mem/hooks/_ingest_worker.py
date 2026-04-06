"""Background worker: send a user message to Graphiti for knowledge extraction.

Invoked as a detached subprocess by user_prompt.py so it can run beyond hook timeout.
Usage: graph-mem-ingest <group_id> <message>
"""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings


async def ingest(group_id: str, content: str) -> None:
    settings = get_settings()
    client = GraphitiClient(
        base_url=settings.graphiti_url, api_key=settings.graphiti_api_key
    )
    await client.add_messages(
        group_id=group_id,
        messages=[
            {
                "content": content,
                "role_type": "user",
                "role": "developer",
                "name": "auto-capture",
                "source_description": "Auto-captured from user message by UserPromptSubmit hook.",
            }
        ],
    )


def main():
    if len(sys.argv) < 3:
        sys.exit(1)
    group_id = sys.argv[1]
    content = sys.argv[2]
    try:
        asyncio.run(ingest(group_id, content))
    except Exception as e:
        print(f"graph-mem ingest worker error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
