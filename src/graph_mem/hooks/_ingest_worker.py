"""Background worker: classify then ingest a user message into Graphiti.

Invoked as a detached subprocess by user_prompt.py so it can run beyond hook timeout.
Usage: python -m graph_mem.hooks._ingest_worker <cwd> <message>

Flow: classify (Haiku CLI ~30-40s) -> if relevant, ingest (Graphiti ~30-60s).
"""

import asyncio
import sys

from graph_mem.client import GraphitiClient
from graph_mem.config import get_settings
from graph_mem.hooks._classifier import classify_message
from graph_mem.project_id import get_project_id

USER_PROFILE_GROUP = "user_profile"


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
    cwd = sys.argv[1]
    content = sys.argv[2]

    try:
        classification = classify_message(content)
        if classification is None or classification == "SKIP":
            sys.exit(0)

        if classification == "USER":
            group_id = USER_PROFILE_GROUP
        else:
            group_id = get_project_id(project_path=cwd)

        asyncio.run(ingest(group_id, content))
    except Exception as e:
        print(f"graph-mem ingest worker error: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
