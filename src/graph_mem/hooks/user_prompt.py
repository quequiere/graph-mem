"""UserPromptSubmit hook: auto-capture valuable info from user messages.

Classifies user messages with Haiku and sends relevant ones to Graphiti.
"""

import json
import os
import subprocess
import sys

from graph_mem.hooks._classifier import classify_message
from graph_mem.project_id import get_project_id

USER_PROFILE_GROUP = "user_profile"


def main():
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        if not raw.strip():
            sys.exit(0)

        hook_input = json.loads(raw)

        # The prompt field contains the user message directly
        message = hook_input.get("prompt", "")
        if not message or len(message.strip()) < 10:
            sys.exit(0)

        classification = classify_message(message)
        if classification is None or classification == "SKIP":
            sys.exit(0)

        if classification == "USER":
            group_id = USER_PROFILE_GROUP
        else:
            cwd = hook_input.get("cwd")
            group_id = get_project_id(project_path=cwd)

        # Truncate for argv safety (Windows ~32KB limit)
        message_truncated = message[:8000]

        # Spawn detached worker -- survives hook timeout
        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            )
        else:
            popen_kwargs["start_new_session"] = True

        subprocess.Popen(
            [sys.executable, "-m", "graph_mem.hooks._ingest_worker", group_id, message_truncated],
            **popen_kwargs,
        )

    except Exception as e:
        print(f"graph-mem UserPromptSubmit error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
