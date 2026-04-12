"""UserPromptSubmit hook: auto-capture valuable info from user messages.

Spawns a detached worker that classifies (Haiku) then ingests (Graphiti).
Classification is done in the worker to avoid hook timeout (~20s) since
the Claude CLI cold-starts in ~30-40s.
"""

import json
import os
import subprocess
import sys


def main():
    try:
        raw = sys.stdin.read() if not sys.stdin.isatty() else ""
        if not raw.strip():
            sys.exit(0)

        hook_input = json.loads(raw)

        message = hook_input.get("prompt", "")
        if not message or len(message.strip()) < 10:
            sys.exit(0)

        cwd = hook_input.get("cwd", "")
        # Truncate for argv safety (Windows ~32KB limit)
        message_truncated = message[:8000]

        # Spawn detached worker -- survives hook timeout
        popen_kwargs: dict = {
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
                | subprocess.CREATE_NO_WINDOW
            )
        else:
            popen_kwargs["start_new_session"] = True

        subprocess.Popen(
            [sys.executable, "-m", "graph_mem.hooks._ingest_worker", cwd, message_truncated],
            **popen_kwargs,
        )

    except Exception as e:
        print(f"graph-mem UserPromptSubmit error: {e}", file=sys.stderr)

    sys.exit(0)


if __name__ == "__main__":
    main()
