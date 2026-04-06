"""Classify user messages using Claude Haiku for memory relevance."""

import subprocess

CLASSIFICATION_PROMPT = """You are a memory classifier for an AI coding assistant. Analyze this user message and classify it.

Respond with EXACTLY one word:
- SKIP: Routine instructions, code requests, debugging questions, greetings, follow-ups, "ok", "merci", questions about code
- USER: Personal preferences, habits, expertise, biographical facts, general workflow decisions
- PROJECT: Architecture decisions, tech stack choices, project conventions, team info, deployment details

Message:
<message>
{message}
</message>

Classification:"""


def classify_message(message: str) -> str | None:
    """Classify a message as SKIP, USER, or PROJECT using Claude Haiku.

    Returns None if classification fails.
    Uses the Claude CLI which authenticates via Max subscription.
    """
    prompt = CLASSIFICATION_PROMPT.format(message=message[:2000])

    try:
        result = subprocess.run(
            ["claude", "--model", "haiku", "--print", "-p", prompt],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode != 0:
            return None

        response = result.stdout.strip().upper()
        for token in ("SKIP", "USER", "PROJECT"):
            if token in response:
                return token
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None
