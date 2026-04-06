"""Derive a stable project identifier from git remote or directory name."""

import os
import re
import subprocess


def normalize_git_url(url: str) -> str:
    """Normalize a git remote URL to 'host/owner/repo' form."""
    ssh_match = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", url)
    if ssh_match:
        return f"{ssh_match.group(1)}/{ssh_match.group(2)}"
    https_match = re.match(r"https?://([^/]+)/(.+?)(?:\.git)?$", url)
    if https_match:
        return f"{https_match.group(1)}/{https_match.group(2)}"
    return url


def get_project_id(project_path: str | None = None) -> str:
    """Get the project group_id for Graphiti.

    Uses git remote origin URL if available, falls back to directory name.
    Returns: 'project_{identifier}'
    """
    path = project_path or os.getcwd()
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            normalized = normalize_git_url(result.stdout.strip())
            # Graphiti requires alphanumeric, dashes, or underscores only
            sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", normalized)
            return f"project_{sanitized}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return f"project_{os.path.basename(os.path.abspath(path))}"
