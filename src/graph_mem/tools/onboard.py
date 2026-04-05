"""Onboarding tools: check_project, onboard_project."""

import os
from typing import Any

from graph_mem.client import GraphitiClient

USER_PROFILE = "user_profile"

MANIFEST_FILES = [
    "package.json", "pyproject.toml", "Cargo.toml", "go.mod",
    "pom.xml", "build.gradle", "Gemfile", "composer.json",
]

MAX_FILE_READ = 2000


async def check_project(
    client: GraphitiClient,
    project_id: str,
) -> dict[str, Any]:
    """Check if a project is known in the knowledge graph."""
    result = await client.search(
        query="project description objectives stack team",
        group_ids=[project_id],
        max_facts=5,
    )
    facts = result.get("facts", [])
    return {"known": len(facts) > 0, "facts": facts}


async def onboard_project(
    client: GraphitiClient,
    project_id: str,
    project_path: str,
    description: str | None = None,
) -> str:
    """Analyze a project and store its essence in the knowledge graph."""
    parts = []

    if description:
        parts.append(f"Developer description: {description}")

    readme_path = _find_readme(project_path)
    if readme_path:
        content = _read_truncated(readme_path)
        parts.append(f"README:\n{content}")

    for manifest in MANIFEST_FILES:
        manifest_path = os.path.join(project_path, manifest)
        if os.path.isfile(manifest_path):
            content = _read_truncated(manifest_path)
            parts.append(f"{manifest}:\n{content}")

    try:
        entries = sorted(os.listdir(project_path))
        dirs = [e for e in entries if os.path.isdir(os.path.join(project_path, e)) and not e.startswith(".")]
        files = [e for e in entries if os.path.isfile(os.path.join(project_path, e)) and not e.startswith(".")]
        parts.append(f"Top-level directories: {', '.join(dirs) if dirs else 'none'}")
        parts.append(f"Top-level files: {', '.join(files) if files else 'none'}")
    except OSError:
        pass

    project_name = project_id.removeprefix("project_").split("/")[-1]
    combined = "\n\n".join(parts) if parts else f"New project: {project_name} (no details available)"

    await client.add_messages(
        group_id=project_id,
        messages=[{
            "content": f"Project onboarding for {project_name}:\n\n{combined}",
            "role_type": "system",
            "role": "graph-mem",
            "name": "project-onboarding",
            "source_description": "Automatic project analysis during first encounter.",
        }],
    )

    summary = description or f"Project {project_name}"
    await client.add_messages(
        group_id=USER_PROFILE,
        messages=[{
            "content": f"Developer works on project: {project_name}. {summary}",
            "role_type": "system",
            "role": "graph-mem",
            "name": "project-reference",
            "source_description": "Project registered in developer profile.",
        }],
    )

    return f"Project '{project_name}' onboarded and registered in profile."


def _find_readme(project_path: str) -> str | None:
    try:
        for entry in os.listdir(project_path):
            if entry.lower().startswith("readme"):
                return os.path.join(project_path, entry)
    except OSError:
        pass
    return None


def _read_truncated(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(MAX_FILE_READ)
            if len(content) == MAX_FILE_READ:
                content += "\n... (truncated)"
            return content
    except OSError:
        return "(could not read file)"
