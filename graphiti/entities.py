"""Entity and edge type definitions for graph-mem knowledge graph.

Passed to graphiti_core.Graphiti.add_episode() so that extracted nodes
get semantic labels (Person, Technology, ...) instead of generic Entity.
"""

from pydantic import BaseModel, Field


# ---- Entity types ----

class Person(BaseModel):
    """A named individual — developer, colleague, manager, or any person
    mentioned by name in conversation."""

    role: str = Field(default="", description="Role or job title if known")


class Organization(BaseModel):
    """A company, team, department, or named organizational unit."""


class Technology(BaseModel):
    """A programming language, framework, library, tool, protocol, or
    service that developers use to build software. Examples: Python,
    FastAPI, Docker, Neo4j, React, Kubernetes."""


class Project(BaseModel):
    """A repository, application, service, or codebase that is being
    developed or maintained."""


class Preference(BaseModel):
    """A developer convention, technical choice, or personal habit.
    Examples: 'prefers pnpm over npm', 'uses vim keybindings',
    'always writes tests first'."""


ENTITY_TYPES: dict[str, type[BaseModel]] = {
    "Person": Person,
    "Organization": Organization,
    "Technology": Technology,
    "Project": Project,
    "Preference": Preference,
}


# ---- Edge types ----

class WorksAt(BaseModel):
    """Person works at an Organization."""


class Uses(BaseModel):
    """Person uses a Technology."""


class WorksOn(BaseModel):
    """Person works on a Project."""


class Prefers(BaseModel):
    """Person has a Preference."""


class BuiltWith(BaseModel):
    """Project is built with a Technology."""


class OwnedBy(BaseModel):
    """Project is owned by an Organization."""


EDGE_TYPES: dict[str, type[BaseModel]] = {
    "WORKS_AT": WorksAt,
    "USES": Uses,
    "WORKS_ON": WorksOn,
    "PREFERS": Prefers,
    "BUILT_WITH": BuiltWith,
    "OWNED_BY": OwnedBy,
}

EDGE_TYPE_MAP: dict[tuple[str, str], list[str]] = {
    ("Person", "Organization"): ["WORKS_AT"],
    ("Person", "Technology"): ["USES"],
    ("Person", "Project"): ["WORKS_ON"],
    ("Person", "Preference"): ["PREFERS"],
    ("Project", "Technology"): ["BUILT_WITH"],
    ("Project", "Organization"): ["OWNED_BY"],
}
