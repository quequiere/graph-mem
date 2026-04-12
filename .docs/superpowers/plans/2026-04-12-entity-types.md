# Entity Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Graphiti produce typed entity nodes (Person, Technology, Project, etc.) instead of generic `Entity` labels.

**Architecture:** Add a new `graphiti/entities.py` defining Pydantic entity/edge types, then patch `graphiti/ingest.py` to pass them to `add_episode()`. Server-side only — no client or DTO changes.

**Tech Stack:** Python, Pydantic, graphiti_core 0.22.0, Docker

---

### Task 1: Create entity type definitions

**Files:**
- Create: `graphiti/entities.py`

- [ ] **Step 1: Create `graphiti/entities.py`**

```python
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
```

- [ ] **Step 2: Verify the file parses correctly**

Run:
```bash
docker exec graph-mem-graphiti-1 python -c "
import sys; sys.path.insert(0, '/app')
from graph_service.entities import ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
print('Entity types:', list(ENTITY_TYPES.keys()))
print('Edge types:', list(EDGE_TYPES.keys()))
print('Edge map entries:', len(EDGE_TYPE_MAP))
"
```

This will fail because the file isn't in the container yet — that's expected. We verify after Docker rebuild in Task 2.

---

### Task 2: Patch ingest.py to pass entity types

**Files:**
- Modify: `graphiti/ingest.py` (lines 1-14 for imports, lines 69-78 for add_episode call)

- [ ] **Step 1: Add import to `graphiti/ingest.py`**

Add after the existing imports (after line 13):

```python
from graph_service.entities import EDGE_TYPE_MAP, EDGE_TYPES, ENTITY_TYPES
```

- [ ] **Step 2: Pass entity/edge types to `add_episode()`**

In the `add_messages_task` function, add the three parameters to the `add_episode()` call. The full function becomes:

```python
    async def add_messages_task(m: Message):
        await graphiti.add_episode(
            uuid=m.uuid,
            group_id=request.group_id,
            name=m.name,
            episode_body=f'{m.role or ""}({m.role_type}): {m.content}',
            reference_time=m.timestamp,
            source=EpisodeType.message,
            source_description=m.source_description,
            entity_types=ENTITY_TYPES,
            edge_types=EDGE_TYPES,
            edge_type_map=EDGE_TYPE_MAP,
        )
```

- [ ] **Step 3: Update the Dockerfile to copy entities.py**

Add one COPY line to `graphiti/Dockerfile`:

```dockerfile
FROM zepai/graphiti:latest

COPY zep_graphiti.py /app/graph_service/zep_graphiti.py
COPY ingest.py /app/graph_service/routers/ingest.py
COPY entities.py /app/graph_service/entities.py

CMD ["/app/.venv/bin/uvicorn", "graph_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 4: Rebuild and restart Docker**

Run:
```bash
docker compose up -d --build
```

Expected: containers rebuild and start successfully.

- [ ] **Step 5: Verify entities.py is loaded in the container**

Run:
```bash
docker exec graph-mem-graphiti-1 python -c "
from graph_service.entities import ENTITY_TYPES, EDGE_TYPES, EDGE_TYPE_MAP
print('Entity types:', list(ENTITY_TYPES.keys()))
print('Edge types:', list(EDGE_TYPES.keys()))
print('Edge map entries:', len(EDGE_TYPE_MAP))
"
```

Expected:
```
Entity types: ['Person', 'Organization', 'Technology', 'Project', 'Preference']
Edge types: ['WORKS_AT', 'USES', 'WORKS_ON', 'PREFERS', 'BUILT_WITH', 'OWNED_BY']
Edge map entries: 6
```

- [ ] **Step 6: Smoke test — ingest an episode and check Neo4j labels**

Run:
```bash
curl -s -X POST http://127.0.0.1:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "test_entity_types",
    "messages": [{
      "content": "Bruno works at Michelin and uses Python to build graph-mem.",
      "role_type": "user",
      "role": "developer",
      "name": "entity-type-test",
      "source_description": "Smoke test for entity type extraction."
    }]
  }'
```

Wait ~30s for async processing, then check Neo4j:

```bash
docker exec graph-mem-neo4j-1 cypher-shell -u neo4j -p graphiti \
  "MATCH (n) WHERE n.group_id = 'test_entity_types' RETURN labels(n) AS labels, n.name AS name ORDER BY n.name;"
```

Expected: nodes should have labels like `["Entity", "Person"]`, `["Entity", "Organization"]`, `["Entity", "Technology"]` instead of just `["Entity"]`.

- [ ] **Step 7: Clean up test data**

```bash
curl -s -X DELETE http://127.0.0.1:8000/group/test_entity_types
```
