# Entity Types for Graphiti REST Server

**Date:** 2026-04-12
**Status:** Approved
**Scope:** Patch Graphiti Docker server to pass entity/edge types to `add_episode()`

## Problem

The Graphiti REST server (`ingest.py`) calls `graphiti_core.Graphiti.add_episode()` without
`entity_types`, `edge_types`, or `edge_type_map`. As a result, all Neo4j nodes get a generic
`Entity` label with no semantic categorization. The front-end visualization (`categorizeNode()`)
compensates with naive keyword matching, but almost everything ends up as "Tech".

`graphiti_core` 0.22.0 (current Docker image) natively supports these parameters — the REST
server simply never passes them.

## Solution

Hardcode entity and edge types **server-side** in a new `graphiti/entities.py` file. The
`ingest.py` route imports them and passes them to every `add_episode()` call. No changes to
the REST DTO, client, or Dockerfile.

### Why server-side, not per-request

Entity types are a property of the graph-mem deployment, not a per-request choice. Passing
Pydantic model definitions over HTTP would require complex serialization. The server already
owns the Graphiti client — it should own the schema too.

## Entity Types

| Type | Description | Custom fields |
|------|-------------|---------------|
| Person | Named individual (developer, colleague, manager) | `role: str` |
| Organization | Company, team, department | — |
| Technology | Language, framework, library, tool, service | — |
| Project | Repository, application, service, codebase | — |
| Preference | Convention, technical choice, developer habit | — |

## Edge Types

| Edge | Description |
|------|-------------|
| WORKS_AT | Person → Organization |
| USES | Person → Technology |
| WORKS_ON | Person → Project |
| PREFERS | Person → Preference |
| BUILT_WITH | Project → Technology |
| OWNED_BY | Project → Organization |

## Edge Type Map (constraints)

```python
{
    ("Person", "Organization"): ["WORKS_AT"],
    ("Person", "Technology"): ["USES"],
    ("Person", "Project"): ["WORKS_ON"],
    ("Person", "Preference"): ["PREFERS"],
    ("Project", "Technology"): ["BUILT_WITH"],
    ("Project", "Organization"): ["OWNED_BY"],
}
```

## Files Changed

| File | Action |
|------|--------|
| `graphiti/entities.py` | **New** — entity types, edge types, edge type map |
| `graphiti/ingest.py` | **Patch** — import entities + pass 3 params to `add_episode()` |

**Not changed:** `src/graph_mem/client.py`, `src/graph_mem/server.py`, DTO, Dockerfile,
`docker-compose.yml`.

## Deployment

```bash
docker compose up -d --build
```

Existing data in Neo4j keeps the old `Entity` label. New episodes will produce typed nodes.
To re-type existing data, clear and re-ingest.

## Constraints

- `graphiti_core` 0.22.0 does not support `custom_extraction_instructions` (added in 0.28+).
  Entity docstrings serve as the LLM's extraction guide.
- The Pydantic models must avoid field name collisions with `EntityNode` core fields
  (`uuid`, `name`, `labels`, `summary`, `group_id`, `created_at`).

## Future Work

- Update front-end `categorizeNode()` to use actual Neo4j labels instead of keyword heuristics
- Consider upgrading to graphiti_core 0.28+ for `custom_extraction_instructions` support
