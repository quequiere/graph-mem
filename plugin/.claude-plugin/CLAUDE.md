# graph-mem

Persistent memory plugin using a temporal knowledge graph (Graphiti + Neo4j).

## Available MCP tools

| Tool | What it does |
|------|-------------|
| `save_memory(content, scope)` | Save a fact, preference, or decision. scope: `user` or `project` |
| `search_memory(query, scope)` | Search the knowledge graph. scope: `all`, `user`, or `project` |

## When to use

- **save_memory**: when the user shares personal info (role, preferences, team), project context (stack, blockers, decisions), or explicitly asks to remember something.
- **search_memory**: when you need context about the user, the current project, or past decisions.

## Scoping

- `user` scope: global profile — follows the user across all projects
- `project` scope: tied to the current git repo (derived from git remote URL)

## Requirements

The Graphiti backend must be running (`docker compose up -d`). If the MCP server can't reach Graphiti at `http://127.0.0.1:8000`, tools will return connection errors.
