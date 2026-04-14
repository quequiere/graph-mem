---
name: graph-mem
description: Persistent developer memory using knowledge graphs. Guides when and how to store and retrieve memories.
---

# graph-mem

Persistent memory plugin using a temporal knowledge graph (Graphiti + Neo4j).

## Available MCP Tools

| Tool | Purpose |
|---|---|
| `save_memory(content, scope)` | Save a fact, preference, or decision. scope: `user` or `project` |
| `search_memory(query, scope)` | Search the knowledge graph. scope: `all`, `user`, or `project` |

## When to Save

**SAVE (`save_memory`):**
- Developer expresses a preference ("I prefer X", "I don't like Y")
- Developer shares personal info (expertise, role, habits)
- Developer briefs a project (stack, team, objectives, problems)
- Developer makes a technical decision
- Developer says "remember that...", "remind me..."

**DO NOT SAVE:**
- Source code or file contents
- Secrets, tokens, passwords
- Ephemeral commands ("run npm install")
- Info derivable from code (file structure, imports, etc.)

## When to Search

**SEARCH (`search_memory`):**
- You need context about the user, the current project, or past decisions
- The user asks "what do you know about...", "do you remember..."
- Before making assumptions about the user's preferences or stack

## Scoping

- `user` scope: global profile — follows the user across all projects
- `project` scope: tied to the current git repo (derived from git remote URL)
- `all` scope (search only): search both user and project memories

## How to Use Injected Context

At session start, context is automatically injected containing:
- **Developer profile**: adapt tone and technicality level
- **Project context**: know what was done, decisions made, current state
- **Reminders**: mention them naturally when relevant

## Requirements

The Graphiti backend must be running (`docker compose up -d`). If the MCP server can't reach Graphiti at `http://127.0.0.1:8000`, tools will return connection errors.
