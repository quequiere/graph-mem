---
name: graph-mem
description: Persistent developer memory using knowledge graphs. Guides when and how to store and retrieve memories.
---

# graph-mem Memory Skill

You have access to a persistent knowledge graph that remembers information about the developer and their projects across sessions. Context is automatically injected at session start.

## When to Store Memory

**STORE (use `save_memory`):**
- Developer expresses a preference ("I prefer X", "I don't like Y")
- Developer shares personal info (expertise, role, habits)
- Developer briefs a project (stack, team, objectives, problems)
- Developer makes a technical decision
- Developer says "remember that...", "remind me..."

**DO NOT STORE:**
- Source code or file contents
- Secrets, tokens, passwords
- Ephemeral commands ("run npm install")
- Info derivable from code (file structure, imports, etc.)

## How to Choose group_id

- Info about the developer (preference, expertise, habit, personal) -> `user_profile`
- Info about a specific project -> `project_{id}` (derived from cwd, provided by tools)
- When in doubt -> `user_profile`

## Available Tools

| Tool | Purpose |
|---|---|
| `get_context` | Get merged developer profile + project context + reminders |
| `save_memory` | Store a specific piece of information |
| `save_session` | Save a session summary (usually automatic) |
| `get_profile` | Get the developer's profile |
| `check_project` | Check if a project is known |
| `onboard_project` | Analyze and register a new project |
| `add_reminder` | Create a reminder |
| `get_reminders` | List active reminders |
| `search_facts` | Search for relationships in the graph |
| `search_entities` | Search for entities by query |
| `add_raw_memory` | Low-level: add an episode directly |
| `reset_memory` | DANGEROUS: purge all data for given groups |
| `status` | Check if Graphiti is running |

## How to Use Injected Context

At session start, context is injected containing:
- **Developer profile**: Adapt tone and technicality level
- **Project context**: Know what was done, decisions made, current state
- **Reminders**: Mention them naturally when relevant
- **Principles**: Respect them in your suggestions (e.g., if "no mocks in integration tests", don't suggest mocks)

## Reminders

When the developer says "remind me to..." or "don't forget to...":
1. Use `add_reminder` with the appropriate group_id
2. Reminders surface automatically at session start via `get_reminders`
