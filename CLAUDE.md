# graph-mem - Project Tracking

## Project Overview

**graph-mem** is a Claude Code plugin that provides persistent, intelligent memory using Graphiti's knowledge graph. Unlike flat memory stores, it leverages entity relationships, temporal awareness, and semantic search to build a rich understanding of the developer over time.

## Status: Design Complete - Ready for Implementation Planning

## Design Spec

Full design document: `.doc/specs/2026-04-05-graph-mem-design.md`

## Architecture Decisions (validated)

- **Language**: Python
- **Architecture**: Local MCP server (stdio) calls Graphiti REST API (httpx over HTTP). No graphiti_core dependency, no MCP-to-MCP.
- **Distribution**: `pip install graph-mem` / `uvx graph-mem` locally. Graphiti deployed via Docker (`zepai/graphiti:latest`).
- **Multi-CLI compatible**: MCP tools usable by any CLI (Claude Code, Cursor, etc.), hooks specific to Claude Code
- **Skills**: Distributable skill to guide CLI agents in using our memory system
- **No code indexing**: Memory captures project essence, developer habits, and workflow patterns - not source code
- **Embeddings**: Handled by Graphiti core internally - no client-side pre-computation needed
- **Privacy**: Not managed for now (future consideration)
- **Open source**: GitHub standards, license TBD

## Group ID Strategy (validated)

- `user_profile` : global developer profile (preferences, expertise, habits, active projects)
- `project_{identifier}` : per-project detail (team, stack, decisions, sessions, blockers)
- Project identifier: normalized git remote URL, fallback to directory name
- At query time: merge results from both scopes

## MCP Tools (validated)

Custom (8): `get_context`, `onboard_project`, `save_session`, `save_memory`, `get_profile`, `check_project`, `add_reminder`, `get_reminders`

Passthrough (5): `add_raw_memory`, `search_entities`, `search_facts`, `reset_memory`, `status`

## Hooks (validated)

- **SessionStart**: check_project + get_context + get_reminders -> inject into agent
- **SessionEnd (Stop)**: LLM summary -> save_session

## Reference Projects (external, not included in this repo)

- [Graphiti](https://github.com/getzep/graphiti) - Knowledge graph framework + REST API server (our backend)

## TODO

- [x] Design spec (`.doc/specs/2026-04-05-graph-mem-design.md`)
- [ ] Implementation plan
- [ ] Set up Python project structure (pyproject.toml, src layout)
- [ ] Implement MCP server with passthrough tools
- [ ] Implement custom tools
- [ ] Implement Claude Code hooks
- [ ] Write skill (SKILL.md)
- [ ] Choose license
- [ ] Set up build pipeline
- [ ] Tests

## Git Convention

- Commits authored by the developer (not Claude)
- Commit regularly, small increments
- Conventional commits style

## Documentation

- `README.md` - Public project description
- `CLAUDE.md` - This file, project tracking for Claude sessions
- `.doc/` - Supplementary documentation
- `.doc/specs/` - Design specifications
