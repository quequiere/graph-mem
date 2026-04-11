# graph-mem

**Your AI coding assistant forgets everything between sessions. graph-mem fixes that.**

[![PyPI](https://img.shields.io/pypi/v/graph-mem)](https://pypi.org/project/graph-mem/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-not%20usable%20yet-red)](https://github.com/quequiere/graph-mem/issues)

> **⚠️ Not usable yet.** This is the first development milestone — the code is written but the project has not been tested end-to-end against a live Graphiti instance. Do not try to use it yet. Follow the repo or [open an issue](https://github.com/quequiere/graph-mem/issues) to be notified when it's ready.

<p align="center">
  <img src="docs/graph-mem-hero.svg" alt="graph-mem knowledge graph — your projects, teammates, decisions and skills connected over time" width="800"/>
</p>

## What it looks like in practice

**Session 1** — Working on a FastAPI project with Julie. graph-mem learns:
> *"Developer uses Python, works on project_atlas with Julie (tech lead), prefers pnpm, uses TDD."*

**Session 12** — You start learning Rust on a side project. graph-mem adds:
> *"Developer is learning Rust (beginner), started project_oxide, still active on project_atlas."*

**Session 47** — You open a new terminal. Before you type anything, your assistant already knows:
> *"You're in the project_atlas repo. Julie pushed a migration yesterday. There's a CI blocker (OOM on E2E tests). You prefer no mocks in integration tests. Reminder: bump deps to v2.1."*

This happens automatically via Claude Code hooks — no manual prompt engineering required.

## Why not just use CLAUDE.md or mem0?

`CLAUDE.md` is a static file you maintain by hand — it doesn't track relationships or evolve over time. mem0 is a flat vector store that retrieves similar text; it doesn't know that Julie is your tech lead on project_atlas, or that the OOM blocker is blocking *that specific project's* CI. graph-mem builds a **temporal knowledge graph**: entities connect to each other, facts carry timestamps, and the graph grows smarter as you work.

> **Privacy:** By default, everything runs on your machine — Ollama, Graphiti, and Neo4j all live in Docker, and no data leaves your host. If you configure a remote provider in `.env` (OpenRouter, OpenAI, etc.), session summaries and saved facts will be sent to that provider during entity extraction. Review what graph-mem stores before enabling a remote mode on sensitive work projects.

## MCP — what is it?

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) is the plugin system that lets tools like graph-mem extend what Claude, Cursor, and other AI assistants can do. If you use Claude Code or Cursor, you likely already have MCP support — you just add graph-mem to your config.

## Quick start

**Prerequisites:** Python 3.11+, Docker & Docker Compose v2.20+.

By default, graph-mem runs **fully local** — no API keys required, no data leaves your machine. Ollama, Graphiti and Neo4j all run in Docker.

### 1. Start the backend

```bash
git clone https://github.com/quequiere/graph-mem && cd graph-mem
cp .env.example .env
docker compose up -d
```

First launch downloads two Ollama models (~3.5 GB). Subsequent launches are instant thanks to the persistent volume. Neo4j needs ~2 GB RAM, Ollama needs ~4 GB.

### 2. Connect your MCP client

```bash
pip install graph-mem
# or run without installing: uvx graph-mem
```

Add to Claude Code (`~/.claude/claude_desktop_config.json` or via `claude mcp add`):

```json
{
  "mcpServers": {
    "graph-mem": {
      "command": "uvx",
      "args": ["graph-mem"],
      "env": { "GRAPHITI_URL": "http://localhost:8000" }
    }
  }
}
```

Same config block works for Cursor, Windsurf, and any other MCP-compatible client.

### 3. (Optional) Enable automatic hooks

Add to `~/.claude/settings.json` for automatic context injection at session start and auto-save at stop:

```json
{
  "hooks": {
    "SessionStart": [{ "command": "graph-mem-session-start" }],
    "Stop": [{ "command": "graph-mem-session-end", "blocking": true }]
  }
}
```

`graph-mem-session-start` and `graph-mem-session-end` are CLI commands installed with `pip install graph-mem`. The `Stop` hook has a 30-second timeout — if Graphiti is unreachable, it exits with a warning and your session ends normally (no data loss; re-save manually via `save_session` next time). Without hooks, call `get_context` and `save_session` manually from the chat.

## Other modes

Need a remote LLM, a local embedder, or a mix? Everything is configurable from [`.env`](.env.example). Each variable has an inline comment showing the remote alternative — just swap the values you want, set `COMPOSE_PROFILES=` to skip the bundled Ollama container, and run `docker compose up -d` again. **The command is always the same.**

## What gets stored?

graph-mem captures the **essence** of your work — not your code. Here's what the knowledge graph tracks:

| Entity type | Examples | How it's captured |
|---|---|---|
| **Developer profile** | Expertise, seniority, habits | Accumulated across sessions |
| **Technologies** | Languages, frameworks, tools | Detected from project context |
| **Projects** | Name, stack, team, status | `onboard_project` or auto-detected |
| **People** | Colleagues, roles, relationships | Mentioned in conversations |
| **Preferences** | Tooling choices, coding style | Stated or observed over time |
| **Principles** | "TDD always", "no mocks in integration" | Stated by the developer |
| **Blockers** | CI failures, environment issues | Reported during sessions |
| **Reminders** | "bump deps", "renew API key" | Explicit `add_reminder` calls |
| **Decisions** | Architecture choices, trade-offs | Captured in session summaries |

Everything is **temporally aware** — graph-mem knows when you started learning Rust, when you switched from npm to pnpm, and when a blocker was resolved.

## MCP Tools

### High-level tools

| Tool | What it does |
|---|---|
| `get_context` | Injects your full profile + current project context + active reminders |
| `onboard_project` | Analyzes and memorizes a new project's structure, stack, and team |
| `save_session` | Summarizes and persists the current session to the knowledge graph |
| `save_memory` | Stores a specific fact, preference, or decision |
| `get_profile` | Retrieves your developer profile |
| `check_project` | Checks whether the current project is known; suggests onboarding if not |
| `add_reminder` | Adds a reminder that will surface in future sessions |
| `get_reminders` | Lists active reminders |

### Passthrough tools (direct Graphiti access)

| Tool | What it does |
|---|---|
| `add_raw_memory` | Adds raw text or JSON directly to the knowledge graph |
| `search_entities` | Searches for entities (nodes) by natural language |
| `search_facts` | Searches for facts (relationships) between entities |
| `reset_memory` | Clears memory for a given scope (`user_profile` or `project_{identifier}`) — destructive, cannot be undone |
| `status` | Checks Graphiti connection health |

## Architecture

```
Your machine                          Docker (local by default)
┌──────────────────────┐              ┌──────────────────────────┐
│ Claude Code / Cursor │              │  Graphiti REST API       │
│         │            │    HTTP      │         │                │
│    stdio│            │ ──────────►  │    graphiti_core         │
│         ▼            │              │      │         │         │
│ graph-mem MCP server │              │    Neo4j     Ollama*     │
└──────────────────────┘              └──────────────────────────┘
                                       * Ollama is optional — disable
                                         via COMPOSE_PROFILES= when
                                         using a remote LLM provider.
```

- **graph-mem** runs locally as an MCP server (stdio transport)
- **Graphiti** runs in Docker, handles entity extraction, embeddings, and graph storage
- **Ollama** runs in Docker by default and serves both the LLM and the embedder; swap to any OpenAI-compatible remote provider by editing `.env`
- Communication is plain HTTP — no JSON-RPC, works behind corporate proxies
- Memory is scoped by `user_profile` (global) and `project_{identifier}` (per-repo, derived from git remote URL); both scopes are merged at query time

## Built on

- **[Graphiti](https://github.com/getzep/graphiti)** — Temporally-aware knowledge graph framework by Zep
- **[FastMCP](https://github.com/jlowin/fastmcp)** — Python MCP server framework
- **[Neo4j](https://neo4j.com)** — Graph database backend

## Contributing

Open an issue first to discuss what you'd like to change. Pull requests welcome.

## License

Apache-2.0 — see [LICENSE](LICENSE)
