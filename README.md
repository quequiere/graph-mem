# graph-mem

**Your AI coding assistant forgets everything between sessions. graph-mem fixes that.**

[![PyPI](https://img.shields.io/pypi/v/graph-mem)](https://pypi.org/project/graph-mem/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-orange)](https://github.com/quequiere/graph-mem/issues)

<p align="center">
  <img src="docs/graph-mem-hero.svg" alt="graph-mem knowledge graph — your projects, teammates, decisions and skills connected over time" width="800"/>
</p>

> **Alpha — not usable yet.** The code is written but hasn't been tested end-to-end against a live Graphiti instance. [Watch the repo](https://github.com/quequiere/graph-mem) to know when it ships.

## Demo

> 🚧 **Under construction.** A short screencast is coming once the embedding benchmark lands. In the meantime, here's the idea:

**Session 1** — You start a FastAPI project with Julie.
> *graph-mem learns: Python dev, project_atlas, Julie (tech lead), prefers pnpm, does TDD.*

**Session 47** — New terminal, new day. Before you type anything:
> *"You're in project_atlas. Julie pushed a migration yesterday. CI is blocked on OOM in E2E. Reminder: bump deps to v2.1."*

No prompt engineering. Claude Code hooks inject the context automatically at session start.

## Why not just CLAUDE.md or mem0?

`CLAUDE.md` is a static file you maintain by hand. `mem0` is a flat vector store that retrieves similar text. Neither knows that Julie is your tech lead *on project_atlas*, or that the OOM blocker is blocking *that specific* CI. **graph-mem is a temporal knowledge graph** — entities connect, facts carry timestamps, and the graph grows smarter as you work.

## Quick start

**Prereqs:** Python 3.11+, Docker Compose v2.20+. The default stack is **fully local** (Ollama + Graphiti + Neo4j in Docker, no API keys, no data leaves your machine).

```bash
git clone https://github.com/quequiere/graph-mem && cd graph-mem
cp .env.example .env
docker compose up -d          # Neo4j + Graphiti + Ollama (~3.5 GB on first run)
pip install graph-mem
```

Then add graph-mem to your MCP client (`~/.claude/claude_desktop_config.json`, Cursor, Windsurf, …):

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

<details>
<summary><b>Enable automatic hooks</b> — context injection at session start, auto-save at stop (recommended)</summary>

Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [{ "command": "graph-mem-session-start" }],
    "Stop": [{ "command": "graph-mem-session-end", "blocking": true }]
  }
}
```

The `Stop` hook has a 30s timeout. If Graphiti is unreachable it exits with a warning and your session ends normally — no data loss, just re-save via `save_session` next time.

</details>

<details>
<summary><b>Use a remote LLM provider</b> (OpenRouter, OpenAI, Anthropic, …)</summary>

Edit `.env` — every variable has an inline comment showing the remote alternative. Then skip the bundled Ollama container:

```bash
COMPOSE_PROFILES= docker compose up -d
```

⚠️ Session summaries and saved facts will be sent to that provider during entity extraction. Review what graph-mem stores before enabling a remote mode on sensitive work projects.

</details>

## What gets stored

graph-mem captures the **essence** of your work — not your code:

- **Developer profile** — expertise, seniority, habits
- **Projects** — name, stack, team, status
- **People** — colleagues, roles, relationships
- **Preferences & principles** — *"TDD always"*, *"no mocks in integration"*
- **Blockers & reminders** — CI failures, deps to bump, keys to renew
- **Decisions** — architecture choices captured in session summaries

Everything is **temporally aware** — graph-mem knows *when* you started learning Rust, *when* you switched from npm to pnpm, *when* a blocker was resolved.

## MCP tools

The three you'll actually use day-to-day:

| Tool | What it does |
|---|---|
| `get_context` | Injects your profile + current project context + active reminders |
| `save_memory` | Stores a specific fact, preference, or decision |
| `save_session` | Summarizes and persists the current session to the graph |

<details>
<summary>Full tool list (onboarding, reminders, passthrough Graphiti access, …)</summary>

**High-level**
- `onboard_project` — analyzes and memorizes a project's stack and team
- `get_profile` — retrieves your developer profile
- `check_project` — checks if the current project is known, suggests onboarding if not
- `add_reminder` / `get_reminders` — explicit reminders surfaced in future sessions

**Passthrough (direct Graphiti access)**
- `add_raw_memory` — raw text or JSON straight into the graph
- `search_entities` / `search_facts` — natural-language search over nodes and relationships
- `reset_memory` — destructive, per scope (`user_profile` or `project_{identifier}`)
- `status` — Graphiti connection health check

</details>

## Architecture

```
Your machine                          Docker (local by default)
┌──────────────────────┐              ┌──────────────────────────┐
│ Claude Code / Cursor │    stdio     │  Graphiti REST API       │
│         │            │  ──────►     │         │                │
│         ▼            │    HTTP      │    graphiti_core         │
│ graph-mem MCP server │              │      │         │         │
└──────────────────────┘              │    Neo4j     Ollama*     │
                                      └──────────────────────────┘
                                       * disable via COMPOSE_PROFILES=
                                         to use a remote LLM provider
```

Memory is scoped by `user_profile` (global) and `project_{identifier}` (per-repo, derived from git remote URL). Both scopes merge at query time. Communication is plain HTTP — no JSON-RPC, works behind corporate proxies.

## Model choice

Two independent models, both swappable via `.env`, both mixable local/remote.

| | Default (local) | Remote alternative |
|---|---|---|
| **LLM** | `gemma3:4b` | `google/gemma-3-4b-it` (OpenRouter) |
| **Embedder** | `qwen3-embedding:4b` | `openai/text-embedding-3-small` @ 1024 |

Both defaults are the winners of their benchmark. Embedder picks in particular: `qwen3-embedding:4b` is the only tested model with FR = EN quality (MRR 0.862 / 0.867) and perfect negation handling; `text-embedding-3-small` @ 1024 is the best hosted result (MRR 0.883) at ~$0.02/1M tokens. Tight on disk? `qwen3-embedding:0.6b` (639 MB, MRR 0.809) is a drop-in lighter fallback.

Reports: [LLM extraction benchmark](.docs/benchmark/2026-04-11-extraction-benchmark-v2.md) · [Embedding benchmark](.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md).

> ⚠️ Switching embedders on an existing install means re-ingesting — two embedders don't share a cosine space. Wipe the Neo4j volume and re-run `/onboard`.

## Built on

[Graphiti](https://github.com/getzep/graphiti) · [FastMCP](https://github.com/jlowin/fastmcp) · [Neo4j](https://neo4j.com)

## License

Apache-2.0 — see [LICENSE](LICENSE). Contributions welcome — open an issue first to discuss.
