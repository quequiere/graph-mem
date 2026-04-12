# Design: graph-mem as a Claude Code native plugin

**Date:** 2026-04-12
**Status:** approved
**Goal:** A new user installs graph-mem with 3 commands — no git clone, no manual JSON editing.

## Target user flow

```bash
# 1. Download compose + configure API key
curl -sL https://raw.githubusercontent.com/quequiere/graph-mem/main/docker-compose.prod.yml -o docker-compose.yml
echo "OPENROUTER_API_KEY=sk-..." > .env
docker compose up -d

# 2. In Claude Code
/plugin marketplace add quequiere/claude-plugins
/plugin install graph-mem
# → MCP server + hooks auto-configured
```

## What changes

### 1. Plugin manifest files (new)

```
graph-mem/
├── .claude-plugin/
│   └── plugin.json          # Plugin metadata
├── .mcp.json                # MCP server declaration
└── hooks/
    └── hooks.json           # SessionStart + Stop hooks
```

**`.claude-plugin/plugin.json`**

```json
{
  "name": "graph-mem",
  "description": "Persistent memory using knowledge graphs",
  "version": "0.1.0",
  "author": { "name": "quequiere" },
  "repository": "https://github.com/quequiere/graph-mem",
  "license": "Apache-2.0"
}
```

**`.mcp.json`**

```json
{
  "mcpServers": {
    "graph-mem": {
      "command": "uvx",
      "args": ["graph-mem"],
      "env": {
        "GRAPHITI_URL": "http://127.0.0.1:8000"
      }
    }
  }
}
```

**`hooks/hooks.json`**

```json
{
  "hooks": {
    "SessionStart": [
      { "command": "graph-mem-session-start" }
    ],
    "Stop": [
      { "command": "graph-mem-session-end", "blocking": true }
    ]
  }
}
```

### 2. Published Docker image (new)

The patched Graphiti image (`graphiti/Dockerfile` + `zep_graphiti.py` + `ingest.py`) is built and pushed to `ghcr.io/quequiere/graph-mem-graphiti` via GitHub Actions on push to main.

### 3. Production compose file (new)

A `docker-compose.prod.yml` at the repo root that references the published image instead of `build: ./graphiti`. Contains Neo4j + Graphiti + Ollama (optional profile). This is the file end users download — no source code needed.

### 4. Marketplace repo (new)

A separate repo `quequiere/claude-plugins` with a `marketplace.json` that lists graph-mem. This lets users do `/plugin marketplace add quequiere/claude-plugins` then `/plugin install graph-mem`.

### 5. GitHub Actions CI (new)

- **Trigger:** push to `main`
- **Jobs:**
  - Build and push `ghcr.io/quequiere/graph-mem-graphiti:latest` (+ version tag)
  - Run unit tests
  - (Future: run integration tests)

## What does NOT change

- `src/graph_mem/` — MCP server code, tools, hooks Python code
- Entry points (`graph-mem`, `graph-mem-session-start`, `graph-mem-session-end`)
- Root `docker-compose.yml` — still used for local development (with `build:`)
- `.env.example` — still the reference for all env vars
- Unit and integration tests

## Key decisions

| Decision | Rationale |
|----------|-----------|
| GHCR over Docker Hub | Free for public repos, tied to GitHub account, no rate limits for authenticated pulls |
| Separate `docker-compose.prod.yml` | Keep dev compose (with `build:`) and prod compose (with `image:`) independent |
| Separate marketplace repo | Decouples plugin registry from plugin source — can host multiple plugins later |
| `uvx` as MCP command | Already the pattern in the current README — installs from PyPI on-the-fly, no pip needed |

## Prerequisites for the user

- Docker & Docker Compose
- An OpenRouter API key (or local Ollama if using the ollama profile)
- Claude Code installed

## Open questions

None — design approved during brainstorming session.
