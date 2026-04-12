# Plugin Install Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make graph-mem installable via `/plugin install` with auto-configured MCP server and hooks, and eliminate `git clone` for end users by publishing the patched Graphiti Docker image.

**Architecture:** Add Claude Code plugin manifest files (`.claude-plugin/plugin.json`, `.mcp.json`, `hooks/hooks.json`), create a production `docker-compose.prod.yml` referencing a published GHCR image, add a GitHub Actions workflow to build/push the image, and create a marketplace repo.

**Tech Stack:** Claude Code plugin system, GitHub Actions, GHCR (ghcr.io), Docker

---

## File Map

| Action | File | Purpose |
|--------|------|---------|
| Create | `.claude-plugin/plugin.json` | Plugin manifest |
| Create | `.mcp.json` | MCP server auto-configuration |
| Create | `hooks/hooks.json` | Hooks auto-configuration |
| Create | `docker-compose.prod.yml` | Production compose (uses published image) |
| Create | `.github/workflows/docker.yml` | CI: build + push Graphiti image to GHCR |
| Modify | `README.md` | Update install instructions |

---

### Task 1: Plugin manifest files

**Files:**
- Create: `.claude-plugin/plugin.json`
- Create: `.mcp.json`
- Create: `hooks/hooks.json`

- [ ] **Step 1: Create `.claude-plugin/plugin.json`**

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

- [ ] **Step 2: Create `.mcp.json`**

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

- [ ] **Step 3: Create `hooks/hooks.json`**

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

- [ ] **Step 4: Verify plugin structure**

Run: `ls -la .claude-plugin/ .mcp.json hooks/hooks.json`
Expected: all three files exist

- [ ] **Step 5: Commit**

```bash
git add .claude-plugin/plugin.json .mcp.json hooks/hooks.json
git commit -m "Add Claude Code plugin manifest, MCP config, and hooks"
```

---

### Task 2: Production Docker Compose

**Files:**
- Create: `docker-compose.prod.yml`

- [ ] **Step 1: Create `docker-compose.prod.yml`**

This is a standalone compose file for end users. It references the published GHCR image instead of `build: ./graphiti`. All services and env vars match the dev compose file, but no build context is needed.

```yaml
# graph-mem production stack
# Usage:
#   curl -sL https://raw.githubusercontent.com/quequiere/graph-mem/main/docker-compose.prod.yml -o docker-compose.yml
#   cp .env.example from the repo or create .env with your API keys
#   docker compose up -d

services:
  neo4j:
    image: neo4j:5.26.0
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-graphiti}
    healthcheck:
      test: ["CMD-SHELL", "neo4j status || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  graphiti:
    image: ghcr.io/quequiere/graph-mem-graphiti:latest
    ports:
      - "${GRAPHITI_PORT:-8000}:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
      - MODEL_NAME=${MODEL_NAME}
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
      - EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL}
      - EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=${NEO4J_PASSWORD:-graphiti}
    depends_on:
      neo4j:
        condition: service_healthy
      ollama-init:
        condition: service_completed_successfully
        required: false

  ollama:
    image: ollama/ollama:latest
    profiles: [ollama]
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 5s
      timeout: 5s
      retries: 10

  ollama-init:
    image: ollama/ollama:latest
    profiles: [ollama]
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: /bin/sh
    command: >
      -c "
      echo 'Pulling LLM model: ${MODEL_NAME}...' &&
      OLLAMA_HOST=http://ollama:11434 ollama pull ${MODEL_NAME} &&
      echo 'Pulling embedding model: ${EMBEDDING_MODEL_NAME}...' &&
      OLLAMA_HOST=http://ollama:11434 ollama pull ${EMBEDDING_MODEL_NAME} &&
      echo 'All models ready.'
      "
    restart: "no"

volumes:
  neo4j_data:
  ollama_data:
```

- [ ] **Step 2: Verify compose file parses**

Run: `docker compose -f docker-compose.prod.yml config --quiet`
Expected: no errors (image doesn't need to exist for config validation)

- [ ] **Step 3: Commit**

```bash
git add docker-compose.prod.yml
git commit -m "Add production compose file using published GHCR image"
```

---

### Task 3: GitHub Actions — Docker image build & push

**Files:**
- Create: `.github/workflows/docker.yml`

- [ ] **Step 1: Create `.github/workflows/docker.yml`**

```yaml
name: Build and push Graphiti image

on:
  push:
    branches: [main]
    paths:
      - "graphiti/**"
      - ".github/workflows/docker.yml"

permissions:
  contents: read
  packages: write

jobs:
  build-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@0c366fd6a839edf440554fa01a7085ccba70ac98 # v6.0.2

      - name: Log in to GHCR
        uses: docker/login-action@9780b0c442fbb1117ed29e0efdff1e18412f7567 # v3.3.0
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@4f58ea79222b3b9dc2c8bbdd6debcef730109a75 # v6.9.0
        with:
          context: ./graphiti
          push: true
          tags: |
            ghcr.io/quequiere/graph-mem-graphiti:latest
            ghcr.io/quequiere/graph-mem-graphiti:${{ github.sha }}
```

- [ ] **Step 2: Validate workflow syntax**

Run: `python -c "import yaml; yaml.safe_load(open('.github/workflows/docker.yml'))"`
Expected: no errors

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/docker.yml
git commit -m "Add CI workflow to build and push patched Graphiti image to GHCR"
```

---

### Task 4: Update README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update the installation section**

Replace the current install section with two paths: quick install (production — no clone) and developer install (with clone). The quick install uses `docker-compose.prod.yml` + `/plugin install`. The developer install keeps the current flow for contributors.

Key changes:
- Lead with the quick install path (download compose, docker up, plugin install)
- Move the current clone-based flow under a "Development" or "Contributing" section
- Remove the manual MCP config and hooks config sections — replaced by `/plugin install`
- Keep the hooks section as a fallback for non-plugin users under a `<details>` block

- [ ] **Step 2: Review the updated README end-to-end**

Read the full README to make sure the flow is coherent and no section references the old manual config as the primary path.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "Update README with plugin-based install as primary path"
```

---

### Task 5: Marketplace repo setup

**Files:**
- External repo: `quequiere/claude-plugins`

- [ ] **Step 1: Create the marketplace repo on GitHub**

Create a new public repo `quequiere/claude-plugins` with a single file:

`marketplace.json`:
```json
{
  "source": "anthropics/claude-code",
  "plugins": [
    {
      "id": "graph-mem",
      "name": "Graph Memory",
      "description": "Persistent memory using knowledge graphs — auto-configures MCP server and hooks",
      "version": "0.1.0",
      "source": "https://github.com/quequiere/graph-mem",
      "sourceType": "git"
    }
  ]
}
```

- [ ] **Step 2: Test the full flow**

```bash
# In Claude Code:
/plugin marketplace add quequiere/claude-plugins
/plugin install graph-mem
```

Verify:
- MCP server `graph-mem` appears in `/mcp`
- Hooks SessionStart and Stop are active
- `graph-mem-session-start` runs on next session start

- [ ] **Step 3: Document marketplace setup in README**

Add a one-liner in the quick install section:
```
/plugin marketplace add quequiere/claude-plugins
/plugin install graph-mem
```
