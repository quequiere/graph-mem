# `.env` / `docker-compose` Rework — Local & Remote Modes — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make graph-mem run fully-local by default (Ollama in Docker) with zero API keys required, while keeping remote and mixed LLM/embedder modes as simple `.env` edits away.

**Architecture:** Decouple LLM and embedder env vars into two independent triplets (`OPENAI_*` + `EMBEDDING_*`). Bundle Ollama + a one-shot `ollama-init` service behind a Docker Compose profile (`ollama`) that ships active by default via `COMPOSE_PROFILES` in `.env`. Patch `graphiti/zep_graphiti.py` so `get_graphiti()` reads embedder config directly from the environment. Rewrite the README around the new default happy path and push mode-switching intelligence into `.env.example` via copy-paste presets.

**Tech Stack:** Docker Compose v2.20+, Ollama, Graphiti (patched), Neo4j 5.26, Python 3.11+.

**Reference spec:** `.docs/superpowers/specs/2026-04-11-env-docker-modes-design.md`

**TDD note:** This rework is entirely configuration and orchestration — there is no new business logic to unit-test. Each task uses explicit smoke-test commands as its verification step instead of a Red/Green unit-test cycle. The one code change (Task 3 patch to `zep_graphiti.py`) is validated by the end-to-end smoke tests in Tasks 5–8 that actually exercise the code path in a live container.

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `.env.example` | Full rewrite | Four commented presets (A–D) + active configuration block |
| `docker-compose.yml` | Full rewrite | `neo4j` + `graphiti` always-on; `ollama` + `ollama-init` behind `profiles: [ollama]` |
| `graphiti/zep_graphiti.py` | Patch `get_graphiti()` only (lines 187–217), plus `import os` | Read `EMBEDDING_*` vars independently from LLM vars |
| `README.md` | Update `Quick start`, `Other modes`, `Privacy`, `Architecture` sections | Reflect local-first onboarding |
| `.docs/superpowers/specs/2026-04-11-env-docker-modes-design.md` | Already exists | Reference spec (do not edit during implementation) |

---

## Task 1: Rewrite `.env.example` with presets + active config

**Why first:** Independent of all other changes. Lets subsequent tasks refer to exact variable names.

**Files:**
- Modify: `.env.example` (full replacement)

- [ ] **Step 1: Replace the entire `.env.example` file**

Write this exact content:

```bash
# ============================================================
#  graph-mem configuration
#
#  Default: everything runs locally in Docker (Ollama).
#  To switch modes, copy values from a preset below into the
#  "Active configuration" section at the bottom.
#
#  Workflow (any mode):  docker compose up -d
# ============================================================


# --- Preset A: Full local (default) ------------------------
# OPENAI_API_KEY=sk-dummy
# OPENAI_BASE_URL=http://ollama:11434/v1
# MODEL_NAME=gemma3:4b
# EMBEDDING_API_KEY=sk-dummy
# EMBEDDING_BASE_URL=http://ollama:11434/v1
# EMBEDDING_MODEL_NAME=nomic-embed-text
# COMPOSE_PROFILES=ollama


# --- Preset B: Full remote (OpenRouter) --------------------
# OPENAI_API_KEY=sk-or-v1-...
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# MODEL_NAME=google/gemma-3-4b-it
# EMBEDDING_API_KEY=sk-or-v1-...
# EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
# EMBEDDING_MODEL_NAME=qwen/qwen3-embedding-8b
# COMPOSE_PROFILES=


# --- Preset C: Remote LLM + local embedder -----------------
# OPENAI_API_KEY=sk-or-v1-...
# OPENAI_BASE_URL=https://openrouter.ai/api/v1
# MODEL_NAME=google/gemma-3-4b-it
# EMBEDDING_API_KEY=sk-dummy
# EMBEDDING_BASE_URL=http://ollama:11434/v1
# EMBEDDING_MODEL_NAME=nomic-embed-text
# COMPOSE_PROFILES=ollama


# --- Preset D: Local LLM + remote embedder -----------------
# OPENAI_API_KEY=sk-dummy
# OPENAI_BASE_URL=http://ollama:11434/v1
# MODEL_NAME=gemma3:4b
# EMBEDDING_API_KEY=sk-or-v1-...
# EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
# EMBEDDING_MODEL_NAME=qwen/qwen3-embedding-8b
# COMPOSE_PROFILES=ollama


# ============================================================
#  Active configuration  (Preset A — full local)
# ============================================================
OPENAI_API_KEY=sk-dummy
OPENAI_BASE_URL=http://ollama:11434/v1
MODEL_NAME=gemma3:4b

EMBEDDING_API_KEY=sk-dummy
EMBEDDING_BASE_URL=http://ollama:11434/v1
EMBEDDING_MODEL_NAME=nomic-embed-text

NEO4J_PASSWORD=graphiti
COMPOSE_PROFILES=ollama
```

- [ ] **Step 2: Verify the file parses as a valid dotenv**

Run:

```bash
docker run --rm -v "$PWD/.env.example:/tmp/.env.example:ro" alpine sh -c "set -a && . /tmp/.env.example && set +a && echo OPENAI_BASE_URL=\$OPENAI_BASE_URL && echo COMPOSE_PROFILES=\$COMPOSE_PROFILES"
```

Expected output:

```
OPENAI_BASE_URL=http://ollama:11434/v1
COMPOSE_PROFILES=ollama
```

If the sourcing fails, there's a syntax error — fix the line it points to.

- [ ] **Step 3: Commit**

```bash
git add .env.example
git commit -m "rework .env.example with per-mode presets and decoupled LLM/embedder vars"
```

---

## Task 2: Patch `graphiti/zep_graphiti.py` to decouple embedder config

**Why second:** The Graphiti image needs rebuilding with the new code before the compose rewrite in Task 3 can be validated.

**Files:**
- Modify: `graphiti/zep_graphiti.py` (add `import os`, rewrite `get_graphiti()` body, lines 187–217)

- [ ] **Step 1: Add `import os` to the top-of-file imports**

Find the import block near lines 9–25 and add `import os` right under the existing `import json`, `import logging`, `import typing` stdlib imports:

```python
import json
import logging
import os
import typing
from typing import Annotated
```

- [ ] **Step 2: Rewrite the body of `get_graphiti()`**

Locate `async def get_graphiti(settings: ZepEnvDep):` (currently line 187). Replace the body (everything from `base_url = settings.openai_base_url` down to the `try / yield client / finally / await client.close()` block) with:

```python
async def get_graphiti(settings: ZepEnvDep):
    # LLM config (from settings — stock Graphiti path)
    llm_api_key = settings.openai_api_key
    llm_base_url = settings.openai_base_url
    llm_model = settings.model_name

    # Embedder config (decoupled — read directly from env, no fallback).
    # KeyError here is intentional: missing EMBEDDING_* vars must fail loud
    # at startup rather than silently reusing the LLM provider.
    embed_api_key = os.environ["EMBEDDING_API_KEY"]
    embed_base_url = os.environ["EMBEDDING_BASE_URL"]
    embed_model = os.environ["EMBEDDING_MODEL_NAME"]

    llm_config = LLMConfig(
        api_key=llm_api_key,
        model=llm_model,
        small_model=llm_model,
        base_url=llm_base_url,
    )
    llm_client = ExampleLLMClient(config=llm_config)

    embedder = OpenAIEmbedder(config=OpenAIEmbedderConfig(
        api_key=embed_api_key,
        embedding_model=embed_model,
        embedding_dim=EMBEDDING_DIM,
        base_url=embed_base_url,
    ))

    # Cross-encoder stays coupled to the LLM client: it reranks via text
    # generation, not embeddings, so it's consistent for it to follow the
    # LLM mode (remote or local).
    cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)

    client = ZepGraphiti(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        yield client
    finally:
        await client.close()
```

- [ ] **Step 3: Static sanity check — file still compiles**

Run:

```bash
python -m py_compile graphiti/zep_graphiti.py
```

Expected: no output, exit code 0. If you see a `SyntaxError`, the edit broke indentation — fix it before continuing.

- [ ] **Step 4: Commit**

```bash
git add graphiti/zep_graphiti.py
git commit -m "decouple embedder config from LLM in graphiti patch"
```

---

## Task 3: Rewrite `docker-compose.yml` with profiles and `ollama-init`

**Why third:** Consumes the new variable names from Task 1 and the patched image source from Task 2.

**Files:**
- Modify: `docker-compose.yml` (full replacement)

- [ ] **Step 1: Replace the entire `docker-compose.yml`**

Write this exact content:

```yaml
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
    build:
      context: ./graphiti
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      # LLM
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENAI_BASE_URL=${OPENAI_BASE_URL}
      - MODEL_NAME=${MODEL_NAME}
      # Embedder (decoupled)
      - EMBEDDING_API_KEY=${EMBEDDING_API_KEY}
      - EMBEDDING_BASE_URL=${EMBEDDING_BASE_URL}
      - EMBEDDING_MODEL_NAME=${EMBEDDING_MODEL_NAME}
      # Neo4j
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

- [ ] **Step 2: Validate the compose file syntax without starting anything**

First, ensure `.env` exists (copy from the example if not):

```bash
[ -f .env ] || cp .env.example .env
```

Then:

```bash
docker compose config --quiet
```

Expected: no output, exit code 0. Any warning or error points at a variable that's missing or a YAML indentation problem.

- [ ] **Step 3: Check which services are selected per profile**

Run with `ollama` profile active (default via `.env`):

```bash
docker compose config --services
```

Expected output (order may vary):

```
graphiti
neo4j
ollama
ollama-init
```

Run without the profile:

```bash
COMPOSE_PROFILES= docker compose config --services
```

Expected output:

```
graphiti
neo4j
```

- [ ] **Step 4: Commit**

```bash
git add docker-compose.yml
git commit -m "docker-compose: bundle ollama behind a profile, wire decoupled embedder vars"
```

---

## Task 4: Smoke test — Full local (Preset A, default)

**Why:** First end-to-end validation that the default path works. Tasks 1–3 are prerequisites.

**Files:**
- None (runtime validation only)

- [ ] **Step 1: Ensure `.env` matches Preset A (the shipped default)**

```bash
cp .env.example .env
grep '^COMPOSE_PROFILES=' .env
```

Expected: `COMPOSE_PROFILES=ollama`

- [ ] **Step 2: Build + start the stack**

```bash
docker compose up -d --build
```

Expected: `neo4j`, `graphiti`, `ollama`, `ollama-init` all created. `ollama-init` starts pulling models — this takes 3–10 minutes on first run depending on connection.

- [ ] **Step 3: Wait for `ollama-init` to complete**

```bash
docker compose logs -f ollama-init
```

Wait for the line `All models ready.` Then Ctrl-C.

Expected: the container exits with code 0. Verify:

```bash
docker compose ps ollama-init
```

Should show `exited (0)`.

- [ ] **Step 4: Verify Graphiti became healthy**

```bash
curl -s http://127.0.0.1:8000/healthcheck
```

Expected: a JSON `{"status": "ok"}` (or equivalent — exact response depends on Graphiti version). Non-empty, not a connection error.

Also check its logs:

```bash
docker compose logs graphiti --tail 30
```

Expected: no Python traceback, no `KeyError`, no mentions of OpenAI authentication failures.

- [ ] **Step 5: End-to-end memory round-trip via the Graphiti REST API**

Add an episode and search it back:

```bash
curl -s -X POST http://127.0.0.1:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "smoketest_local",
    "messages": [{
      "content": "graph-mem default mode test: the capital of France is Paris",
      "role_type": "user",
      "name": "smoketest",
      "timestamp": "2026-04-11T12:00:00Z"
    }]
  }'
```

Expected: a 202 Accepted or similar success response. Wait ~20 seconds for async ingestion.

Then search:

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "capital of France", "group_ids": ["smoketest_local"], "max_facts": 5}'
```

Expected: at least one fact in the response that references Paris. If the facts array is empty, something in the extraction pipeline broke — check `docker compose logs graphiti` for the actual error.

- [ ] **Step 6: Leave the stack running if all checks pass; move to Task 5**

No commit for this task — it's a verification-only task.

---

## Task 5: Smoke test — Full remote (Preset B)

**Files:**
- None (runtime validation only)

**Precondition:** you need a valid OpenRouter API key. Export it before starting: `export OPENROUTER_KEY=sk-or-v1-...`. If you don't have one, skip this task and document it as "deferred pending key".

- [ ] **Step 1: Stop the current stack and switch `.env` to Preset B**

```bash
docker compose down
```

Edit `.env`, replacing the active configuration block with Preset B values:

```bash
OPENAI_API_KEY=$OPENROUTER_KEY
OPENAI_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=google/gemma-3-4b-it

EMBEDDING_API_KEY=$OPENROUTER_KEY
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_MODEL_NAME=qwen/qwen3-embedding-8b

NEO4J_PASSWORD=graphiti
COMPOSE_PROFILES=
```

(Substitute the real key value for `$OPENROUTER_KEY` — dotenv doesn't expand shell vars.)

- [ ] **Step 2: Start the stack and verify no Ollama service launches**

```bash
docker compose up -d
docker compose ps
```

Expected: only `neo4j` and `graphiti` are listed. No `ollama`, no `ollama-init`.

- [ ] **Step 3: Verify Graphiti comes up and can reach OpenRouter**

```bash
sleep 5
curl -s http://127.0.0.1:8000/healthcheck
docker compose logs graphiti --tail 30
```

Expected: healthcheck returns OK; no KeyError on EMBEDDING_*; no "connection refused" to `ollama:11434`.

- [ ] **Step 4: End-to-end round-trip on the remote stack**

Repeat Task 4 Step 5 but with `group_id: "smoketest_remote"`. Verify the search returns a fact mentioning Paris.

- [ ] **Step 5: No commit. Tear down and move on.**

```bash
docker compose down
```

---

## Task 6: Smoke test — Preset C (remote LLM + local embedder)

**Files:**
- None (runtime validation only)

**Precondition:** `$OPENROUTER_KEY` is set.

- [ ] **Step 1: Switch `.env` to Preset C values**

Edit `.env` to match Preset C exactly (see Task 1 for content). Keep `COMPOSE_PROFILES=ollama`.

- [ ] **Step 2: Start the stack**

```bash
docker compose up -d
docker compose ps
```

Expected: `neo4j`, `graphiti`, `ollama`, `ollama-init` all present. `ollama-init` may finish instantly if `nomic-embed-text` was already pulled in Task 4.

- [ ] **Step 3: Verify the embedder talks to the local Ollama while the LLM talks to OpenRouter**

Tail graphiti logs while a request is in flight:

```bash
docker compose logs -f graphiti &
curl -s -X POST http://127.0.0.1:8000/messages \
  -H "Content-Type: application/json" \
  -d '{
    "group_id": "smoketest_mixed_c",
    "messages": [{
      "content": "mixed mode test: Rust is a systems programming language",
      "role_type": "user",
      "name": "smoketest",
      "timestamp": "2026-04-11T12:00:00Z"
    }]
  }'
sleep 20
# Ctrl-C the background logs -f
```

Expected in the logs: HTTP calls to both `ollama:11434` (embeddings) and `openrouter.ai` (generation). The exact log format depends on Graphiti, but both hosts should be reachable and neither should error.

- [ ] **Step 4: Round-trip search**

```bash
curl -s -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "programming language", "group_ids": ["smoketest_mixed_c"], "max_facts": 5}'
```

Expected: a fact mentioning Rust.

- [ ] **Step 5: No commit. Tear down.**

```bash
docker compose down
```

---

## Task 7: Smoke test — Missing `EMBEDDING_*` vars fail loudly

**Files:**
- None (runtime validation only)

**Why:** Verifies the "no silent fallback" contract from the spec. This is the regression that the Task 2 `KeyError` is supposed to catch.

- [ ] **Step 1: Write an intentionally broken `.env`**

```bash
cp .env.example .env.broken
sed -i '/^EMBEDDING_/d' .env.broken
mv .env.broken .env
cat .env | grep -c '^EMBEDDING_'
```

Expected last line: `0` (no `EMBEDDING_*` lines remain).

- [ ] **Step 2: Try to start Graphiti**

```bash
docker compose up -d graphiti
sleep 5
docker compose logs graphiti --tail 30
```

Expected: Graphiti container is **not** running (`docker compose ps graphiti` shows it exited or restarting). The logs must include a `KeyError` pointing at `'EMBEDDING_API_KEY'` (or `EMBEDDING_BASE_URL` / `EMBEDDING_MODEL_NAME` — whichever is checked first). This is the correct behavior.

If instead Graphiti boots successfully, the patch from Task 2 didn't take effect — go check the rebuilt image and the `get_graphiti()` code.

- [ ] **Step 3: Restore the good `.env` and tear down**

```bash
cp .env.example .env
docker compose down
```

- [ ] **Step 4: No commit.**

---

## Task 8: Rewrite `README.md` — Quick start & Other modes sections

**Why:** Final, user-facing piece. All the backend pieces are already validated by Tasks 4–7.

**Files:**
- Modify: `README.md` (replace the `## Quick start` section and add a new `## Other modes` section)

- [ ] **Step 1: Read the current `## Quick start` section to know what lines to replace**

Run:

```bash
grep -n '^## ' README.md
```

Note the line numbers for `## Quick start` and the next `##` heading that follows it (currently `## What gets stored?`). You'll replace everything between them.

- [ ] **Step 2: Replace the `## Quick start` section with the new content**

Replace the entire block starting at `## Quick start` (inclusive) and ending at the line before `## What gets stored?` with:

```markdown
## Quick start

**Prerequisites:** Python 3.11+, Docker & Docker Compose v2.20+.

By default, graph-mem runs **fully local** — no API keys required, no data
leaves your machine. Ollama, Graphiti and Neo4j all run in Docker.

### 1. Start the backend

```bash
git clone https://github.com/quequiere/graph-mem && cd graph-mem
cp .env.example .env
docker compose up -d
```

First launch downloads two Ollama models (~3.5 GB). Subsequent launches are
instant thanks to the persistent volume. Neo4j needs ~2 GB RAM, Ollama
needs ~4 GB.

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

Add to `~/.claude/settings.json` for automatic context injection at session start
and auto-save at stop:

```json
{
  "hooks": {
    "SessionStart": [{ "command": "graph-mem-session-start" }],
    "Stop": [{ "command": "graph-mem-session-end", "blocking": true }]
  }
}
```

`graph-mem-session-start` and `graph-mem-session-end` are CLI commands installed
with `pip install graph-mem`. The `Stop` hook has a 30-second timeout — if
Graphiti is unreachable, it exits with a warning and your session ends normally
(no data loss; re-save manually via `save_session` next time). Without hooks,
call `get_context` and `save_session` manually from the chat.

## Other modes

Need a remote LLM, a local embedder, or a mix of both? Four presets are
documented in [`.env.example`](.env.example) — full local, full remote, and
two mixed configurations. Copy the preset you want into the "Active
configuration" block at the bottom of your `.env`.

**The command is always the same**: `docker compose up -d`
```

(Note: the old `Phase 1 — Start the backend` / `Phase 2 — Connect your client`
structure, and the `env` block with `OPENAI_API_KEY`, are both removed. The
new step 2 has `GRAPHITI_URL` but no API key.)

- [ ] **Step 3: Verify the replaced section renders correctly**

```bash
grep -n '^## ' README.md
```

Expected: `## Quick start`, then `## Other modes`, then `## What gets stored?`
appearing in that order. No duplicate headings, no `Phase 1` / `Phase 2` left
over.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "README: rewrite Quick start around local-first default, add Other modes section"
```

---

## Task 9: Update `README.md` — Privacy note & Architecture diagram

**Why:** These two sections still describe the old remote-only world and will contradict the new Quick start if not updated.

**Files:**
- Modify: `README.md` (Privacy block around the current line 33; Architecture ASCII diagram around current lines 138–149)

- [ ] **Step 1: Replace the Privacy block**

Find the existing `> **Privacy:** ...` blockquote (currently ~line 33, right after "Why not just use CLAUDE.md or mem0?"). Replace it with:

```markdown
> **Privacy:** By default, everything runs on your machine — Ollama, Graphiti,
> and Neo4j all live in Docker, and no data leaves your host. If you configure
> a remote provider in `.env` (OpenRouter, OpenAI, etc.), session summaries and
> saved facts will be sent to that provider during entity extraction. Review
> what graph-mem stores before enabling a remote mode on sensitive work projects.
```

- [ ] **Step 2: Update the Architecture ASCII diagram**

Find the `## Architecture` section and the ASCII diagram that follows. Replace the diagram with:

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

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "README: update Privacy note and Architecture diagram for local-first default"
```

---

## Task 10: Final sanity check across all four presets

**Why:** Everything was tested in isolation. This is the final "does the whole story still hang together" pass.

**Files:**
- None (verification only)

- [ ] **Step 1: Confirm the committed `.env.example` still boots Preset A cleanly**

```bash
docker compose down
cp .env.example .env
docker compose up -d
sleep 10
curl -s http://127.0.0.1:8000/healthcheck
docker compose ps
```

Expected: healthcheck OK, all four services listed (`neo4j`, `graphiti`, `ollama`, `ollama-init`).

- [ ] **Step 2: Read the README Quick start section out loud (mental exercise)**

Verify:
- A new user could follow the three numbered steps without prior context
- No mention of `OPENAI_API_KEY` as a required prerequisite
- `## Other modes` section points at `.env.example` and says "command is always the same"

- [ ] **Step 3: Show the final `git log` for the branch**

```bash
git log --oneline main..HEAD
```

Expected: one commit per Task 1, 2, 3, 8, 9 (and any additional commit for Task 4 or Task 7 if the user chose to commit verification notes, which this plan does not). Clean, in order.

- [ ] **Step 4: Tear down**

```bash
docker compose down
```

- [ ] **Step 5: (Optional) Restore the graph-mem hook / MCP config to the new default URL**

If your local `~/.claude` settings still reference old env vars like `OPENAI_API_KEY` in the graph-mem MCP block, update them to match the new README. This is outside the repo but worth verifying that the end-to-end story you just documented actually works from a fresh Claude Code session.

---

## Self-Review Notes

**Spec coverage:**
- ✅ Full local as default → Tasks 1, 3, 4
- ✅ Decoupled LLM/embedder env vars → Tasks 1, 2, 3
- ✅ Ollama bundled in Docker via profile → Task 3
- ✅ `ollama-init` with auto-pull → Task 3 + Task 4 step 3
- ✅ Patch `get_graphiti()` → Task 2
- ✅ No silent fallback when `EMBEDDING_*` missing → Task 7
- ✅ `docker compose up -d` is always the command → demonstrated in Tasks 4, 5, 6
- ✅ README Quick start rewrite → Task 8
- ✅ README "Other modes" section pointing at `.env.example` → Task 8
- ✅ Privacy note + Architecture diagram updated → Task 9
- ✅ All four presets (A, B, C, D) have a smoke-test path → Tasks 4, 5, 6. Preset D (local LLM + remote embedder) is not explicitly tested; it's symmetric to Preset C and shares the same code path, so this is an accepted gap. Add an ad-hoc test if Task 6 raises any surprise.

**Placeholder scan:** no "TBD" / "TODO" / "fill in details" / "similar to Task N" patterns left in the plan. All code blocks are concrete.

**Type consistency:** variable names (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `MODEL_NAME`, `EMBEDDING_API_KEY`, `EMBEDDING_BASE_URL`, `EMBEDDING_MODEL_NAME`, `COMPOSE_PROFILES`, `NEO4J_PASSWORD`) are identical across `.env.example`, `docker-compose.yml`, and `zep_graphiti.py`. Service names (`neo4j`, `graphiti`, `ollama`, `ollama-init`) match between the compose file and every command that references them.

**TDD note reminder:** the project CLAUDE.md requires either (a) passing through `superpowers:test-driven-development` and `superpowers:subagent-driven-development`, or (b) explicit user confirmation in French to bypass. Because this rework is config/orchestration with no new business logic, unit TDD doesn't apply cleanly; each task substitutes an explicit smoke-test verification. **The executor must get explicit user confirmation (in French) before starting implementation, acknowledging that the classic Red/Green unit-test cycle is replaced by smoke-test verifications here.**
