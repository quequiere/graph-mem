# Design — Rework `.env` / `docker-compose` for local & remote modes

**Date:** 2026-04-11
**Status:** Design approved, pending user review
**Scope:** `.env.example`, `docker-compose.yml`, `graphiti/zep_graphiti.py`, `README.md`

## Problem

The current setup tightly couples LLM and embedder configuration through a single
`OPENAI_API_KEY` / `OPENAI_BASE_URL` pair, and the only documented mode is "remote
via OpenRouter". This blocks two common use cases:

1. **Offline / privacy-first users** who want everything running on their machine
   with no external API calls.
2. **Hybrid users** who want a remote LLM (quality) but a local embedder (cost,
   latency, or vice-versa).

The onboarding story in `README.md` also starts from the assumption that an
OpenAI-compatible API key is required, which contradicts the project's
privacy-first positioning.

## Goals

- Make "full local" the **default** mode — `docker compose up -d` after cloning
  must just work, with no API key required.
- Decouple LLM and embedder configuration so each can point at a different
  provider independently.
- Keep the switch between modes simple: one file to edit (`.env`), one command
  to run (`docker compose up -d`).
- Bundle Ollama inside the Docker stack so users don't need to install anything
  extra for the default path.
- Rewrite the README onboarding around the new default, keep the "Other modes"
  section short and discoverable without cluttering the happy path.

## Non-goals

- Benchmarking or picking a final local embedding model. `nomic-embed-text` is
  a provisional default; the real choice will come from the benchmark work
  tracked separately in `.docs/benchmark/`.
- Supporting providers beyond "any OpenAI-compatible endpoint". Ollama,
  OpenRouter, OpenAI, LocalAI, vLLM, etc. all share that contract.
- Automating migration from the old single-pair `.env` to the new two-pair
  layout. Users edit `.env` themselves — this is a pre-1.0 tool.
- Changes to the MCP server itself (`src/graph_mem/`). This rework is entirely
  on the backend stack.

## Architecture

### Supported modes (2 common + 2 advanced)

| Mode | LLM | Embedder | `COMPOSE_PROFILES` |
|---|---|---|---|
| Full local (default) | Ollama `gemma3:4b` | Ollama `nomic-embed-text` | `ollama` |
| Full remote | OpenRouter `google/gemma-3-4b-it` | OpenRouter `qwen/qwen3-embedding-8b` | *(empty)* |
| Remote LLM + local embedder | OpenRouter | Ollama | `ollama` |
| Local LLM + remote embedder | Ollama | OpenRouter | `ollama` |

The switch between modes is done entirely by editing `.env`. The command to
launch the stack is always the same: `docker compose up -d`.

### Environment variable layout

Two independent triplets, no fallback logic:

```
# LLM
OPENAI_API_KEY
OPENAI_BASE_URL
MODEL_NAME

# Embedder
EMBEDDING_API_KEY
EMBEDDING_BASE_URL
EMBEDDING_MODEL_NAME
```

Naming rationale: keeping `OPENAI_*` for the LLM preserves compatibility with
Graphiti's stock settings object (which expects those exact names) and minimizes
the patch surface in `zep_graphiti.py`. The embedder triplet uses `EMBEDDING_*`
to make the decoupling explicit.

### Docker Compose layout

```
┌─ Always active ───────┐   ┌─ profile: [ollama] ──────┐
│ neo4j                  │   │ ollama                    │
│ graphiti               │   │ ollama-init (pulls models)│
└───────────────────────┘   └───────────────────────────┘
```

- `neo4j` and `graphiti` have no profile → always start.
- `ollama` and `ollama-init` live behind `profiles: [ollama]` → start only
  when the `ollama` profile is active.
- The default `.env` sets `COMPOSE_PROFILES=ollama`, so a clean clone gets the
  full local stack out of the box.
- To disable the bundled Ollama (full remote, or user has Ollama on the host
  already), set `COMPOSE_PROFILES=` in `.env`.

### `ollama-init` service

A one-shot init container that pulls both models at first boot and then exits:

```yaml
ollama-init:
  image: ollama/ollama:latest
  profiles: [ollama]
  depends_on:
    ollama: { condition: service_healthy }
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
```

`graphiti` declares `depends_on: ollama-init: { condition: service_completed_successfully, required: false }`.
The `required: false` flag (Compose spec v2.20+) makes the dependency a no-op
when the `ollama` profile is disabled, so `graphiti` still starts cleanly in
full-remote mode.

The `ollama_data` named volume persists pulled models across restarts, so the
first boot takes ~5 minutes (models download ~3.5 GB total) and subsequent
boots are instant.

## Components & files

### `.env.example`

Layout: four commented presets (A–D, one per mode) followed by the active
configuration section that ships with Preset A (full local) applied.

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

### `docker-compose.yml`

```yaml
services:
  neo4j:
    image: neo4j:5.26.0
    ports: ["7474:7474", "7687:7687"]
    volumes: [neo4j_data:/data]
    environment:
      - NEO4J_AUTH=neo4j/${NEO4J_PASSWORD:-graphiti}
    healthcheck:
      test: ["CMD-SHELL", "neo4j status || exit 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  graphiti:
    build: { context: ./graphiti, dockerfile: Dockerfile }
    ports: ["8000:8000"]
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
    ports: ["11434:11434"]
    volumes: [ollama_data:/root/.ollama]
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

Notes:

- `OPENAI_*` and `EMBEDDING_*` have no default value in the compose interpolation,
  so a missing variable in `.env` produces an explicit Compose error at startup —
  intentional, avoids silent misconfiguration.
- Requires Docker Compose v2.20+ for `depends_on.required`. This is called out
  in the README prerequisites section.

### `graphiti/zep_graphiti.py`

Only `get_graphiti()` changes (lines 187–217). The rest of the file
(`ExampleLLMClient`, `ZepGraphiti` class, schema-to-example helpers) is
untouched.

```python
import os

async def get_graphiti(settings: ZepEnvDep):
    # LLM config (from settings — stock Graphiti path)
    llm_api_key = settings.openai_api_key
    llm_base_url = settings.openai_base_url
    llm_model = settings.model_name

    # Embedder config (decoupled — read directly from env, no fallback)
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

Behavior when `EMBEDDING_*` vars are missing: `os.environ[...]` raises
`KeyError`, which surfaces as a loud startup error in the Graphiti container
logs. No silent fallback. This is the intended behavior — it forces the user
to have a valid `.env`.

### `README.md`

Two sections rewritten:

**`## Quick start`** — single happy path, ~15 lines:

```markdown
## Quick start

**Prerequisites:** Python 3.11+, Docker & Docker Compose v2.20+.

By default, graph-mem runs **fully local** — no API keys required, no data
leaves your machine.

    git clone https://github.com/quequiere/graph-mem && cd graph-mem
    cp .env.example .env
    docker compose up -d

First launch downloads two Ollama models (~3.5 GB). Subsequent launches are
instant thanks to the persistent volume. Neo4j needs ~2 GB RAM, Ollama
needs ~4 GB.

Then connect your MCP client (see next section).
```

**`## Other modes`** — 3–4 lines, pointing at `.env.example`:

```markdown
## Other modes

Need a remote LLM, a local embedder, or a mix of both? Four presets are
documented in `.env.example` — full local, full remote, and two mixed
configurations. Copy the preset you want into the "Active configuration"
block.

**The command is always the same**: `docker compose up -d`
```

**`## Privacy`** block (currently line 33) is updated to reflect that the
default is fully local:

> **Privacy:** By default, everything runs on your machine — no data leaves it.
> If you configure a remote provider in `.env`, session summaries and saved
> facts will be sent to that provider during entity extraction. Review what
> graph-mem stores before enabling remote mode on sensitive work projects.

**`## Architecture`** ASCII diagram is updated to show the optional Ollama
service alongside Graphiti and Neo4j in the Docker box.

## Testing

Manual verification for each mode:

1. **Full local (Preset A)** — clean clone, `cp .env.example .env`,
   `docker compose up -d`. Verify: `ollama` and `ollama-init` start, models
   download on first boot, Graphiti becomes healthy, `save_memory` /
   `search_memory` work end-to-end through the MCP client.

2. **Full remote (Preset B)** — copy Preset B values, set real OpenRouter key,
   `docker compose up -d`. Verify: no `ollama*` containers start, Graphiti
   connects to OpenRouter, end-to-end works.

3. **Preset C (remote LLM + local embedder)** — same exercise. Verify the
   embedder hits the local Ollama (`http://ollama:11434/v1`) while the LLM
   calls OpenRouter. Check the Graphiti logs for both endpoints being contacted.

4. **Preset D (local LLM + remote embedder)** — same exercise, mirror of C.

5. **Missing `EMBEDDING_*` vars** — delete them from `.env`, `docker compose up`.
   Verify Graphiti fails loudly with a `KeyError: 'EMBEDDING_API_KEY'` in the
   logs (not a silent fallback to `OPENAI_*`).

6. **Ollama on host (bonus)** — stop the Docker Ollama, install Ollama on the
   host, set `COMPOSE_PROFILES=` and `OPENAI_BASE_URL=http://host.docker.internal:11434/v1`,
   verify end-to-end still works.

No automated tests are added for this rework — the existing integration tests
in `tests/integration/` cover the Graphiti API surface, which is unchanged.
The rework is entirely in configuration and orchestration; the unit of change
is "does the stack come up and process a memory round-trip", which is a manual
smoke test.

## Risks & open questions

- **`required: false` compatibility** — requires Docker Compose v2.20 (March
  2024). Users on older versions will need to upgrade. Mitigation: call out
  the version in README prerequisites. Fallback if needed: split into
  `docker-compose.yml` + `docker-compose.local.yml` with an explicit
  `-f docker-compose.local.yml` override — but this changes the
  "always the same command" property, so we accept the Compose v2.20 floor.

- **`nomic-embed-text` is a provisional pick** — no benchmark yet for local
  embedding quality with our extraction pipeline. Tracked for follow-up,
  not a blocker for this rework (easy swap via `.env`).

- **Disk space on first boot** — ~3.5 GB download can surprise users on
  limited bandwidth. Mitigation: explicit mention in README and in the
  `ollama-init` log output during the pull.

- **`gemma3:4b` quality for entity extraction** — this is a small model
  compared to the current `gemma-4-26b-a4b-it`. Extraction quality may drop
  noticeably. If it does, the fix is a `.env` change, not a code change, so
  we still ship this rework and iterate on the model choice separately.

- **Cross-encoder follows LLM** — documented trade-off: in "local LLM + remote
  embedder" mode, reranking stays local (cheap, maybe lower quality). In
  "remote LLM + local embedder" mode, reranking is remote (more API cost).
  Accepted because the cross-encoder uses generation, not embeddings, so
  pairing it with the LLM is the semantically correct choice.
