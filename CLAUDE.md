# graph-mem - Project Tracking

## Project Overview

**graph-mem** is a Claude Code plugin that provides persistent, intelligent memory using Graphiti's knowledge graph. Unlike flat memory stores, it leverages entity relationships, temporal awareness, and semantic search to build a rich understanding of the developer over time.

## Current Status

**Working end-to-end** on branch `feat/openrouter-graphiti-patch`. Stack: Neo4j 5.26 + patched Graphiti + OpenRouter (gemma-4-26b + qwen3-embedding-8b).

## Design Spec

Full design document: `.docs/specs/2026-04-05-graph-mem-design.md`

## Architecture

```
Claude Code (stdio) → graph-mem MCP server (Python, local)
                          → Graphiti REST API (Docker, httpx over HTTP)
                              → Neo4j 5.26 (Docker)
                              → OpenRouter API (LLM + embeddings)
```

- **Language**: Python, installed via `pip install -e .`
- **No graphiti_core dependency**: pure REST/HTTP client
- **LLM backend**: OpenRouter (configurable via `.env`)
- **Embeddings**: qwen/qwen3-embedding-8b (4096 native, truncated to 1024 client-side)

## Docker Stack

```bash
docker compose up -d --build   # from project root
```

Requires `.env` with `OPENROUTER_API_KEY`. See `docker-compose.yml`.

**Patched files in `graphiti/`:**
- `zep_graphiti.py` — ExampleLLMClient (fixes schema echoing for Gemma/Qwen/Llama), configures embedder + cross-encoder from env vars
- `ingest.py` — AsyncWorker catches exceptions instead of dying silently (upstream bug)
- `Dockerfile` — extends `zepai/graphiti:latest` with patches

## MCP Tools (current — minimal surface)

Only 3 tools exposed to keep model attention focused:

| Tool | Purpose |
|---|---|
| `__IMPORTANT__graph_mem` | Workflow reminder — tells model how to use graph-mem |
| `save_memory(content, scope)` | Save info. scope="user" or "project" |
| `search_memory(query, scope)` | Search. scope="all", "user", or "project" |

`scope` replaces the old `group_id` parameter — the server computes the correct group_id automatically from cwd's git remote URL. This prevents the model from inventing group_ids.

Other tools (onboard, context, reminders, profile, etc.) still exist in `src/graph_mem/tools/` but are only used internally by hooks.

## Hooks

Configured in `~/.claude/settings.json`:

- **SessionStart** (matcher: `startup|clear|compact`): injects instructions + recalled context (profile + project + reminders). Must complete in <10s — uses `asyncio.gather` to parallelize 4 Graphiti searches (~1.5s).
- **Stop**: saves session summary to knowledge graph via `graph-mem-session-end`

### Key discovery: hooks output is injected as invisible system context
Hook stdout goes "to Claude" (system context), NOT displayed to the user. This is normal Claude Code behavior. The model sees it but the user doesn't.

### Key discovery: hook timeout causes silent cancellation
If the hook takes too long, Claude Code cancels it silently (logged as "cancelled" in debug). Original sequential version took ~6s and was cancelled. Parallelized version takes ~1.5s and works reliably.

**Debug hooks**: `claude --debug` then check `~/.claude/debug/<session-id>.txt`, search for "SessionStart.*success" or "SessionStart.*cancelled".

## Graphiti Patches (important)

### ExampleLLMClient (schema echoing fix)
Many models via OpenRouter (Gemma, Qwen, Llama) return JSON Schema descriptors as field values instead of actual data. `ExampleLLMClient` in `graphiti/zep_graphiti.py` converts schema instructions to concrete examples via `_schema_to_example()`. This is required for entity extraction to work.

### AsyncWorker (silent failure fix)
The stock Graphiti `AsyncWorker` in `ingest.py` only catches `CancelledError`. Any other exception kills the worker silently — all subsequent jobs are lost. Our patch catches all exceptions and logs them.

### Embedder configuration
Stock Graphiti only configures the LLM client from env vars, leaving the embedder on OpenAI defaults. Our patch wires `OPENAI_BASE_URL`, `EMBEDDING_MODEL_NAME` through to the embedder and cross-encoder.

## Windows-specific

- **IPv6 bug**: `localhost` resolves to `::1` on Windows but Docker only listens on `0.0.0.0`. Config default is `http://127.0.0.1:8000`.
- **Docker Desktop i/o timeout**: Docker commands sometimes fail with "i/o timeout" on Windows. Just retry.

## Reference: claude-mem architecture (inspiration)

claude-mem (by thedotmack) is our reference for how memory plugins work in Claude Code.
Source: https://github.com/thedotmack/claude-mem / https://docs.claude-mem.ai/hooks-architecture

### Full lifecycle diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        SESSION LIFECYCLE                            │
└─────────────────────────────────────────────────────────────────────┘

 ① SessionStart (au lancement + /clear + compact)
 ┌──────────────────────────────────────────────────────────────────┐
 │  Smart Install → vérifie deps, démarre le Worker (Bun :37777)   │
 │  Context Hook  → query SQLite (10 derniers résumés + 50 obs)    │
 │               → formate en "index progressif" compact            │
 │               → injecte via additionalContext (invisible user)   │
 │                                                                  │
 │  Claude voit: "# [claude-mem] recent context                    │
 │                Session 1: investigated auth bug...               │
 │                Session 2: implemented API endpoint..."           │
 └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ② UserPromptSubmit (à chaque message utilisateur)
 ┌──────────────────────────────────────────────────────────────────┐
 │  Pas de LLM ! Juste un HTTP POST au Worker (~20ms)              │
 │  → Stocke le prompt brut dans SQLite (user_prompts)             │
 │  → Sert de marqueur chronologique dans la timeline              │
 └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ③ Claude travaille (utilise Read, Edit, Bash, etc.)
 ┌──────────────────────────────────────────────────────────────────┐
 │  Le modèle peut aussi appeler les MCP tools claude-mem :        │
 │  • search → index compact (~50-100 tokens/résultat)             │
 │  • timeline → contexte chronologique                            │
 │  • get_observations → détails complets (si besoin)              │
 │  Pattern "progressive disclosure" = ~10x économie de tokens     │
 └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ④ PostToolUse (après CHAQUE appel d'outil)
 ┌──────────────────────────────────────────────────────────────────┐
 │  Pas de LLM ! Fire-and-forget HTTP POST au Worker (~8ms)        │
 │  Données capturées :                                            │
 │  • session_id, tool_name, tool_input, tool_output, timestamp    │
 │  → Enqueue dans observation_queue (SQLite)                      │
 │                                                                  │
 │  Le Worker (background, Bun) :                                  │
 │  • Poll la queue toutes les 1s                                  │
 │  • Compresse avec Claude Agent SDK (Haiku, CLI auth) → "obs"    │
 │  • Stocke dans SQLite pour future injection                     │
 └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ⑤ Stop (quand Claude finit de répondre)
 ┌──────────────────────────────────────────────────────────────────┐
 │  Seul moment où un LLM est appelé directement dans un hook :    │
 │  → Rassemble les observations de la session depuis SQLite       │
 │  → Claude Agent SDK génère un résumé structuré                  │
 │  → Tags: "investigated", "learned", "completed", "decided"      │
 │  → Stocké dans SQLite comme session summary                     │
 └──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
 ⑥ SessionEnd (fermeture)
 ┌──────────────────────────────────────────────────────────────────┐
 │  Marque la session comme terminée dans SQLite                   │
 │  Le Worker finit les opérations en cours                        │
 └──────────────────────────────────────────────────────────────────┘
```

### Architecture technique

```
Hooks (rapides, <1s)              Worker (background, Bun :37777)
┌─────────────┐                   ┌──────────────────────┐
│ SessionStart│──context─────────▶│ SQLite               │
│ UserPrompt  │──prompt──────────▶│  ├─ sessions          │
│ PostToolUse │──observation─────▶│  ├─ user_prompts      │
│ Stop        │──summarize──────▶│  ├─ observation_queue  │
└─────────────┘                   │  └─ observations      │
                                  │                        │
                                  │ Claude Agent SDK       │
                                  │ (Haiku, CLI auth)      │
                                  │                        │
                                  │ HTTP :37777            │
                                  │ + UI viewer (React)    │
                                  └──────────────────────┘
```

### Key design principles

- **No LLM in hooks** except Stop. All hooks are fire-and-forget HTTP to the Worker.
- **The model decides** when to use MCP search tools — claude-mem doesn't force it.
- **claude-mem doesn't auto-save user info**. It captures tool activity (observations). The model must explicitly decide to save.
- **Agent SDK uses CLI auth** (Max subscription), NOT a separate API key. Compression costs nothing extra.
- **Progressive disclosure**: SessionStart injects a compact index, not a full dump. Model fetches details on demand via MCP tools.
- **Graceful degradation**: if the Worker crashes, hooks log warnings but never block Claude Code.

### Our current gap vs claude-mem

We depend on the model calling `save_memory` explicitly. The model often doesn't think to do it (e.g., "Je démarre un projet C# avec Sylvie" → model treats it as action request, not info to save). claude-mem sidesteps this by capturing all tool activity automatically via PostToolUse, then compressing it — the model never needs to "decide" to save.

## TODO

- [x] Design spec
- [x] Implementation (MCP server, tools, hooks, skill)
- [x] Unit tests (36)
- [x] Integration tests (9, Docker + Ollama)
- [x] License (Apache-2.0)
- [x] OpenRouter support (patched Graphiti)
- [x] Minimal MCP tool surface (3 tools)
- [x] Scope-based group_id (no more model-invented group_ids)
- [x] Parallelized SessionStart hook (~1.5s)
- [ ] Improve model compliance: make Claude reliably call save_memory when user shares info
- [ ] PostToolUse hook for automatic observation capture (like claude-mem)
- [ ] Build pipeline / CI
- [ ] Publish to PyPI

## Git Convention

- Conventional commits style
- Current branch: `feat/openrouter-graphiti-patch` (6 commits ahead of main)

## Key Files

- `src/graph_mem/server.py` — MCP server (3 tools)
- `src/graph_mem/hooks/session_start.py` — SessionStart hook
- `src/graph_mem/hooks/session_end.py` — Stop hook
- `src/graph_mem/tools/context.py` — get_context (parallelized searches)
- `src/graph_mem/project_id.py` — git remote → project_id
- `src/graph_mem/config.py` — GRAPHITI_URL default 127.0.0.1:8000
- `graphiti/zep_graphiti.py` — ExampleLLMClient + embedder patch
- `graphiti/ingest.py` — AsyncWorker error handling patch
- `docker-compose.yml` — Neo4j 5.26 + patched Graphiti
- `tests/integration/` — e2e tests with Docker
