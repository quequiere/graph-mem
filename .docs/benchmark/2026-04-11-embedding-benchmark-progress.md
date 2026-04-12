# Embedding Benchmark — Progress log

**Last update:** 2026-04-11 (before conversation compaction)
**Branch:** `feat/auto-capture-hook`
**Harness dir:** `.benchmark/embedding/`
**Spec:** `.docs/superpowers/specs/2026-04-11-embedding-benchmark-design.md`
**Original plan:** `.docs/superpowers/plans/2026-04-11-embedding-benchmark.md`  **(outdated — see deviations below)**

---

## Where we are

**Status:** Ready to execute the hosted phase. Waiting on user to switch Claude to max-reflexion mode and give the go-ahead. **Do not run the hosted phase without the user's explicit go.**

**Next command when the user says go:**
```bash
cd .benchmark/embedding && BENCHMARK_SSL_VERIFY=false PYTHONIOENCODING=utf-8 python run_benchmark.py --phase hosted
```

## Deviations from the original plan (read these carefully)

The plan under `.docs/superpowers/plans/2026-04-11-embedding-benchmark.md` is now **outdated**. It still shows the original 16-task TDD workflow. What actually happened:

1. **TDD was dropped at the user's request.** The user said "zap le tdd c'est pour du benchmarking". No pytest suite, no red/green cycles, no MockTransport. Validation happens via `--dry-run` and the real run, not unit tests. Memory: `feedback_no_tdd_for_benchmarks.md`.

2. **User handles all git commits personally.** Do not run `git add` or `git commit` in any task. Write files and let the user commit. Memory: `feedback_git_commits.md`.

3. **Hosted set reduced from 6 vendors to 5 OpenRouter-routed models.** The user only has `OPENROUTER_API_KEY` — Voyage, Cohere, Mistral were dropped because they are NOT available on OpenRouter's `/v1/embeddings` endpoint. OpenRouter DOES expose:
   - `openai/text-embedding-3-small` (1536d)
   - `openai/text-embedding-3-large` (3072d)
   - `google/gemini-embedding-001` (3072d)
   - `baai/bge-m3` (1024d) — *open-source, also benchmarked locally below*
   - `qwen/qwen3-embedding-8b` (4096d) — *graph-mem's current production model, also benchmarked locally*
   The last two let us measure hosted-vs-local quality on identical weights (tests GGUF quantization impact for Ollama).

4. **Corporate proxy requires `BENCHMARK_SSL_VERIFY=false`.** The Michelin proxy intercepts HTTPS with a self-signed root. `adapters/_http.py` reads this env var and passes `verify=False` to httpx. Always prefix hosted/local runs with this env var. Memory: `feedback_corp_proxy_ssl.md`.

5. **`pronoun` category was dropped from the dataset.** The corpus is 100% third-person so first-person pronoun queries had no clean ground truth. Queries rebalanced to 15 user_fact / 10 project_fact / 5 temporal / 5 negation / 5 mixed_lang = 40 total.

## What's done

| # | Task | Status | Notes |
|---|---|---|---|
| 1 | Scaffold `.benchmark/embedding/` | ✅ | 9 files: .env.example, requirements.txt, pyproject.toml, .gitignore, __init__.py × 3, tests/conftest.py, results/.gitkeep |
| 2 | Core modules | ✅ | `dataset.py` (load_dataset + Corpus/Query/Dataset dataclasses + DatasetError), `ranking.py` (l2_normalize, cosine_matrix, rank_ids, truncate_and_renormalize), `metrics.py` (QueryEval, evaluate_query, aggregate, hard-negative mean rank) |
| 3 | All adapters | ✅ | `adapters/base.py` (protocol + EmbedResult), `adapters/_http.py` (default_client + l2_normalise_rows + SSL verify env plumbing), 7 concrete adapters: openai, voyage, cohere, google, mistral, ollama, st. **The non-OR adapters (voyage/cohere/mistral/google) exist but are unused.** |
| 4 | Runner + CLI + models.yaml | ✅ | `runner.py` (RunConfig, run_benchmark, _environment_snapshot), `run_benchmark.py` (ModelEntry, load_models_yaml, build_adapter with `openrouter` family alias, main CLI with --phase/--only/--dry-run/--force), **new `models.yaml`** (see below) |
| 5 | Analyzer | ✅ | `analyze.py` (aggregate_results, write_markdown_summary, _fmt_cost, main) |
| 6 | Dataset authoring | ✅ | `dataset.json` — 100 corpus / 40 queries / 26 HN-bearing / 5 personas (Bruno, Julie, Karim, Lea, Thomas) / lang 68 en + 27 fr + 5 mixed |
| 7 | Dry-run validation | ✅ | All 5 hosted OR models returned 200, latencies 117-265 ms, est. cost $0.004 full-run |
| — | USER GATE | ⏳ | **Stopped here. Waiting for the user's go + max-reflexion switch.** |
| 8 | Execute hosted phase | ⏳ | Next step |
| 9 | Execute local phase | ⏳ | Requires Ollama + HF downloads |
| 10 | Analysis writeup | ⏳ | Final task — write `.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md` |

## Current `models.yaml` structure

```
HOSTED (via OpenRouter — family: openrouter)
├── or-openai-3-small    (1536d, dim_runs: [1536, 1024], ~$0.02/1M)
├── or-openai-3-large    (3072d, dim_runs: [3072, 1024], ~$0.13/1M)
├── or-gemini-001        (3072d, dim_runs: [3072, 1024], ~$0.15/1M)
├── or-bge-m3            (1024d, dim_runs: [1024],       ~$0.01/1M)
└── or-qwen3-8b          (4096d, dim_runs: [4096, 1024], ~$0.01/1M)

LOCAL
├── qwen3-embedding-8b   (Ollama,                 dim_runs: [1024])
├── qwen3-embedding-4b   (Ollama,                 dim_runs: [1024])
├── bge-m3               (sentence-transformers,  dim_runs: [1024])
├── jina-v3              (sentence-transformers,  dim_runs: [1024])
├── nomic-v1.5           (sentence-transformers,  dim_runs: [768])
└── stella-en-1.5b       (sentence-transformers,  dim_runs: [1024], ref-only — English)
```

Total expected result files: **8 hosted rows + 6 local rows = 14 rows** in the final comparative table.

## Dry-run outcome (for reference)

```
=== or-openai-3-small (openrouter / openai/text-embedding-3-small) ===
  dry-run ok — canary shape (3, 1536), p50 latency 165.3 ms

=== or-openai-3-large (openrouter / openai/text-embedding-3-large) ===
  dry-run ok — canary shape (3, 3072), p50 latency 117.1 ms

=== or-gemini-001 (openrouter / google/gemini-embedding-001) ===
  dry-run ok — canary shape (3, 3072), p50 latency 204.0 ms

=== or-bge-m3 (openrouter / baai/bge-m3) ===
  dry-run ok — canary shape (3, 1024), p50 latency 265.1 ms

=== or-qwen3-8b (openrouter / qwen/qwen3-embedding-8b) ===
  dry-run ok — canary shape (3, 4096), p50 latency 157.8 ms
```

All 5 hosted models respond correctly via OR. The OR response includes a `usage.cost` field (real USD cost per call) — the runner currently estimates cost from char/4 tokens, which is approximate. If we want accurate cost we can plumb `usage.cost` from the OR response into the adapter; out of scope for now unless the user asks.

## Key files to check after compaction

| File | Purpose |
|---|---|
| `.benchmark/embedding/models.yaml` | Authoritative list of enabled models — current set is 5 hosted (OR) + 6 local |
| `.benchmark/embedding/dataset.json` | 100 docs / 40 queries / 5 personas, hand-authored ground truth |
| `.benchmark/embedding/run_benchmark.py` | CLI — supports `--phase hosted|local`, `--only <slug>`, `--dry-run`, `--force` |
| `.benchmark/embedding/runner.py` | Core `run_benchmark(adapter, dataset, cfg)` — dumps `results/<slug>_dim<N>.json` |
| `.benchmark/embedding/adapters/_http.py` | SSL verify is gated by `BENCHMARK_SSL_VERIFY` env var (default `true`) |
| `.benchmark/embedding/.env` | Contains `OPENROUTER_API_KEY` only — no other vendor keys |
| `.docs/superpowers/specs/2026-04-11-embedding-benchmark-design.md` | Original design spec — still accurate on methodology, OUTDATED on model list |
| `.docs/superpowers/plans/2026-04-11-embedding-benchmark.md` | Original TDD-flavored 16-task plan — OUTDATED on both workflow (no TDD) and model set |

## How to resume

1. Read this progress file first
2. Check `C:\Users\E040358\.claude\projects\C--Users-E040358-src-graph-mem\memory\MEMORY.md` for feedback memories (git commits, TDD skip, SSL proxy)
3. Ask the user: "Prêt à lancer la phase hosted ? (5 modèles via OpenRouter, ~$0.004, ~2-4 min)". **Do not launch without explicit go.**
4. When the user says go:
   ```bash
   cd .benchmark/embedding && BENCHMARK_SSL_VERIFY=false PYTHONIOENCODING=utf-8 python run_benchmark.py --phase hosted
   ```
5. After hosted phase: check Ollama is running (`ollama list`) and that `qwen3-embedding-8b` + `qwen3-embedding-4b` are pulled. If not, ask user to pull them before running local phase.
6. Local phase:
   ```bash
   cd .benchmark/embedding && BENCHMARK_SSL_VERIFY=false PYTHONIOENCODING=utf-8 python run_benchmark.py --phase local
   ```
   Note: sentence-transformers models (bge-m3, jina-v3, nomic, stella) will auto-download from HF on first use. Big disk hit, may take time.
7. Analyze:
   ```bash
   cd .benchmark/embedding && python analyze.py
   ```
8. Write the final analysis to `.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md` following the same format as the extraction analysis at `.docs/benchmark/2026-04-11-extraction-benchmark-analysis.md`.

## Open questions for the next session

- **Cost accuracy:** should the OpenAIAdapter parse `usage.cost` from the OR response and record real cost instead of the char/4 estimate? Not critical but would make the cost column more trustworthy.
- **Ollama availability:** do the Qwen3 embedding models exist under those exact tags in the user's local Ollama? Untested — needs `ollama list` before launching the local phase.
- **HuggingFace downloads through the proxy:** `sentence-transformers` may hit the same corporate SSL issue when pulling weights from HF. If so, set `HF_HUB_DISABLE_TELEMETRY=1` and point `REQUESTS_CA_BUNDLE` at the corporate cert, or configure the HF client to skip verify.
