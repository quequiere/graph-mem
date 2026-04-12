# graph-mem — Embedding Benchmark Design Spec

**Date:** 2026-04-11
**Status:** Draft — awaiting user review
**Target dir:** `.benchmark/embedding/`
**Analysis output:** `.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md`

---

## 1. Purpose

Identify the best text embedding model(s) for `search_memory` in graph-mem,
focusing on bilingual (FR/EN) short-fact retrieval. Deliver a ranked
recommendation for:

- **Hosted default** — the best commercial API to use when online cost is OK
- **Local default** — the best self-hosted model for offline / sovereign setups
- **Hard negatives resistance** — which models survive temporal / negation /
  pronoun traps

The benchmark must be reproducible, cheap enough to re-run, and produce a
comparative table aligned with the existing extraction-benchmark analysis
format.

## 2. Scope

### In scope
- Retrieval quality (recall@1, recall@5, MRR) on a 100-doc / 40-query dataset
- Hard-negative discrimination (temporal / negation / pronoun edge cases)
- Matryoshka / elastic-dim truncation to 1024 vs native dimension
- Latency per document and per query (P50 / P95)
- Cost estimation per 1M input tokens and per full benchmark run
- Bilingual FR / EN coverage reflecting graph-mem usage patterns

### Out of scope
- Fine-tuning or domain adaptation
- Hybrid retrieval (BM25 + dense fusion)
- Reranking models (cross-encoders)
- End-to-end evaluation inside Graphiti (too many confounders)
- Quantized / int8 / binary embedding variants (future work)

## 3. Evaluation methodology

**Core metric set per model:**

| Metric | Definition |
|---|---|
| `recall@1` | Fraction of queries whose top-1 retrieved doc is in `relevant` |
| `recall@5` | Fraction of queries with at least one relevant doc in top-5 |
| `MRR` | Mean reciprocal rank of the first relevant doc (primary score) |
| `HN-penalty` | Mean rank of hard negatives across queries (higher = better) |
| `embed_latency_p50_ms` | Median per-doc embedding latency |
| `embed_latency_p95_ms` | 95th percentile per-doc embedding latency |
| `dim_native` / `dim_used` | Native vector dim and the dim actually used |
| `cost_per_1m_input_usd` | Vendor-listed price per 1M input tokens (null if local) |
| `cost_per_run_usd` | Total cost of a full run (corpus + queries) |

Scoring is computed on cosine similarity between L2-normalised embeddings.
For models supporting Matryoshka truncation, we report **two rows**:
native dim and truncated to 1024.

## 4. Dataset

### 4.1 Structure

File: `.benchmark/embedding/dataset.json`

```json
{
  "version": "1.0",
  "description": "Short-fact retrieval benchmark for graph-mem (FR/EN mix).",
  "corpus": [
    {
      "id": "c001",
      "text": "Bruno uses Python and likes Neo4j.",
      "lang": "en",
      "tags": ["user", "tech"]
    }
  ],
  "queries": [
    {
      "id": "q001",
      "query": "what tech does Bruno use?",
      "lang": "en",
      "relevant": ["c001", "c042"],
      "hard_negatives": ["c017"],
      "category": "user_fact"
    }
  ]
}
```

### 4.2 Target size

| Set | Count | Notes |
|---|---:|---|
| Corpus docs | 100 | 5 fictional personas × ~20 cross-linked facts |
| Queries | 40 | ~2.5× each persona, ~20 with hard negatives |
| Hard-negative-bearing queries | 20 | split across temporal / negation / pronoun / mixed-language |

### 4.3 Language distribution

- English: 60 docs / 24 queries
- French: 30 docs / 12 queries
- Mixed FR/EN or code-switched: 10 docs / 4 queries

Rationale: matches the user's observed FR/EN balance declared in the project's
CLAUDE.md and in the extraction benchmark's dataset.

### 4.4 Categories

Each query carries exactly one `category`:

| Category | Example query | Why |
|---|---|---|
| `user_fact` | "what tech does Bruno use?" | Baseline direct retrieval |
| `project_fact` | "where does Atlas run?" | Baseline project retrieval |
| `temporal` | "what did Bruno use before Python?" | Past vs present discrimination |
| `negation` | "who does NOT work on Atlas?" | Negation handling |
| `pronoun` | "who coaches me?" | First-person anchor |
| `mixed_lang` | "qui utilise Kubernetes ?" against EN docs | Cross-lingual retrieval |

### 4.5 Construction

1. **Seed (25 docs)** — reuse the 25 inputs from
   `.benchmark/llm/dataset.json`, rewritten as stand-alone factual sentences
   (strip extraction hints).
2. **Expansion (75 docs)** — I author additional docs in-session around 5
   fictional personas (dev Python, data scientist, SRE, student, tech lead),
   each with ~15 cross-linked facts.
3. **Queries (40)** — hand-written to ensure reliable relevance judgments.
4. **Hard negatives** — manually picked per query; docs that share surface
   features with the relevant ones but differ on temporal / negation / pronoun
   / language axes.
5. **Review gate** — user reviews `dataset.json` before the first benchmark
   run against any hosted API.

### 4.6 Ground-truth reliability

Every `relevant` and `hard_negatives` list is hand-authored. No LLM is used to
generate the ground truth — only to propose paraphrases, which are then
accepted or rejected by the human reviewer.

## 5. Models under test

Selected from the Perplexity top-10 report (2026-04-11) plus two budget
baselines for anchoring.

### 5.1 Hosted (commercial APIs)

| Slug | Model | Native dim | Also run at | Max ctx | API | Approx $/1M |
|---|---|---:|---:|---:|---|---:|
| `openai-3-large` | `text-embedding-3-large` | 3072 | 1024 | 8191 | OpenAI direct | 0.13 |
| `openai-3-small` | `text-embedding-3-small` | 1536 | 1024 | 8191 | OpenAI direct | 0.02 |
| `voyage-3-large` | `voyage-3-large` | 2048 | 1024 | 32000 | Voyage direct | 0.18 |
| `cohere-embed-v4` | `embed-v4.0` | 1024 | — | 512 | Cohere direct | 0.10 |
| `gemini-embed-2` | Gemini Embedding 2 | 3072 | 1024 | 8192 | Google direct | 0.20 |
| `mistral-embed` | `mistral-embed` | 1024 | — | 8000 | Mistral direct | 0.10 |

**Dim truncation:** models flagged with "also run at 1024" produce two rows in
the final table — the native-dim run and a Matryoshka-truncated-to-1024 run.

### 5.2 Local (open-source, self-hosted)

| Slug | Model | Native dim | Used at | Host | Notes |
|---|---|---:|---:|---|---|
| `qwen3-embedding-8b` | `Qwen/Qwen3-Embedding-8B` | 4096 | 1024 | Ollama | graph-mem baseline |
| `qwen3-embedding-4b` | `Qwen/Qwen3-Embedding-4B` | 2560 | 1024 | Ollama | Lighter Qwen3 |
| `jina-v3` | `jinaai/jina-embeddings-v3` | 1024 | — | sentence-transformers | CC-BY-NC free tier |
| `bge-m3` | `BAAI/bge-m3` | 1024 | — | sentence-transformers / Ollama | 100+ languages |
| `nomic-v1.5` | `nomic-ai/nomic-embed-text-v1.5` | 768 | — | Ollama | Budget option |

### 5.3 Reference-only (not recommended for graph-mem but useful as a ceiling)

| Slug | Model | Native dim | Why reference-only |
|---|---|---:|---|
| `stella-en-1.5b-v5` | `NovaSearch/stella_en_1.5B_v5` | 1024 | English-only; ceiling for pure-EN queries |

**Total:** 12 model configurations, 6 hosted + 5 local + 1 reference. With
Matryoshka dual runs (4 hosted models × 2 dim), that's **16 total rows** in
the final comparative table.

### 5.4 Selection rationale

- **Hosted mix** covers the five top commercial providers recommended by the
  Perplexity scan + `openai-3-small` as a cheap anchor to see if the premium
  OpenAI tier is worth the 6× price bump.
- **Local mix** keeps the current graph-mem baseline (`qwen3-embedding-8b`)
  and opposes it to the three most-cited open-source contenders (Jina v3,
  BGE-M3, Nomic v1.5) plus a lighter Qwen3 variant.
- **Stella** isolates whether going English-only buys meaningful quality —
  it is not a viable pick for graph-mem but it answers "how much are we
  leaving on the table by being bilingual?".

## 6. Harness

### 6.1 File layout

```
.benchmark/embedding/
├── .env.example           # all API keys (OPENAI, VOYAGE, COHERE, GOOGLE, MISTRAL)
├── requirements.txt       # httpx, pyyaml, numpy, python-dotenv, tiktoken,
│                          # sentence-transformers, torch
├── models.yaml            # per-model config (endpoint, dim, truncate_to, enabled)
├── dataset.json           # corpus + queries + relevance
├── run_benchmark.py       # main entry point
├── analyze.py             # aggregates results/*.json into markdown table
└── results/
    ├── <model-slug>.json       # embeddings + metrics for one run
    └── summary.json            # aggregated comparison
```

### 6.2 `models.yaml` schema

```yaml
models:
  openai-3-large:
    enabled: true
    family: openai
    model: text-embedding-3-large
    api_key_env: OPENAI_API_KEY
    endpoint: https://api.openai.com/v1/embeddings
    dim_native: 3072
    dim_runs: [3072, 1024]   # produces two rows
    price_per_1m_usd: 0.13
    max_ctx: 8191

  qwen3-embedding-8b:
    enabled: true
    family: ollama
    model: qwen3-embedding-8b
    endpoint: http://localhost:11434/api/embeddings
    dim_native: 4096
    dim_runs: [1024]
    price_per_1m_usd: null

  jina-v3:
    enabled: true
    family: sentence-transformers
    model: jinaai/jina-embeddings-v3
    dim_native: 1024
    dim_runs: [1024]
    price_per_1m_usd: null
```

Each `family` dispatches to a specific client adapter inside
`run_benchmark.py`. The adapter is responsible for calling the right
endpoint or loading the right local model.

### 6.3 Client families and adapters

| Family | Transport | Client |
|---|---|---|
| `openai` | HTTP | `httpx`, OpenAI-compat `/v1/embeddings` |
| `voyage` | HTTP | `httpx`, Voyage `/v1/embeddings` |
| `cohere` | HTTP | `httpx`, Cohere `/v2/embed` |
| `google` | HTTP | `httpx`, Vertex AI / Generative Language API |
| `mistral` | HTTP | `httpx`, Mistral `/v1/embeddings` |
| `ollama` | HTTP | `httpx`, Ollama `/api/embeddings` |
| `sentence-transformers` | In-process | `sentence-transformers` library |

All adapters share a common `embed(texts: list[str]) -> list[list[float]]`
contract. Batching is handled by each adapter (default batch size 32, tunable
in `models.yaml`).

### 6.4 Run flow

For each enabled model × each requested `dim_runs` value:

1. **Warm-up** — embed 3 throwaway strings to prime connection pools and
   local model load.
2. **Embed corpus** — time individual calls, record P50 / P95.
3. **Embed queries** — same.
4. **Truncate** if `dim_used < dim_native` (simple slice + L2 renormalise).
5. **Cosine top-k** — for each query, rank all corpus docs by cosine similarity.
6. **Compute metrics** — recall@1, recall@5, MRR, HN-penalty per category,
   then aggregate.
7. **Dump** — `results/<slug>_<dim>.json` with:
   - metadata (model, dim, timestamps, versions)
   - per-query ranks and scores
   - aggregate metrics
   - latency distribution
   - cost estimate

### 6.5 CLI

```bash
python run_benchmark.py                                   # all enabled models
python run_benchmark.py --phase hosted                    # only commercial APIs
python run_benchmark.py --phase local                     # only local models
python run_benchmark.py --only qwen3-embedding-8b         # a single model
python run_benchmark.py --dry-run                         # skip API calls, validate config
```

`--dry-run` is mandatory before the first paid run: it loads `models.yaml`,
resolves API keys, embeds a 3-doc canary against each model, and prints the
expected cost for the full run. **The user explicitly gates the switch from
`--dry-run` to a live run.**

### 6.6 Analysis step

`python analyze.py` reads every `results/*.json` and writes:

- `results/summary.json` — machine-readable comparative table
- `.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md` — human
  analysis document following the same format as the extraction analysis,
  with TL;DR, comparison table, per-category breakdown, and final picks for
  hosted and local graph-mem defaults.

## 7. Cost and budget safeguards

**Budget envelope for the full hosted run:**

Input tokens per run (rough estimate):
- Corpus: 100 docs × ~50 tokens avg = 5,000 tokens
- Queries: 40 queries × ~15 tokens avg = 600 tokens
- Total: ~5,600 tokens per run

Per-model cost for one run:

| Model | $/1M | Cost per run |
|---|---:|---:|
| `openai-3-large` | 0.13 | $0.00073 |
| `openai-3-small` | 0.02 | $0.00011 |
| `voyage-3-large` | 0.18 | $0.00101 |
| `cohere-embed-v4` | 0.10 | $0.00056 |
| `gemini-embed-2` | 0.20 | $0.00112 |
| `mistral-embed` | 0.10 | $0.00056 |
| **Total hosted run** | — | **~$0.004** |

Even with 5 iterations during development, total hosted spend stays well
under $0.05. No budget risk.

**Safeguards:**
- `--dry-run` mandatory before first live run
- User explicit go-ahead before the first live run against any paid API
- Results cached on disk; re-runs only happen on deliberate `--force`

## 8. Reproducibility

- Every `results/*.json` includes: model slug, dim, timestamp, git commit,
  library versions, dataset version, hardware (CPU/GPU/RAM for local runs).
- The dataset is versioned via the `version` field in `dataset.json`;
  changing it invalidates prior results for the same slug.
- All prompts / templates used during dataset authoring are committed in
  `dataset.json` itself (as top-level metadata), not buried in scripts.

## 9. Deliverables

1. `.benchmark/embedding/` full harness (dataset, runner, analyzer)
2. `.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md` final analysis
3. Recommended `.env` values for graph-mem (embedding model + dim) based on
   the analysis
4. Updated `CLAUDE.md` mention of the current embedding recommendation

## 10. Execution gates (for the human reviewer)

Two explicit stop points where the user must confirm before proceeding:

1. **Dataset review** — after dataset construction, before any model is run.
2. **Hosted launch** — after `--dry-run` passes, before the first live hosted
   call. At this point the user switches me to max-reflexion mode as
   requested.

## 11. Open questions / known risks

- **OpenRouter coverage** — OpenRouter does not currently expose all five
  commercial embedding APIs. The spec assumes direct API keys per vendor.
  If the user only has OpenRouter, a reduced hosted set (OpenAI + Gemini via
  OpenRouter) is the fallback.
- **Voyage availability** — Voyage is not on OpenRouter. Requires a Voyage
  API key (generous free tier is available for evaluation).
- **Jina v3 license** — CC-BY-NC means it's free to benchmark but the graph-
  mem production deployment would need a paid Jina Cloud tier or a different
  pick if Jina wins.
- **Stella English-only** — flagged as reference-only; the final pick for
  graph-mem must remain bilingual.
- **Hard-negative authoring bias** — 20 HN queries hand-authored by one
  person in one session. There is a real risk of overfitting to that
  person's writing style. Mitigation: dataset is small enough to re-review
  and expand later if the results look suspicious.

## 12. Non-goals (for explicitness)

- We do not optimise any model (no LoRA, no calibration).
- We do not tune batch sizes beyond reasonable defaults.
- We do not test cold-start latency for local models beyond a single warm-up.
- We do not evaluate ingestion throughput end-to-end in Graphiti.
