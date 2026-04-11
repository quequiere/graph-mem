# graph-mem — Entity/Relation Extraction Benchmark Analysis

**Date:** 2026-04-11
**Dataset:** `.benchmark/dataset.json` (25 cases, FR/EN, 12 categories)
**Raw results:** `.benchmark/results/result.json`
**Goal:** pick one or more local Ollama models that can replace OpenRouter-hosted LLMs as the graph-mem extraction backend.

---

## 1. TL;DR — Recommendation

| Scenario | Pick | Why |
|---|---|---|
| **Local backend (primary)** | `gemma3:4b` (Ollama) | Best type accuracy of the whole benchmark (**0.902**), strong semantic F1 (E=0.874 / R=0.697), acceptable 10.4 s latency, zero API cost. |
| **Local backend (if <10 s latency is a hard requirement)** | `phi4-mini:latest` | 6.9 s latency, decent quality (E=0.803 / R=0.610). Trades ~7 % entity quality for ~30 % speed. |
| **Hosted fallback (quality floor)** | `google/gemini-2.5-flash-lite` | Highest semantic relation F1 (**0.708**), 1.6 s latency, ≈ $0.002 per 25-case run. Much cheaper than Claude for comparable quality. |
| **Hosted reference (top quality, cost no object)** | `deepseek/deepseek-chat-v3-0324` | Best semantic entity F1 (**0.951**) and excellent consistency. 7.3 s latency, ≈ $0.003/run. |
| **To avoid for graph-mem** | `gemma4:e2b`, `gemma4:e4b`, `qwen3:8b`, `mistral-nemo:12b`, `llama3.2:3b`, `granite3.3:8b` | Too slow, or broken schema, or relation F1 < 0.55. |

**Bottom line:** `gemma3:4b` via Ollama is the right local replacement for OpenRouter in graph-mem. It beats every hosted model on **type accuracy** (which matters a lot for a typed knowledge graph) while keeping relation F1 on par with DeepSeek V3. Its 10.4 s latency is above the 10 s target for the `SessionStart` hook budget — acceptable for async `save_memory`, borderline for synchronous recall.

---

## 2. Comparative table (all 15 runs)

Sorted by composite semantic F1 (0.5·entity + 0.5·relation), then by latency.

| # | Model | Backend | Success | Lat. (ms) | E F1 (strict) | R F1 (strict) | **E F1 (sem)** | **R F1 (sem)** | Type acc. | In tok. | Out tok. | Cost 25 cases |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `deepseek-chat-v3-0324` | OpenRouter | 25/25 | 7 327 | 0.931 | 0.816 | **0.951** | 0.699 | 0.580 | 9 330 | 1 970 | **$0.00338** |
| 2 | `gemini-2.5-flash-lite` | OpenRouter | 25/25 | **1 608** | 0.893 | 0.795 | 0.929 | **0.708** | 0.327 | 9 330 | 2 911 | $0.00210 |
| 3 | `mistral-small-3.2-24b` | OpenRouter | 25/25 | 1 483 | 0.903 | 0.784 | 0.923 | 0.579 | 0.792 | 9 330 | 2 391 | $0.00118 |
| 4 | **`gemma3:4b`** | **Ollama local** | 25/25 | 10 387 | 0.825 | 0.724 | 0.874 | 0.697 | **0.902** | 9 330 | 2 132 | **local** (≈ $0.00054 on OR) |
| 5 | `gemma4:e2b` (= Gemma 3n E2B) | Ollama local | 25/25 | 37 048 | 0.890 | 0.782 | 0.918 | 0.652 | 0.694 | 9 330 | 1 814 | local (free tier only on OR) |
| 6 | `gpt-4o-mini` | OpenRouter | 25/25 | 1 316 | 0.890 | 0.693 | 0.910 | 0.622 | 0.064 | 9 330 | 1 242 | $0.00214 |
| 7 | `claude-haiku-4.5` | OpenRouter | 25/25 | 1 969 | 0.891 | 0.782 | 0.891 | 0.694 | 0.647 | 9 330 | 2 408 | $0.02137 |
| 8 | `qwen-turbo` | OpenRouter | 25/25 | 1 576 | 0.840 | 0.749 | 0.860 | 0.593 | 0.646 | 9 330 | 1 868 | $0.00055 |
| 9 | `granite3.3:8b` | Ollama local | 25/25 | 21 316 | 0.835 | 0.579 | 0.865 | 0.511 | 0.605 | 9 330 | 2 518 | local (no OR equiv.) |
| 10 | `phi4-mini:latest` | Ollama local | 25/25 | **6 941** | 0.787 | 0.639 | 0.803 | 0.610 | 0.733 | 9 330 | 2 377 | local (no OR equiv.) |
| 11 | `llama3.2:3b` | Ollama local | 25/25 | 8 261 | 0.827 | 0.413 | 0.860 | 0.325 | 0.711 | 9 330 | 2 481 | local (≈ $0.00132 on OR) |
| 12 | `gemma-4-26b-a4b-it` | OpenRouter | 25/25 | 8 712 | 0.735 | 0.720 | 0.783 | 0.582 | 0.341 | 9 330 | 2 782 | $0.00223 |
| 13 | `gemma4:e4b` (= Gemma 3n E4B) | Ollama local | **21/25** | 17 464 | 0.725 | 0.525 | 0.745 | 0.453 | 0.514 | 9 330 | 1 629 | local (≈ $0.00025 on OR) |
| 14 | `mistral-nemo:12b` | Ollama local | 25/25 | 23 859 | 0.458 | 0.576 | 0.472 | 0.536 | 0.667 | 9 330 | 1 754 | local (≈ $0.00026 on OR) |
| 15 | `qwen3:8b` | Ollama local | **9/25** | 137 268 | 0.339 | 0.357 | 0.339 | 0.272 | 0.667 | 9 330 | 666 | local (≈ $0.00073 on OR) |

Notes on the cost column:
- **Input tokens** are counted exactly for each case using `tiktoken` (`cl100k_base`) on the full `EXTRACTION_PROMPT` + input text. All models share the same input distribution → 9 330 total / ~373 per case / ~358 base prompt.
- **Output tokens** are counted from the stored `raw_response`, which the benchmark runner truncates to 600 characters. Three hosted runs hit that cap on 1–2 cases (`gemini-flash-lite`, `gemma4-26b`, `granite3.3-8b`), so the real output could be marginally higher — the cost estimate is a lower bound, but the error is < 10 % on the affected rows.
- **"local"** = no dollar cost on your machine (only GPU/CPU power). The parenthesised `$` figure is **what it would have cost on OpenRouter** if the equivalent hosted model had been used — helps you reason about "is it worth running this locally at all?".
- Pricing was fetched live from the OpenRouter `/api/v1/models` endpoint on 2026-04-11. `phi4-mini`, `granite3.3-8b` and `gemma4:e2b` (paid tier) have no OpenRouter equivalent and are marked accordingly.

---

## 3. Cost analysis

### 3.1 Cost per extraction call

Unit cost for a single extraction call (average over the 25-case run):

| Model | $ / extraction | $ / 1k extractions | $ / 100 k extractions |
|---|---:|---:|---:|
| `claude-haiku-4.5` | $0.000855 | $0.855 | **$85.48** |
| `deepseek-v3` | $0.000135 | $0.135 | $13.50 |
| `gemma-4-26b-a4b-it` | $0.0000893 | $0.089 | $8.93 |
| `gpt-4o-mini` | $0.0000856 | $0.086 | $8.56 |
| `gemini-2.5-flash-lite` | $0.0000839 | $0.084 | $8.39 |
| `mistral-small-3.2-24b` | $0.0000471 | $0.047 | $4.71 |
| `qwen-turbo` | **$0.0000218** | **$0.022** | **$2.18** |
| `gemma3:4b` *(local; OR equiv.)* | $0.0000216 | $0.022 | $2.16 |
| `mistral-nemo:12b` *(local; OR equiv.)* | $0.0000105 | $0.011 | $1.05 |
| `gemma4:e4b` *(local; OR equiv.)* | $0.0000101 | $0.010 | $1.01 |
| `llama3.2:3b` *(local; OR equiv.)* | $0.0000529 | $0.053 | $5.29 |
| `qwen3:8b` *(local; OR equiv.)* | $0.0000293 | $0.029 | $2.93 |
| `gemma4:e2b` *(local; only free tier on OR)* | $0 | $0 | $0 |
| `phi4-mini`, `granite3.3:8b` | — | no OpenRouter equivalent | — |

### 3.2 Realistic graph-mem usage scenario

Assuming a single developer using graph-mem:
- **5 Claude Code sessions / day**, each triggering the `SessionStart` hook → ~4 semantic search calls (already parallelised, no extraction).
- **~20 `save_memory` calls / day** from user messages → 20 extractions.
- **~30 `PostToolUse`-derived extractions / day** (planned future feature, cf. CLAUDE.md TODO).

That's roughly **50 extractions / day / user**, i.e. **1 500 / month / user**.

Monthly hosted cost per user:
- `claude-haiku-4.5`: **~$1.28 / user / month** — prohibitive at team scale.
- `deepseek-v3`: **~$0.20 / user / month** — excellent quality / cost ratio.
- `gemini-2.5-flash-lite`: **~$0.13 / user / month** — best deal on the quality ceiling.
- `qwen-turbo`: **~$0.03 / user / month** — basically free, but quality drop on types & relations.
- **`gemma3:4b` local: $0 / user / month** (plus electricity / GPU amortisation).

At 10 users (a small team) with the same workload, the only backend that breaks $10/month is Claude Haiku (~$128). DeepSeek stays around $2, Gemini around $1.30. **Running gemma3:4b locally saves ~$2–$128/month depending on which hosted model you were replacing, in exchange for ~8 s of extra latency per call.**

### 3.3 Cost of the benchmark itself

Full hosted-only run of this benchmark on OpenRouter (15 models × 25 cases, only the 7 hosted ones actually incurring charges): **~$0.073** total.

---

## 4. Per-category quality (semantic F1) — top candidates

| Category | Claude H. | DeepSeek V3 | Gemini FL | GPT-4o-mini | Mistral Sm. | **gemma3:4b** | gemma4:e2b | phi4-mini | Qwen Turbo |
|---|---|---|---|---|---|---|---|---|---|
| simple_entity | 0.80 / 0.80 | 1.00 / 0.67 | 0.91 / 0.80 | 1.00 / 0.80 | 1.00 / 0.80 | 0.91 / 0.80 | 1.00 / 0.80 | 0.80 / 0.80 | 0.91 / 0.67 |
| simple_relation | 1.00 / 0.77 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.83 | 1.00 / 0.67 | 0.76 / 0.84 | 0.94 / 0.84 | 1.00 / 0.84 | 1.00 / 0.61 |
| dedup | 0.94 / 0.44 | 0.94 / 0.57 | 0.88 / 0.27 | 1.00 / 0.44 | 1.00 / 0.44 | 1.00 / 0.44 | 0.84 / 0.44 | 0.91 / 0.67 | 0.84 / 0.67 |
| temporal | 0.91 / 0.50 | 1.00 / 0.50 | 1.00 / 0.50 | 1.00 / 1.00 | 1.00 / 0.50 | 1.00 / 0.91 | 1.00 / 0.25 | 1.00 / 0.60 | 0.86 / 0.86 |
| user_preference | 0.67 / 1.00 | 0.84 / 0.91 | 0.62 / 0.74 | 0.57 / 0.22 | 0.67 / 0.74 | 0.91 / 0.74 | 0.67 / 0.50 | 0.49 / 0.49 | 0.74 / 0.40 |
| project_fact | 0.96 / 0.40 | 0.96 / 0.17 | 0.96 / 0.50 | 1.00 / 0.40 | 1.00 / 0.17 | 0.91 / 0.57 | 0.89 / 0.47 | 0.72 / 0.33 | 1.00 / 0.17 |
| mixed | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.17 | 1.00 / 0.25 | 1.00 / 0.17 |
| pronoun | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.67 | 1.00 / 0.50 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.50 | 0.80 / 1.00 |
| negation | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 | 1.00 / 0.00 | 1.00 / 1.00 |
| nested | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 0.67 | 1.00 / 1.00 | 1.00 / 0.67 | 1.00 / 1.00 |
| abstract | 0.80 / 1.00 | 0.50 / 1.00 | 1.00 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.80 / 1.00 | 0.00 / 0.50 | 0.00 / 0.00 |
| edge | 0.67 / 0.67 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.67 / 1.00 | 1.00 / 1.00 | 0.40 / 1.00 | 0.67 / 1.00 |

Values are **entity F1 / relation F1** (semantic). Highlights:

- **`mixed` is hard for everyone** (relation F1 ≈ 0.17 across the board) — this is a dataset signal, not a model signal. Probably worth revisiting how relations are labelled in that category before drawing conclusions from it.
- **`gemma3:4b` shines on `user_preference` (0.91/0.74)** and `temporal` (1.00/0.91) — the two categories most relevant to a personal memory system. It's the only local model that stays strong on these.
- **`abstract` is an edge case** (1 single test), and `gemma3:4b`, `gpt-4o-mini`, `mistral-small`, `qwen-turbo` all score 0 on entities for it. Don't over-weight this single point.
- **`gemini-flash-lite`** has the most uniform profile across categories — it almost never collapses. The only weak spot is `user_preference` entities.
- **`claude-haiku`** has the best `user_preference` relation score (1.00) — notable because that's the category where a personal-memory plugin will be used the most.

---

## 5. Type accuracy — a sleeper metric

Type accuracy = among entities whose **name** matches the expected, what fraction also has the expected **type**. Critical for graph-mem because Neo4j node labels come from the extracted type.

| Model | Type accuracy |
|---|---:|
| **`gemma3:4b`** | **0.902** |
| `mistral-small-3.2-24b` | 0.792 |
| `phi4-mini` | 0.733 |
| `llama3.2:3b` | 0.711 |
| `gemma4:e2b` | 0.694 |
| `mistral-nemo:12b` | 0.667 |
| `qwen3:8b` (non-representative) | 0.667 |
| `claude-haiku-4.5` | 0.647 |
| `qwen-turbo` | 0.646 |
| `granite3.3:8b` | 0.605 |
| `deepseek-v3` | 0.580 |
| `gemma4:e4b` | 0.514 |
| `gemma-4-26b-a4b-it` | 0.341 |
| `gemini-2.5-flash-lite` | 0.327 |
| `gpt-4o-mini` | **0.064** |

`gpt-4o-mini` is spectacularly bad at types (6.4 %), which is worth knowing if you were considering it for typed extraction. `gemma3:4b` is 14× better on this metric than `gpt-4o-mini` and 2.8× better than `gemini-flash-lite`. This is the biggest differentiator in favor of `gemma3:4b`.

---

## 6. Caveats & known issues

1. **`qwen3:8b` — 9/25 success (thinking mode).** The benchmark sends `think: false` but the stock Ollama Modelfile for `qwen3:8b` still emits `<think>…</think>` tags that break JSON parsing. **Result not representative** — would need a Modelfile override (`TEMPLATE /nothink` or a wrapper) to retest.
2. **`gemma4:e4b` — 4/25 failed_to_parse_json.** Schema non-compliance: the model emits malformed relation objects. Not fixable via prompt, likely a tokeniser/template issue in Ollama.
3. **`gemma4:26b-local` — run stopped mid-benchmark** (not in `result.json`). The OpenRouter version `gemma-4-26b-a4b-it` is in the results and scores poorly (Esem 0.783, Rsem 0.582, type 0.341), so running it locally was unlikely to be promising.
4. **OpenRouter runs used `verify=False`** (corporate proxy SSL interception). Response content is unaffected.
5. **`raw_response` is truncated at 600 chars** in `result.json`, which undercounts output tokens on ≤ 3 hosted runs (see §2 note). Impact on cost estimates: < 10 %.
6. **Per-case sample size is small** (1–3 cases per category × 12 categories = 25 total). Treat category-level F1 scores as directional, not statistically significant.
7. **Single-turn, short inputs.** This benchmark does not test long-context extraction (>500 tokens of input), multi-entity disambiguation across paragraphs, or streaming. Local models with short context windows (e.g. `gemma3:4b` has 8k ctx in Ollama default) may degrade on real session summaries from the Stop hook.
8. **Latency is single-request.** `gemma3:4b` at 10 s and `gemma4:e2b` at 37 s could be reduced significantly with persistent context or batching — the current benchmark reloads context on every call.

---

## 7. Proposed next steps

1. **Adopt `gemma3:4b` as the default local backend** for graph-mem's Graphiti extraction LLM. Update `docker-compose.yml` / `.env` to point `OPENAI_BASE_URL` at a local Ollama (`http://host.docker.internal:11434/v1`) and `LLM_MODEL_NAME=gemma3:4b`.
2. **Verify `gemma3:4b` passes the existing `ExampleLLMClient` schema echoing test** (cf. `graphiti/zep_graphiti.py`). Gemma models are known culprits — the patch may or may not be needed depending on the prompt Graphiti actually sends at runtime.
3. **Re-run `qwen3:8b` with a proper `/nothink` Modelfile.** If it recovers to > 0.8 entity F1 it becomes a strong second candidate (more params = likely better on long context).
4. **Keep `gemini-2.5-flash-lite` as the hosted fallback** in `.env` / config, activated when `ollama` is unreachable. Best quality/cost/latency ratio among hosted models; 8× cheaper than Claude Haiku for equivalent quality.
5. **Expand the dataset before any retraining decision.** The current 25 cases are too few for the `mixed`, `abstract`, `edge` categories to be meaningful. Target ≥ 5 cases per category before re-benchmarking.
6. **Add a long-input category** (~500–1 500 tokens, simulating a `Stop` hook session summary). This is the actual production workload and is not tested today.
7. **Measure end-to-end latency through Graphiti**, not just the raw LLM call. Graphiti's entity-resolution step adds 1–3 × the LLM latency — the 10 s figure for `gemma3:4b` is a floor, not a ceiling.

---

## 8. Appendix — methodology

- **Prompt:** identical across all models, from `.benchmark/run_benchmark.py:45-65`. 358 base tokens, + ~15 tokens per input case on average.
- **Parameters:** `temperature=0.0`, `max_tokens=1024`, `think=false` for Ollama thinking models.
- **Scoring** (also from `run_benchmark.py`):
  - *Strict:* set-based name match for entities; partial triple credit for relations (≥2/3 components).
  - *Semantic:* fall back to `nomic-embed-text` cosine similarity ≥ 0.85 for residual entities/relations that missed the strict match.
  - *Type accuracy:* among name-matched entities, share with correct type.
- **Tokenisation** for cost: `tiktoken cl100k_base` applied to actual prompt + stored `raw_response`. Not the native tokeniser of each model — close enough for cost order-of-magnitude, but each provider bills on its own tokeniser. Expect ±15 % error vs. a billed invoice.
- **Pricing source:** live fetch from `https://openrouter.ai/api/v1/models` on 2026-04-11. Prices may change.
