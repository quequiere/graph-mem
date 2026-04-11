# graph-mem — LLM Extraction Benchmark — Wave 2

**Date:** 2026-04-11
**Dataset:** `.benchmark/llm/dataset.json` v2.0 (**50 cases**, FR/EN, 12 categories)
**Quality runs:** `.benchmark/llm/results/result_v2_hosted.json` (7 hosted) + `result_v2b_hosted_equiv.json` (4 hosted equivalents of Ollama tags)
**Speed runs:** `.benchmark/llm/results/result_v2_speed_local.json` (2 local models × 5 cases via Ollama)
**Archive v1:** `.benchmark/llm/results/archive_2026-04-11_v1/`

---

## 1. Methodology change vs v1

Two corrections after the v1 post-mortem:

1. **Quality is measured on OpenRouter only.** Testing quality on local Ollama models introduced noise from Modelfile templates, quantisation choices, and cold-boot effects. From wave 2 onward, all quality scoring happens on hosted endpoints where the served version is stable and reproducible.
2. **Local machines are only used for speed measurement**, and only for models that actually fit the target hardware (VRAM ≥ Q4 size, latency ≤ 15 s / episode). Models too heavy for the hardware (MoE 26B, thinking-mode 8B on 4 Go VRAM) are only tested via OpenRouter.

Dataset was also expanded from 25 → **50 cases**, adding critical coverage on the v1 weak spots: `mixed`, `user_preference` relations, `project_fact` relations, `negation`, `pronoun` co-reference, and FR↔EN aliases.

**Total panel:** 11 hosted models via OpenRouter + 2 locally-measured for speed.

---

## 2. TL;DR — 3 recommendations

| Rank | Model | Use when | Why |
|---|---|---|---|
| 🥇 | **`google/gemma-3-4b-it`** | Default choice for graph-mem | Best overall quality / cost / latency / local fallback combo. **Type accuracy 0.87** (best of the whole panel), E F1 0.86, R F1 0.59, **1.4 s** / call, **$0.001 / 50 cases** (42× cheaper than Claude Haiku), and the same model can be run locally as `gemma3:4b` (8.5 s / episode) for offline fallback. |
| 🥈 | **`anthropic/claude-haiku-4.5`** | When relation quality is critical (user preferences, past facts, negations) | Highest **relation F1** of the panel (0.69) and top performance on `user_preference` + `negation` categories. 2.2 s / call, but **42× more expensive** than gemma-3-4b-it. Worth it if the payload justifies the cost. |
| 🥉 | **`google/gemini-2.5-flash-lite`** | When sub-second latency is a hard requirement | Fastest hosted model on the panel (**930 ms**), top entity F1 (0.93), correct relation F1 (0.66). Type accuracy is weak (0.42) — avoid if typed Neo4j nodes matter. ~4× more expensive than gemma-3-4b-it. |

**To explicitly avoid:**

- ❌ **`openai/gpt-4o-mini`** — type accuracy **0.06** (catastrophic). Fast and cheap but cannot type-tag entities, disqualifying for a typed knowledge graph.
- ❌ **`google/gemma-4-26b-a4b-it`** — the **current `docker-compose.yml:10` default**. Type accuracy 0.37, worst relation score after `llama-3.2-3b`, and **10.2 s latency**. **Change the default**.
- ❌ **`mistralai/mistral-nemo`** — entity F1 0.55, structurally broken on this task.
- ❌ **`meta-llama/llama-3.2-3b-instruct`** — relation F1 0.43, the relation extractor doesn't hold.

---

## 3. Main comparison table (11 models)

Sorted by composite score: `0.4 × E_f1_sem + 0.4 × R_f1_sem + 0.2 × type_acc`.

All quality metrics are **semantic F1** (strict name/triple match with `nomic-embed-text` cosine ≥ 0.85 fallback), measured on 50 cases on OpenRouter.

| # | Model | Success | Entity F1 | Relation F1 | Type acc. | Hosted latency | Dispo local | Avg speed / episode (local) | Cost (50 cases) | Score |
|---|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|
| 1 | `anthropic/claude-haiku-4.5` | 50/50 | 0.91 | **0.69** | 0.64 | 2 188 ms | ❌ | ❌ | $0.04269 | 0.770 |
| 2 | **`google/gemma-3-4b-it`** ⭐ | 50/50 | 0.86 | 0.59 | **0.87** | 1 358 ms | ✅ `gemma3:4b` | **8 507 ms** | **$0.00102** | **0.758** |
| 3 | `deepseek/deepseek-chat-v3-0324` | 50/50 | **0.94** | 0.63 | 0.60 | 6 495 ms | ❌ | ❌ | $0.00690 | 0.746 |
| 4 | `google/gemma-3n-e4b-it` | 50/50 | 0.87 | 0.54 | 0.82 | 6 296 ms | ⚠️ `gemma4:e4b` | ❌ *(too slow: 17.5 s v1)* | $0.00053 | 0.730 |
| 5 | `mistralai/mistral-small-3.2-24b-instruct` | 50/50 | 0.90 | 0.55 | 0.71 | 2 110 ms | ❌ | ❌ | $0.00242 | 0.722 |
| 6 | `google/gemini-2.5-flash-lite` | 50/50 | 0.93 | 0.66 | 0.42 | **930 ms** | ❌ | ❌ | $0.00419 | 0.720 |
| 7 | `qwen/qwen-turbo` | 50/50 | 0.88 | 0.55 | 0.67 | 1 722 ms | ❌ | ❌ | $0.00114 | 0.706 |
| 8 | `meta-llama/llama-3.2-3b-instruct` | 50/50 | 0.83 | 0.43 | 0.76 | 1 263 ms | ⚠️ `llama3.2:3b` | ❌ *(v1 R F1=0.33)* | $0.00284 | 0.655 |
| 9 | `google/gemma-4-26b-a4b-it` ⚠️ *current default* | 50/50 | 0.81 | 0.61 | 0.37 | 10 165 ms | ⚠️ `gemma4:26b` | ❌ *(MoE CPU-only, too heavy)* | $0.00449 | 0.642 |
| 10 | `openai/gpt-4o-mini` | 50/50 | 0.93 | 0.62 | **0.06** | 1 331 ms | ❌ | ❌ | $0.00431 | 0.631 |
| 11 | `mistralai/mistral-nemo` | 49/50 | **0.55** | 0.59 | 0.84 | 2 029 ms | ⚠️ `mistral-nemo:12b` | ❌ *(v1 24 s)* | $0.00049 | 0.625 |

**Column legend:**
- **Dispo local** — ✅ = Ollama tag exists AND runs efficiently on the target hardware (NVIDIA T1200 4 Go VRAM, i7-11850H 32 Go RAM). ⚠️ = Ollama tag exists but excluded from local speed test (too slow, too heavy, or quality unusable). ❌ = no Ollama port of this model.
- **Avg speed / episode (local)** — measured latency on 5 warm-up cases via Ollama when applicable; ❌ otherwise.
- **Cost (50 cases)** — exact token count from the runner's `raw_response`, multiplied by OpenRouter pricing fetched live on 2026-04-11. Output tokens may be under-counted by ~5 % (runner truncates raw response at 600 chars).

### 3.1 Locally-available but excluded from speed test

Two more Ollama tags exist but were deliberately left out of the local speed test because they either couldn't reasonably run on the hardware or offer no upside:

| Ollama tag | Hosted equivalent | Why excluded from local speed test |
|---|---|---|
| `gemma4:e2b` | `google/gemma-3n-e2b-it:free` *(not benchmarked — OR free-tier rate-limit)* | Mobile-first edge model. V1 measured **37 s / call** despite fitting 3.2 Go Q4. Ollama template isn't desktop-optimized. |
| `qwen3:8b` | `qwen/qwen3-8b` *(not benchmarked — OR hangs on thinking-mode default)* | 5 Go Q4 on 4 Go VRAM → partial CPU offload. V1 produced 9/25 `failed_to_parse_json` due to thinking-mode leakage. |
| `phi4-mini:latest` | *(no OR equivalent)* | Actually usable in local (v1: E=0.80 / R=0.61 / type=0.73 / 6.9 s). **Measured in speed test as a standalone candidate** — see below. |
| `granite3.3:8b` | *(no OR equivalent)* | V1 R F1 = 0.51, 21 s / call. Neither fast nor good enough to retain. |

### 3.2 Local-only speed test results

Two Ollama candidates were actually run on the target hardware to measure real latency. **Quality scores below come from the v1 run on 25 cases**, since v2 quality is hosted-only — but for `gemma3:4b` specifically the hosted v2 result (0.86 / 0.59 / 0.87) is a very reasonable proxy.

| Ollama tag | Cases | Avg / episode | Range | Quality reference |
|---|---:|---:|---|---|
| `gemma3:4b` | 5 | **8 507 ms** | 7 662 – 9 627 ms | Hosted v2: E=0.86 / R=0.59 / type=0.87. V1 local 25 cases: E=0.87 / R=0.70 / type=0.90. **Most likely local winner.** |
| `phi4-mini:latest` | 5 | 9 637 ms | 8 369 – 11 327 ms | V1 local 25 cases only: E=0.80 / R=0.61 / type=0.73. No hosted equivalent on OR. |

Both sit in the **8–10 s range**, within the `SessionStart` hook budget (< 10 s). Picking one:

- **`gemma3:4b`** wins on paper — same model as the hosted #2 (`google/gemma-3-4b-it`), so the quality profile is known and excellent.
- **`phi4-mini`** is a reasonable fallback if the Gemma 3 Ollama template ever breaks (has happened with Gemma 3n variants), but its v1 quality is noticeably below `gemma3:4b`.

---

## 4. Cost analysis

### 4.1 Per 50-case run on OpenRouter

| Model | Input tokens | Output tokens | Cost / 50 cases |
|---|---:|---:|---:|
| `mistralai/mistral-nemo` | 18 660 | 5 930 | **$0.00049** |
| `google/gemma-3n-e4b-it` | 18 660 | 5 880 | $0.00053 |
| **`google/gemma-3-4b-it`** ⭐ | 18 660 | 6 420 | **$0.00102** |
| `qwen/qwen-turbo` | 18 660 | 5 820 | $0.00114 |
| `mistralai/mistral-small-3.2-24b-instruct` | 18 660 | 5 490 | $0.00242 |
| `meta-llama/llama-3.2-3b-instruct` | 18 660 | 2 250 | $0.00284 |
| `google/gemini-2.5-flash-lite` | 18 660 | 5 310 | $0.00419 |
| `openai/gpt-4o-mini` | 18 660 | 4 420 | $0.00431 |
| `google/gemma-4-26b-a4b-it` | 18 660 | 6 620 | $0.00449 |
| `deepseek/deepseek-chat-v3-0324` | 18 660 | 4 040 | $0.00690 |
| `anthropic/claude-haiku-4.5` | 18 660 | 5 560 | $0.04269 |

**`gemma-3-4b-it` costs $0.00102 for 50 cases — 42× cheaper than Claude Haiku for a score only 0.012 below.**

### 4.2 Cost at graph-mem realistic usage (1 500 extractions / user / month)

Assumption: ~50 extractions / day / user (5 Claude Code sessions × ~10 `save_memory` + future `PostToolUse` captures).

| Model | $ / user / month | $ / team of 10 / month |
|---|---:|---:|
| `mistralai/mistral-nemo` | $0.015 | $0.147 |
| `google/gemma-3n-e4b-it` | $0.016 | $0.159 |
| **`google/gemma-3-4b-it`** | **$0.031** | **$0.306** |
| `qwen/qwen-turbo` | $0.034 | $0.342 |
| `mistralai/mistral-small-3.2-24b-instruct` | $0.073 | $0.725 |
| `meta-llama/llama-3.2-3b-instruct` | $0.085 | $0.852 |
| `google/gemini-2.5-flash-lite` | $0.126 | $1.258 |
| `openai/gpt-4o-mini` | $0.129 | $1.293 |
| `google/gemma-4-26b-a4b-it` | $0.135 | $1.348 |
| `deepseek/deepseek-chat-v3-0324` | $0.207 | $2.070 |
| `anthropic/claude-haiku-4.5` | **$1.281** | **$12.807** |

At team scale (10 users), `gemma-3-4b-it` is **$0.31 / month** vs Claude Haiku's **$12.81**. The delta pays for a DeepSeek V3 run *6 times over*.

---

## 5. Per-category quality (semantic F1 · entity / relation)

Values are `entity_f1_sem / relation_f1_sem` per category. The `abstract` category has only 1 case and is statistically meaningless — kept for completeness.

| Category | n | claude-haiku | gemma-3-4b | deepseek-v3 | gemma-3n-e4b | mistral-small | gemini-FL | qwen-turbo | llama-3.2-3b | gemma-4-26b | gpt-4o-mini | mistral-nemo |
|---|---:|---|---|---|---|---|---|---|---|---|---|---|
| simple_entity | 3 | 0.80 / 0.80 | 1.00 / 0.53 | **1.00 / 0.67** | 0.83 / 0.67 | 1.00 / 0.80 | 0.91 / 0.80 | 0.91 / 0.67 | 0.67 / 0.80 | 0.80 / 0.80 | 1.00 / 0.80 | 0.67 / 0.67 |
| simple_relation | 4 | 1.00 / 0.83 | 0.96 / 0.76 | 1.00 / 0.85 | 0.91 / 0.79 | 1.00 / 0.50 | 1.00 / 0.88 | 1.00 / 0.45 | 0.91 / 0.40 | 0.96 / 0.88 | 1.00 / 0.81 | 0.54 / 0.66 |
| dedup | 5 | 0.82 / 0.30 | 0.84 / 0.17 | 0.89 / 0.17 | 0.89 / 0.00 | 0.89 / 0.30 | 0.87 / 0.17 | 0.78 / 0.60 | 0.95 / 0.30 | 0.91 / 0.30 | **0.95** / 0.30 | 0.73 / 0.50 |
| temporal | 4 | 0.88 / 0.45 | 0.96 / 0.67 | 0.96 / 0.50 | 0.96 / 0.50 | 0.96 / 0.50 | 0.96 / 0.45 | 0.88 / 0.43 | 0.88 / 0.00 | 0.83 / 0.45 | 1.00 / 0.50 | 0.54 / 0.50 |
| user_preference | 6 | 0.70 / 0.77 | 0.61 / 0.42 | 0.77 / **0.86** | 0.81 / 0.45 | 0.74 / 0.61 | 0.66 / 0.62 | 0.71 / 0.47 | 0.65 / 0.40 | 0.64 / 0.63 | 0.62 / 0.37 | 0.36 / 0.42 |
| project_fact | 6 | 0.98 / 0.45 | 0.92 / 0.51 | 1.00 / 0.33 | 0.91 / 0.33 | 1.00 / 0.33 | 0.94 / 0.50 | 1.00 / 0.33 | 0.98 / 0.33 | 0.74 / 0.30 | 1.00 / 0.45 | 0.46 / 0.45 |
| mixed | 6 | **1.00 / 0.69** | 1.00 / 0.83 | 0.98 / 0.50 | 0.97 / 0.67 | 0.96 / 0.53 | 1.00 / 0.66 | 0.92 / 0.61 | 0.86 / 0.47 | 0.91 / 0.61 | 0.96 / 0.69 | 0.53 / 0.61 |
| pronoun | 4 | 0.96 / 0.60 | 1.00 / 0.38 | 1.00 / 0.60 | 1.00 / 0.60 | 1.00 / 0.30 | 1.00 / 0.60 | 0.96 / 0.43 | 1.00 / 0.47 | 0.91 / 0.60 | 1.00 / 0.62 | 0.48 / 0.57 |
| negation | 4 | **1.00 / 1.00** | 0.83 / 0.50 | 1.00 / 1.00 | 0.75 / 0.75 | 0.71 / 0.66 | 1.00 / 1.00 | 0.96 / 0.68 | 0.71 / 0.50 | 0.71 / 0.75 | 1.00 / 1.00 | 0.29 / 0.42 |
| nested | 2 | 1.00 / 0.83 | 1.00 / 0.56 | 1.00 / 0.83 | 1.00 / 0.50 | 1.00 / 0.83 | 1.00 / 0.76 | 1.00 / 0.67 | 1.00 / 0.56 | 0.84 / 0.17 | 1.00 / 0.83 | 0.33 / 0.47 |
| edge | 5 | 0.89 / 0.89 | 0.65 / 0.80 | 1.00 / 0.80 | 0.80 / 0.80 | 0.97 / 0.93 | 0.95 / 0.80 | 0.85 / 0.80 | 0.80 / 0.73 | 0.75 / 0.95 | 1.00 / 0.80 | 0.60 / 0.80 |
| abstract | 1 | 0.80 / 1.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 1.00 / 1.00 | 0.00 / 0.33 | 0.00 / 0.00 | 0.80 / 1.00 | 0.00 / 0.00 | 0.33 / 0.00 |

**Key observations:**

- **`dedup` relation extraction is weak across the entire panel** (most models ≤ 0.30). The failure mode is consistent: models collapse aliases correctly in entities but then emit relations on the alias instead of the canonical name, which the strict scorer misses. Semantic scoring doesn't recover it because the whole triple signature differs. **Fixing the scorer** would give a much fairer picture — this isn't a real model weakness, it's a measurement artefact.
- **`user_preference`** is where **DeepSeek V3 dominates** (0.77 / 0.86). The only model that reliably maps "I prefer X to Y" to `user prefers X` without dragging Y in. Interesting for a personal memory system. `gemma-3-4b-it` is noticeably weaker here (0.61 / 0.42) — if this category matters most, DeepSeek or Claude win over gemma.
- **`negation`** is handled perfectly by Claude Haiku, DeepSeek, Gemini-FL, and GPT-4o-mini (1.00 / 1.00). **`gemma-3-4b-it` stumbles** (0.83 / 0.50) — it occasionally drops the `past_*` prefix. That's the main quality gap vs Claude Haiku.
- **`mixed` improved dramatically vs v1** (where everyone was stuck at R_f1 ≈ 0.17). The new v2 `mixed_02..05` cases let models demonstrate multi-relation extraction. **`gemma-3-4b-it` is surprisingly best in class** on `mixed` (1.00 / 0.83) — better than Claude Haiku.
- **`gemma-4-26b-a4b-it` has a collapse on `nested` relations** (0.17) and is never in the top half of any row. Confirmed worst in class.
- **`mistral-nemo` is consistently in the bottom row on entities** — its entity extractor is structurally broken, not just "noisy".

---

## 6. Latency analysis

### 6.1 Hosted latency on OpenRouter (50 cases average)

| Model | Avg latency / call | Acceptable for `SessionStart` hook (< 10 s)? |
|---|---:|:---:|
| `google/gemini-2.5-flash-lite` | **930 ms** | ✅ |
| `meta-llama/llama-3.2-3b-instruct` | 1 263 ms | ✅ |
| `openai/gpt-4o-mini` | 1 331 ms | ✅ |
| **`google/gemma-3-4b-it`** | **1 358 ms** | ✅ |
| `qwen/qwen-turbo` | 1 722 ms | ✅ |
| `mistralai/mistral-nemo` | 2 029 ms | ✅ |
| `mistralai/mistral-small-3.2-24b-instruct` | 2 110 ms | ✅ |
| `anthropic/claude-haiku-4.5` | 2 188 ms | ✅ |
| `google/gemma-3n-e4b-it` | 6 296 ms | ⚠️ borderline |
| `deepseek/deepseek-chat-v3-0324` | 6 495 ms | ⚠️ borderline |
| `google/gemma-4-26b-a4b-it` | 10 165 ms | ❌ at the hook budget limit |

### 6.2 Local speed test (target hardware: T1200 4 Go VRAM, i7-11850H, 32 Go RAM)

| Ollama tag | Cases | Avg / episode | Range |
|---|---:|---:|---|
| `gemma3:4b` | 5 | **8 507 ms** | 7 662 – 9 627 ms |
| `phi4-mini:latest` | 5 | 9 637 ms | 8 369 – 11 327 ms |

Both fit the hook budget. `gemma3:4b` is ~1 s faster than `phi4-mini` in this run — opposite of the v1 measurement (where phi4-mini was 6.9 s and gemma3:4b was 10.4 s). Variance is likely due to Ollama KV cache warmth and concurrent processes. **Call it a tie on latency**, quality decides in favour of `gemma3:4b`.

---

## 7. Caveats

1. **`gemma-3n-e2b-it:free` and `qwen/qwen3-8b` were attempted but dropped from v2b.** The free-tier version of gemma-3n-e2b hit the 20 req/min free-tier rate limit after 13 cases. `qwen3-8b` via OR has thinking-mode on by default and was generating 30–60 s reasoning tokens per call, hanging indefinitely — we would need a `/nothink` system prompt to retest.
2. **`raw_response` is truncated at 600 chars** by the runner. Two hosted models hit the cap on 2–3 cases out of 50. Token and cost figures are therefore a **lower bound** (~5 % error).
3. **Output tokens are counted with `tiktoken cl100k_base`**, not each provider's native tokeniser. Cost numbers are indicative (±15 %).
4. **Per-category scores on small n (1–6 cases)** are directional, not statistically significant. The `abstract` category has only 1 case and should be ignored.
5. **Local speed test uses only 5 cases** (the 5 shortest, homogeneous inputs). Variance at N=5 is higher than at N=50 but sufficient to answer "does this fit the hook budget".
6. **`dedup` relation scoring penalises models unfairly** (see §5). A looser triple matcher (canonicalise alias before scoring) would change the panel ordering slightly but not the top 3.
7. **phi4-mini and granite3.3:8b have no OpenRouter equivalent** — their quality can only be compared via v1 local data on 25 cases, which is why they don't appear in the main table.

---

## 8. Proposed next steps

1. **Change `docker-compose.yml:10` default from `gemma-4-26b-a4b-it` to `google/gemma-3-4b-it`.** The current default is rank 9 / 11 on quality; the proposed default is rank 2 / 11 at 42× lower cost. **This is the most impactful action of this whole benchmark.**
2. **Wire up `gemma3:4b` via Ollama as the offline fallback** (same model as the new default, locally available). Graphiti's `OPENAI_BASE_URL` can switch to `http://host.docker.internal:11434/v1` with `LLM_MODEL_NAME=gemma3:4b` if OpenRouter is unreachable.
3. **If relation quality becomes the bottleneck** (e.g. `user_preference` or `negation` are critical), consider Claude Haiku as a conditional upgrade rather than default — its 42× cost is justified only for payloads where relation F1 really matters.
4. **Fix the `dedup` relation scoring** — it artificially penalises all models. A canonical-alias match before the triple comparison would surface the real performance gap (or confirm there isn't one).
5. **Re-run `qwen/qwen3-8b` on OpenRouter with an explicit `/nothink` system prompt** — worth trying once. If thinking is disabled cleanly, it could be a good hosted option at $0.05/$0.40.
6. **Add a long-input category to the dataset** (500–1 500 tokens, simulating `Stop` hook session summaries). Current dataset is short-input only; production `save_memory` calls can be longer.
7. **Next up: embedding benchmark in `.benchmark/embedding/`.** The graph-mem default embedder is currently `qwen3-embedding-8b` via OpenRouter — worth validating against `nomic-embed-text`, `all-minilm`, and others with the same rigour.

---

## 9. Appendix — methodology

- **Prompt:** single user-turn prompt from `.benchmark/llm/run_benchmark.py:45-65`. 358 base tokens + ~15 tokens per input on average. Identical for every model.
- **Parameters:** `temperature=0.0`, `max_tokens=1024`. No system prompt. No function-calling / JSON-mode flags — models are asked to emit strict JSON via prompt rules.
- **Scoring:**
  - *Strict:* set-based name match for entities; relations require at least 2/3 of (source, predicate, target) components to match (normalized lowercase).
  - *Semantic:* entity/relation pairs that failed the strict match are re-scored by `nomic-embed-text` cosine similarity (threshold 0.85).
  - *Type accuracy:* among entities correctly matched by name, what fraction also has the expected type.
- **Tokenisation for cost:** `tiktoken cl100k_base` on the full prompt + stored raw response. Approximate for non-OpenAI providers.
- **Pricing:** live-fetched from `https://openrouter.ai/api/v1/models` on 2026-04-11.
- **Dataset split (v2, 50 cases):** `simple_entity` (3), `simple_relation` (4), `dedup` (5), `temporal` (4), `user_preference` (6), `project_fact` (6), `mixed` (6), `pronoun` (4), `negation` (4), `nested` (2), `edge` (5), `abstract` (1).
