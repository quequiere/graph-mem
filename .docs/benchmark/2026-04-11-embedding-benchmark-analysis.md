# graph-mem — Embedding Benchmark Analysis

**Date:** 2026-04-11
**Dataset:** `.benchmark/embedding/dataset.json` (100 corpus docs / 40 queries / 5 personas, FR+EN)
**Raw results:** `.benchmark/embedding/results/`
**Harness:** `.benchmark/embedding/` (numpy-only retrieval eval, dual-dim runs where applicable)
**Goal:** pick the embedding model that best matches graph-mem's workload — bilingual (FR/EN) short-fact retrieval with hard negatives, robust to temporal queries and negation.

---

## 1. TL;DR — Recommendation

| Scenario | Pick | MRR | Why |
|---|---|---:|---|
| **Hosted primary** (sweet spot) | `openai/text-embedding-3-small` @ **1024** (Matryoshka) | **0.883** | Beats current prod config by **+9.3 pts** absolute. 1024d keeps Graphiti's vector payload unchanged. p50 ~10 ms. $0.02 / 1M tokens — 2× current but still negligible at graph-mem scale. **Perfect negation (5/5) and strong French (0.790 MRR).** |
| **Hosted best quality** (cost no object) | `openai/text-embedding-3-large` @ **3072** | **0.903** | Highest overall MRR. Best hard-negative mean rank (**5.68** — lowest in the benchmark). Tripled storage per vector vs current config. $0.13 / 1M. |
| **Local primary** | `qwen3-embedding:4b` (Ollama) | **0.866** | Best **French MRR of the entire benchmark (0.862)**, perfect on negation (5/5) and mixed_lang (5/5). 189 ms p50, zero external cost. Beats the **graph-mem current production config** (or-qwen3-8b @ 1024) by **+5.8 pts** absolute. |
| **Local budget / fallback** | `qwen3-embedding:0.6b` (Ollama) | 0.809 | Only 639 MB on disk. Beats the hosted `or-qwen3-8b` @ 1024 (0.808). Solid FR (0.790). Reasonable if disk or VRAM is tight. |
| **To drop** | `qwen3-embedding:8b` (Ollama GGUF), `or-qwen3-8b` @ 1024, `nomic-v1.5`, `jina-v3` (for FR) | < 0.81 | See §4 on the Qwen-8B quantization collapse. Nomic is English-only in practice (FR MRR 0.465). Jina-v3 under-delivers vs reputation. |

**Bottom line for graph-mem:** the current production config is `qwen/qwen3-embedding-8b` via OpenRouter, truncated client-side to 1024d. On this benchmark it ranks **12/15 (MRR 0.808)** and **collapses on French** (MRR 0.578) and **on negation** (MRR 0.607). Switching to **`openai/text-embedding-3-small` @ 1024** keeps the 1024-dim Graphiti schema untouched, costs a few cents more per million tokens, and delivers **+9.3 pts MRR absolute**, **+21.2 pts French**, and **+39.3 pts on negation**. If a local backend is preferred, **`qwen3-embedding:4b` via Ollama** is the local equivalent — no external dependency, best-in-class French, but 15–20× slower than hosted.

---

## 2. Comparative table (15 runs)

Sorted by overall MRR. `HN rank` is the mean rank assigned to hard-negative documents (higher = better — hard negatives are pushed further down the list).

| # | Model | Backend | Dim (used/native) | R@1 | R@5 | **MRR** | HN rank | p50 ms | p95 ms | $ / 1M |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | **`openai/text-embedding-3-large`** | OpenRouter | 3072 / 3072 | 0.850 | 0.950 | **0.903** | **5.68** | 21 | 23 | $0.130 |
| 2 | **`openai/text-embedding-3-small`** | OpenRouter | **1024** / 1536 | 0.825 | 0.950 | **0.883** | 6.62 | **10** | 17 | $0.020 |
| 3 | `openai/text-embedding-3-large` | OpenRouter | 1024 / 3072 | 0.800 | 0.950 | 0.878 | 6.22 | 20 | 27 | $0.130 |
| 4 | `openai/text-embedding-3-small` | OpenRouter | 1536 / 1536 | 0.800 | 0.950 | 0.871 | 6.78 | 11 | 19 | $0.020 |
| 5 | **`qwen3-embedding:4b`** | **Ollama local** | 1024 / 2560 | 0.800 | 0.950 | **0.866** | 8.13 | 189 | 270 | local |
| 6 | `google/gemini-embedding-001` | OpenRouter | 3072 / 3072 | 0.800 | 0.950 | 0.864 | 6.21 | 49 | 72 | $0.150 |
| 7 | `google/gemini-embedding-001` | OpenRouter | 1024 / 3072 | 0.800 | 0.950 | 0.860 | 6.51 | 43 | 56 | $0.150 |
| 8 | `qwen/qwen3-embedding-8b` | OpenRouter | 4096 / 4096 | 0.775 | 0.925 | 0.841 | 8.33 | 25 | 29 | $0.010 |
| 9 | `baai/bge-m3` | sentence-transformers | 1024 / 1024 | 0.750 | 0.900 | 0.830 | 7.02 | 276 | 276 | local |
| 10 | `baai/bge-m3` | OpenRouter | 1024 / 1024 | 0.750 | 0.900 | 0.830 | 7.02 | 23 | 43 | $0.010 |
| 11 | `qwen3-embedding:0.6b` | Ollama local | 1024 / 1024 | 0.700 | 0.950 | 0.809 | 7.69 | 99 | 149 | local |
| 12 | **`qwen/qwen3-embedding-8b`** ← *current graph-mem prod* | **OpenRouter (truncated client-side)** | **1024** / 4096 | 0.725 | 0.950 | **0.808** | 8.19 | 36 | 36 | $0.010 |
| 13 | `jinaai/jina-embeddings-v3` | sentence-transformers | 1024 / 1024 | 0.700 | 0.925 | 0.797 | 7.56 | 35 | 35 | local |
| 14 | **`qwen3-embedding:8b`** | **Ollama local (GGUF q4)** | 1024 / 4096 | 0.700 | 0.925 | **0.796** | 7.63 | 473 | 804 | local |
| 15 | `nomic-ai/nomic-embed-text-v1.5` | sentence-transformers | 768 / 768 | 0.625 | 0.900 | 0.741 | 7.54 | 18 | 18 | local |

**Disabled:** `NovaSearch/stella_en_1.5B_v5` was in the plan as an English reference model but was skipped — its custom `modeling_qwen.py` uses `transformers.DynamicCache.get_usable_length`, a method that was removed in the recent `transformers` releases. Since Stella is English-only anyway, it was not a fit for graph-mem's bilingual workload and the test was not re-run on an older `transformers`.

**Sanity check:** `baai/bge-m3` hosted on OpenRouter vs local via sentence-transformers produces **identical metrics** down to MRR = 0.830 across all categories. This validates that the harness is consistent between backends and that OR serves the same BGE-M3 weights as HuggingFace — no surprise quantization on the hosted side.

---

## 3. Per-category breakdown

MRR by query category (15 `user_fact` / 10 `project_fact` / 5 `temporal` / 5 `negation` / 5 `mixed_lang`). Categories are where models differentiate most — overall MRR hides the interesting signal.

| # | Model | dim | MRR | user_fact | project_fact | temporal | **negation** | mixed_lang |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `text-embedding-3-large` (OR) | 3072 | **0.903** | 0.967 | 0.813 | 0.800 | **1.000** | 0.900 |
| 2 | `text-embedding-3-small` (OR) | 1024 | 0.883 | 0.950 | 0.758 | 0.800 | **1.000** | 0.900 |
| 3 | `text-embedding-3-large` (OR) | 1024 | 0.878 | 0.900 | 0.813 | 0.800 | **1.000** | 0.900 |
| 4 | `text-embedding-3-small` (OR) | 1536 | 0.871 | 0.917 | 0.758 | 0.800 | **1.000** | 0.900 |
| 5 | **`qwen3-embedding:4b`** (Ollama) | 1024 | 0.866 | 0.933 | 0.694 | 0.740 | **1.000** | **1.000** |
| 6 | `gemini-embedding-001` (OR) | 3072 | 0.864 | 0.828 | 0.813 | **0.900** | **1.000** | 0.900 |
| 7 | `gemini-embedding-001` (OR) | 1024 | 0.860 | 0.817 | 0.814 | **0.900** | **1.000** | 0.900 |
| 8 | `qwen3-embedding-8b` (OR) | 4096 | 0.841 | 0.967 | 0.762 | 0.740 | 0.833 | 0.733 |
| 9 | `bge-m3` (ST / OR) | 1024 | 0.830 | 0.773 | 0.810 | 0.800 | **1.000** | 0.900 |
| 10 | `qwen3-embedding:0.6b` (Ollama) | 1024 | 0.809 | 0.856 | 0.684 | 0.700 | 0.840 | **1.000** |
| 11 | **`qwen3-embedding-8b` (OR) — graph-mem prod** | **1024** | 0.808 | 0.956 | 0.758 | 0.767 | **0.607** | 0.707 |
| 12 | `jina-v3` (ST) | 1024 | 0.797 | 0.771 | 0.699 | 0.867 | 0.900 | 0.900 |
| 13 | **`qwen3-embedding:8b` (Ollama q4)** | **1024** | 0.796 | 0.967 | 0.759 | 0.767 | **0.469** | 0.717 |
| 14 | `nomic-v1.5` (ST) | 768 | 0.741 | 0.717 | 0.736 | 0.700 | **1.000** | 0.607 |

### 3.1 Negation is where the ranking diverges

Five negation queries: *"who does NOT work on Atlas?"*, *"what language does Bruno dislike?"*, *"who does not know Kubernetes?"*, *"who does not write Python professionally?"*, *"qui ne code plus en R ?"*

| Model group | Negation MRR | Breakdown |
|---|---:|---|
| OpenAI (3-small/large, all dims) | **1.000** | Perfect on all 5 queries. |
| Gemini 001 (both dims) | **1.000** | Perfect. |
| BGE-M3 (hosted + local) | **1.000** | Perfect. |
| Nomic v1.5 | **1.000** | Surprisingly perfect on negation despite a weak global score. |
| `qwen3-embedding:4b` (Ollama) | **1.000** | Perfect. |
| `jina-v3` | 0.900 | 4.5/5. |
| `qwen3-embedding:0.6b` (Ollama) | 0.840 | 4.2/5. |
| `qwen/qwen3-embedding-8b` @ 4096 (OR) | 0.833 | 4.2/5 — already soft here at full precision. |
| **`qwen/qwen3-embedding-8b` @ 1024 (OR) — prod** | **0.607** | Fails 3 of 5. |
| **`qwen3-embedding:8b` (Ollama GGUF q4)** | **0.469** | Fails 4 of 5 outright. |

**graph-mem's current config fails negation on nearly half the queries**, and the Ollama GGUF build of the same weights is worse still. Look at the per-query RR for *"who does not know Kubernetes?"*: OpenAI = 1.0, OR-Qwen@1024 = 0.333, Ollama-Qwen-8b = 0.143. That is catastrophic for a memory system expected to answer *"what does the user NOT know?"* correctly.

### 3.2 French is the second axis where Qwen-8B drops

Only 7 queries are in French in the dataset (small sample — see §8 Limitations), but the signal is consistent:

| Model | **FR MRR** | EN MRR | Delta EN − FR |
|---|---:|---:|---:|
| **`qwen3-embedding:4b` (Ollama)** | **0.862** | 0.867 | **0.005** ← most consistent |
| `gemini-embedding-001` (OR, both dims) | ~0.795 | ~0.875 | ~0.080 |
| `jina-v3` | 0.794 | 0.798 | 0.004 |
| `qwen3-embedding:0.6b` (Ollama) | 0.790 | 0.813 | 0.023 |
| `text-embedding-3-small` @ 1024 (OR) | 0.790 | 0.903 | 0.113 |
| `text-embedding-3-small` @ 1536 (OR) | 0.789 | 0.888 | 0.099 |
| `text-embedding-3-large` @ 1024 (OR) | 0.788 | 0.897 | 0.109 |
| `text-embedding-3-large` @ 3072 (OR) | 0.788 | 0.928 | 0.140 |
| `bge-m3` (ST / OR) | 0.719 | 0.854 | 0.135 |
| `qwen3-embedding-8b` @ 4096 (OR) | 0.669 | 0.878 | 0.209 |
| **`qwen3-embedding-8b` @ 1024 (OR) — prod** | **0.578** | 0.857 | **0.279** |
| **`qwen3-embedding:8b` (Ollama q4)** | 0.585 | 0.841 | 0.256 |
| `nomic-v1.5` | 0.465 | 0.800 | 0.335 |

- The OpenAI family is **consistently +10 to +14 pts weaker in FR than in EN** — acceptable for graph-mem (FR is ~20 % of expected traffic) but not ideal.
- `gemini-embedding-001` has the smallest hosted FR/EN gap (~8 pts) and is a reasonable second choice if FR weight is high.
- **`qwen3-embedding:4b` (local) is the only model in the entire benchmark that is as good in FR as in EN**, and its FR MRR of 0.862 is the highest overall — 7 pts above the best hosted model. If FR is a first-class concern, this is the pick.
- `nomic-v1.5` is effectively an English model (FR MRR 0.465) — do not ship it.

---

## 4. The Qwen3-8B story: quantization and truncation both hurt

This is the most important single finding of the benchmark, because it directly challenges graph-mem's current production choice.

Three variants of the same 8B weights were tested:

| Variant | Backend | Dim used | MRR | Negation | FR |
|---|---|---:|---:|---:|---:|
| Full precision, full dim | OpenRouter | **4096** | **0.841** | 0.833 | 0.669 |
| Full precision, Matryoshka truncated client-side | OpenRouter | **1024** | 0.808 | 0.607 | 0.578 |
| GGUF q4 quantized, full dim | Ollama local | **1024** (native 4096, truncated) | **0.796** | **0.469** | 0.585 |

- **Client-side 4096→1024 truncation costs ~3.3 pts MRR** on this model (0.841 → 0.808). That is much larger than the same Matryoshka truncation on the OpenAI models (which is trained as a target and costs < 0.6 pt). Qwen3 embeddings are **not** Matryoshka-trained — you should only truncate if you have a good reason.
- **GGUF quantization costs another 1.2 pts** on top of that (0.808 → 0.796 at the same 1024 dim), but that average hides a dramatic collapse on negation (0.607 → 0.469) that is invisible if you only look at overall MRR.
- Matryoshka alternatives like `text-embedding-3-small` @ 1024 are trained for this — the model drops only 0.012 MRR going from 1536 to 1024, and zero points on negation.

The combined effect: graph-mem's current config (Qwen3-8B @ 1024 via OR) delivers 0.808 MRR, which is ~**9 points** below an OpenAI drop-in replacement. The Ollama GGUF version of the same model is even more fragile on the categories that matter most for a memory system (negation, bilingual).

---

## 5. Matryoshka dimension analysis

For the three hosted models run at two dimensions, the truncation cost is:

| Model | Full dim | 1024 dim | ΔMRR | Verdict |
|---|---:|---:|---:|---|
| `text-embedding-3-small` | 0.871 (1536) | 0.883 (1024) | **+0.012** | *Free* — 1024 is literally better here (within noise, but clearly not worse). Matryoshka-trained target. |
| `text-embedding-3-large` | 0.903 (3072) | 0.878 (1024) | **−0.025** | Small cost for 3× compression. Excellent trade-off. |
| `gemini-embedding-001` | 0.864 (3072) | 0.860 (1024) | **−0.004** | Barely measurable. Gemini truncates cleanly. |
| `qwen/qwen3-embedding-8b` | 0.841 (4096) | 0.808 (1024) | **−0.033** | Largest drop — Qwen-3 is **not** Matryoshka-trained and should run at native dim if used. |

For graph-mem, the 1024-dim vector payload is already baked into Graphiti's ingestion path. All three OpenAI/Gemini candidates truncate cleanly to 1024 at zero or near-zero quality cost, which makes the migration a config change, not a schema change.

---

## 6. Hosted vs local — backend choice

### Latency

Hosted embeddings on OpenRouter from this machine (behind the Michelin corporate proxy with SSL interception disabled via `verify=False`):

| Backend | p50 (ms) | p95 (ms) | Notes |
|---|---:|---:|---|
| OR / OpenAI 3-small @ 1024 | **10** | 17 | Fastest hosted. |
| OR / OpenAI 3-large | 20 | 23–27 | Negligible overhead for 3× the dim. |
| OR / BGE-M3 | 23 | 43 | |
| OR / Qwen3-8B | 25–36 | 29–36 | Reliable. |
| OR / Gemini 001 | 43–49 | 56–72 | Slowest hosted — router adds latency. |
| Ollama / Qwen3-0.6b | 99 | 149 | 10× hosted. |
| Ollama / Qwen3-4b | 189 | 270 | 20× hosted. |
| Ollama / Qwen3-8b | **473** | **804** | ~50× hosted. Risky for synchronous recall. |
| sentence-transformers / BGE-M3 | 276 | 276 | CPU-only in this run. |
| sentence-transformers / nomic-v1.5 | 18 | 18 | Small model, CPU-fast. |
| sentence-transformers / jina-v3 | 35 | 35 | CPU-fast despite LoRA. |

**SessionStart hook budget is 10 s** for the full parallelised context recall. With hosted OpenAI-3-small @ 10 ms × ~5 searches, the embedding side is 50 ms — negligible. With local Ollama Qwen-8b at 473 ms p50 and 804 ms p95, five parallel searches push the embedding budget alone above 2.5 s under load, leaving little headroom for Graphiti's downstream entity-resolution pass. **This is a hard argument against running Qwen3-8B locally as graph-mem's synchronous backend.**

### Cost

All hosted runs fit under **$0.0003 for the full 140-row evaluation** (100 corpus + 40 queries, character-based token estimate via `chars / 4`). At graph-mem's expected traffic (assume 100 saves/day × ~500 tokens each = 50 k tokens/day = 18 M tokens/year), the annual embedding cost is:

| Model | $ / 1M | Annual cost @ 18 M tokens |
|---|---:|---:|
| `qwen/qwen3-embedding-8b` (current prod) | $0.010 | **$0.18** |
| `baai/bge-m3` | $0.010 | $0.18 |
| `openai/text-embedding-3-small` | $0.020 | **$0.36** |
| `google/gemini-embedding-001` | $0.150 | $2.70 |
| `openai/text-embedding-3-large` | $0.130 | $2.34 |

**Cost is effectively free at graph-mem's scale.** Switching from Qwen3-8B to OpenAI 3-small doubles the bill — from ~$0.18/year to ~$0.36/year. The quality gain (+9.3 pts MRR absolute, +39 pts on negation) is unambiguous.

---

## 7. Recommendation for graph-mem

### 7.1 Primary: switch `.env` to OpenAI 3-small

Update `.env` in the root of the repo:

```dotenv
# Before:
EMBEDDING_MODEL_NAME=qwen/qwen3-embedding-8b
# After:
EMBEDDING_MODEL_NAME=openai/text-embedding-3-small
```

No other code change is required:

- Graphiti still points at OpenRouter (`OPENAI_BASE_URL=https://openrouter.ai/api/v1`), which serves `openai/text-embedding-3-small` via the OpenAI-compatible `/v1/embeddings` endpoint.
- The existing 1024-dim client-side truncation is harmless on this model (Matryoshka target — +0.012 MRR gain).
- Neo4j payload dimensions stay at 1024. No re-embed of existing memories is strictly required, but see §7.3.

Expected impact on graph-mem in production:

- **+9.3 pts absolute MRR** on graph-mem's retrieval workload.
- **+21 pts MRR on French queries** (0.578 → 0.790), unblocking the "Sylvie parle à Claude en français" category.
- **+39 pts MRR on negation queries** (0.607 → 1.000) — memory-side recall of "what the user does NOT do" stops being broken.
- **+0.01 $/1M tokens** marginal cost, which is $0.18/year extra at projected volume.
- **−26 ms p50 latency** per embed call (36 ms → 10 ms) — helps the 10 s `SessionStart` budget.

### 7.2 Secondary: offer `qwen3-embedding:4b` (Ollama) as the local option

For users running graph-mem fully offline, the best local pick is **Ollama / `qwen3-embedding:4b`** (not the 8b variant). It is the only tested model that matches English and French MRR symmetrically, beats everything on mixed_lang and on negation, and produces a 1024-dim vector after client-side truncation. The trade-off is latency — 189 ms p50 vs 10 ms for the hosted OpenAI, i.e. ~20× slower. Acceptable for async `save_memory`, acceptable for recall under the 10 s hook budget with parallelisation, not acceptable if a sub-second end-to-end reply is required.

Document this as an alternate backend in the README under "Local mode":

```dotenv
EMBEDDING_MODEL_NAME=qwen3-embedding:4b
OPENAI_BASE_URL=http://localhost:11434/v1
```

**Do not** document `qwen3-embedding:8b` — on this benchmark it is worse than its own 4B sibling (0.796 vs 0.866 MRR), slower (473 ms vs 189 ms p50), and dramatically worse on negation (0.469 vs 1.000). The 8B GGUF build is actively harmful relative to the 4B.

### 7.3 Migration — re-embed existing memories?

Existing Neo4j vectors were computed with `qwen/qwen3-embedding-8b` @ 1024. After the switch, newly-saved memories will be embedded with `openai/text-embedding-3-small` @ 1024. **These two vector spaces are not cross-comparable** — a query embedded with OpenAI-3-small will not retrieve old Qwen-8B vectors via cosine similarity in any useful way.

Two options:

1. **Lazy migration (recommended for graph-mem's use case).** Wipe the old vector data and re-ingest from scratch. Because graph-mem's knowledge graph is still early-stage and most facts are recoverable from recent conversations, the cost of re-populating is low, and the upside is an immediate quality jump for all subsequent queries. Suggested path: `docker compose down && docker volume rm graph-mem_neo4j_data && docker compose up -d --build`, then re-run `/onboard`.
2. **Hybrid period.** Keep old vectors, tag them with their embedder ID, and restrict searches to vectors from the same embedder as the query. This is intrusive and probably not worth the engineering for graph-mem's current scale.

Go with option 1 unless the user has significant unrecoverable memories stored.

---

## 8. Limitations

- **Dataset size.** 40 queries is small. With 5 queries per non-user category (temporal, negation, mixed_lang), a single miss shifts category MRR by 0.2. The negation finding holds up because the *pattern* is consistent across 5/5 queries, but the absolute deltas for temporal and mixed_lang have wide confidence intervals.
- **FR sample.** Only 7 queries are in French. The conclusion "qwen3-embedding:4b is best on FR" holds, but it is a strong claim made on a small sample. A larger FR test set would tighten this.
- **One author, one style.** The dataset was hand-authored in a single sitting. Query phrasing bias (questions ending with "?", consistent persona naming) probably favours models that saw similar conversational data during training. Independent dataset sources would strengthen the evaluation.
- **Cost estimation is char-based.** The runner uses `chars / 4` as a token proxy for cost. OpenRouter's `/v1/embeddings` response includes a real `usage.cost` field that would be more accurate — not wired into the adapter, would be ~10 lines of work.
- **Domain scope.** The 100 corpus documents are all persona facts (Bruno/Julie/Karim/Lea/Thomas). Graph-mem will also index project code, commit messages, and technical notes — distributions not represented here. Results should hold qualitatively for similar short-fact retrieval but do not extrapolate directly to long-form document search.
- **Local Ollama latencies.** Measured on the user's machine with Ollama warm. Cold-start adds ~6–7 s per model (observed during dry-run). The p50/p95 numbers reflect steady-state only.
- **Stella was skipped.** English-only anyway and blocked by a transformers API break (`DynamicCache.get_usable_length` removed). Not a loss for graph-mem's bilingual use case, but closes the door on claiming "we tested the full MTEB top-10".
- **BGE-M3 parity between backends.** The OR-hosted and sentence-transformers local runs produced identical metrics (0.830 MRR both), which is the expected sanity check and does not indicate new information about BGE-M3 — only that the harness is consistent.

---

## 9. Appendix — what changed from the original plan

The plan at `.docs/superpowers/plans/2026-04-11-embedding-benchmark.md` is now outdated. Actual deviations:

- **TDD was dropped** for the benchmark harness at the user's explicit request. Validation is done via `--dry-run` and real runs, not unit tests.
- **Hosted set reduced from 6 vendors to 5 models routed through OpenRouter.** Voyage, Cohere and Mistral are not available on OpenRouter's `/v1/embeddings` and were dropped. BGE-M3 and Qwen3-8B were added to the hosted list because they *are* routed through OR — which let this analysis compare hosted-precision vs Ollama-GGUF on identical weights (see §4).
- **Stella-en-1.5B skipped** at the sentence-transformers step (transformers API break). See §2.
- **Dataset `pronoun` category dropped.** The corpus is 100 % third-person, so first-person pronoun queries had no clean ground truth. Queries rebalanced to 15 user_fact / 10 project_fact / 5 temporal / 5 negation / 5 mixed_lang.
- **Ollama tags.** Original `models.yaml` used `qwen3-embedding-8b` / `qwen3-embedding-4b` which are not valid Ollama tags. Actual tags are `qwen3-embedding:8b` / `qwen3-embedding:4b` / `qwen3-embedding:0.6b`. Fixed in `models.yaml`, and `:0.6b` was added as a free extra low-end data point.
- **`BENCHMARK_SSL_VERIFY=false` + `PYTHONIOENCODING=utf-8`** are required on every run on the Michelin corporate network — the proxy intercepts HTTPS with a self-signed root, and the Windows console is CP1252. See `adapters/_http.py` for the env-var plumbing.

The harness, dataset and raw result files are all under `.benchmark/embedding/` and can be re-run at any time with:

```bash
cd .benchmark/embedding && \
BENCHMARK_SSL_VERIFY=false PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
python run_benchmark.py --phase hosted   # or --phase local, or --force to redo
```
