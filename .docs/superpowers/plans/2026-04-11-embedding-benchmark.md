# Embedding Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **⚠ Commit policy:** The user handles all git commits personally. **Do not run `git commit` or `git add` in any task.** At the end of each task, notify the user that the task is complete and let them commit when they want.

**Goal:** Benchmark 12 text embedding models (6 hosted + 5 local + 1 English reference) on a bilingual FR/EN retrieval task with hard negatives, to pick the best default embedding models for graph-mem's `search_memory`.

**Architecture:** Standalone Python harness under `.benchmark/embedding/`. Config-driven via `models.yaml`, seven adapter families dispatched from a single `run_benchmark.py`, pure-function metrics module, JSON result files consumed by a separate `analyze.py` that produces the final markdown analysis. No coupling to the graph-mem production code.

**Tech stack:** Python 3.11+, `httpx` (sync) for all HTTP adapters, `sentence-transformers` + `torch` for local HF models, `numpy` for cosine/ranking, `PyYAML`, `python-dotenv`, `tiktoken`, `pytest` for the test suite.

**Spec:** `.docs/superpowers/specs/2026-04-11-embedding-benchmark-design.md`

---

## File structure

```
.benchmark/embedding/
├── .env.example              # All API keys (OPENAI, VOYAGE, COHERE, GOOGLE, MISTRAL)
├── requirements.txt          # Python deps
├── pyproject.toml            # pytest config (optional, minimal)
├── models.yaml               # 12 model configurations
├── dataset.json              # 100 docs + 40 queries + hard negatives
├── adapters/
│   ├── __init__.py
│   ├── base.py               # EmbedAdapter protocol + EmbedResult dataclass
│   ├── fake.py               # Deterministic fake adapter for tests
│   ├── openai_adapter.py
│   ├── voyage_adapter.py
│   ├── cohere_adapter.py
│   ├── google_adapter.py
│   ├── mistral_adapter.py
│   ├── ollama_adapter.py
│   └── st_adapter.py         # sentence-transformers (in-process)
├── metrics.py                # recall@k, MRR, HN-penalty (pure functions)
├── ranking.py                # cosine matrix, top-k ranking (pure functions)
├── dataset.py                # dataset loader + schema validation
├── runner.py                 # benchmark execution loop (takes an adapter)
├── run_benchmark.py          # CLI entry point
├── analyze.py                # results/*.json → summary.json + analysis.md
├── results/                  # Gitignored except .gitkeep
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_metrics.py
    ├── test_ranking.py
    ├── test_dataset.py
    ├── test_adapters_contract.py
    ├── test_runner.py
    └── test_analyze.py
```

**File responsibility summary:**
- `adapters/base.py` defines a single `EmbedAdapter` protocol — one method: `embed(texts: list[str]) -> EmbedResult`. All seven adapter files implement this protocol.
- `metrics.py`, `ranking.py`, `dataset.py` are 100% pure functions — trivially testable.
- `runner.py` holds the orchestration loop but receives an adapter instance, so it's unit-testable with the fake adapter.
- `run_benchmark.py` is the CLI wrapper around `runner.py`. Thin.
- `analyze.py` is standalone; it reads `results/*.json` and emits the final markdown.

---

## Task 1: Scaffold directory and dependencies

**Files:**
- Create: `.benchmark/embedding/.env.example`
- Create: `.benchmark/embedding/requirements.txt`
- Create: `.benchmark/embedding/pyproject.toml`
- Create: `.benchmark/embedding/.gitignore`
- Create: `.benchmark/embedding/results/.gitkeep`
- Create: `.benchmark/embedding/__init__.py`
- Create: `.benchmark/embedding/adapters/__init__.py`
- Create: `.benchmark/embedding/tests/__init__.py`
- Create: `.benchmark/embedding/tests/conftest.py`

- [ ] **Step 1: Create `.env.example` with every vendor key documented**

```bash
# Hosted embedding APIs — one key per vendor
# All five are OPTIONAL: any model whose key is missing will be skipped with a warning.

# OpenAI — https://platform.openai.com/api-keys
OPENAI_API_KEY=sk-proj-...

# Voyage AI — https://dash.voyageai.com/api-keys
VOYAGE_API_KEY=pa-...

# Cohere — https://dashboard.cohere.com/api-keys
COHERE_API_KEY=...

# Google Generative Language API — https://aistudio.google.com/app/apikey
GOOGLE_API_KEY=...

# Mistral — https://console.mistral.ai/api-keys
MISTRAL_API_KEY=...

# Ollama endpoint (local)
OLLAMA_BASE_URL=http://127.0.0.1:11434
```

- [ ] **Step 2: Create `requirements.txt`**

```
httpx>=0.27
pyyaml>=6.0
numpy>=1.26
python-dotenv>=1.0
tiktoken>=0.7
sentence-transformers>=3.0
torch>=2.3
pytest>=8.0
pytest-mock>=3.12
```

- [ ] **Step 3: Create minimal `pyproject.toml` for pytest**

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
addopts = "-ra --strict-markers"
```

- [ ] **Step 4: Create `.gitignore`**

```
results/*.json
!results/.gitkeep
.env
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 5: Create empty `__init__.py` files**

Create empty files at:
- `.benchmark/embedding/__init__.py`
- `.benchmark/embedding/adapters/__init__.py`
- `.benchmark/embedding/tests/__init__.py`

- [ ] **Step 6: Create `.benchmark/embedding/tests/conftest.py`**

```python
"""Shared pytest fixtures for the embedding benchmark tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def tiny_dataset() -> dict:
    """A minimal dataset used by metrics / runner tests.

    Three corpus docs, two queries, one query has a hard negative.
    """
    return {
        "version": "test",
        "corpus": [
            {"id": "c1", "text": "Bruno uses Python.", "lang": "en", "tags": []},
            {"id": "c2", "text": "Bruno used to use Ruby.", "lang": "en", "tags": []},
            {"id": "c3", "text": "Julie works on Atlas.", "lang": "en", "tags": []},
        ],
        "queries": [
            {
                "id": "q1",
                "query": "what tech does Bruno use?",
                "lang": "en",
                "relevant": ["c1"],
                "hard_negatives": ["c2"],
                "category": "temporal",
            },
            {
                "id": "q2",
                "query": "who works on Atlas?",
                "lang": "en",
                "relevant": ["c3"],
                "hard_negatives": [],
                "category": "user_fact",
            },
        ],
    }


@pytest.fixture
def tiny_dataset_file(tmp_path: Path, tiny_dataset: dict) -> Path:
    p = tmp_path / "dataset.json"
    p.write_text(json.dumps(tiny_dataset), encoding="utf-8")
    return p
```

- [ ] **Step 7: Verify the scaffold is importable**

Run: `cd .benchmark/embedding && python -c "import adapters, tests"`
Expected: no output, exit code 0

- [ ] **Step 8: Notify user — Task 1 complete, ready for commit.**

---

## Task 2: Dataset schema + loader (TDD)

**Files:**
- Create: `.benchmark/embedding/dataset.py`
- Create: `.benchmark/embedding/tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_dataset.py`:

```python
"""Tests for dataset loading and schema validation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dataset import (
    Corpus,
    Dataset,
    DatasetError,
    Query,
    load_dataset,
)


def test_load_dataset_returns_parsed_structure(tiny_dataset_file: Path) -> None:
    ds = load_dataset(tiny_dataset_file)
    assert isinstance(ds, Dataset)
    assert len(ds.corpus) == 3
    assert len(ds.queries) == 2
    assert ds.corpus[0].id == "c1"
    assert ds.queries[0].relevant == ["c1"]
    assert ds.queries[0].hard_negatives == ["c2"]


def test_load_dataset_builds_id_index(tiny_dataset_file: Path) -> None:
    ds = load_dataset(tiny_dataset_file)
    assert ds.corpus_by_id["c1"].text == "Bruno uses Python."


def test_unknown_relevant_id_raises(tmp_path: Path) -> None:
    bad = {
        "version": "test",
        "corpus": [{"id": "c1", "text": "x", "lang": "en", "tags": []}],
        "queries": [
            {
                "id": "q1",
                "query": "q",
                "lang": "en",
                "relevant": ["c999"],
                "hard_negatives": [],
                "category": "user_fact",
            }
        ],
    }
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(DatasetError, match="unknown corpus id"):
        load_dataset(p)


def test_duplicate_corpus_id_raises(tmp_path: Path) -> None:
    bad = {
        "version": "test",
        "corpus": [
            {"id": "c1", "text": "a", "lang": "en", "tags": []},
            {"id": "c1", "text": "b", "lang": "en", "tags": []},
        ],
        "queries": [],
    }
    p = tmp_path / "dup.json"
    p.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate corpus id"):
        load_dataset(p)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cd .benchmark/embedding && pytest tests/test_dataset.py -v`
Expected: ImportError — `dataset` module does not exist.

- [ ] **Step 3: Write the minimal implementation**

Create `.benchmark/embedding/dataset.py`:

```python
"""Dataset loader for the embedding benchmark.

The dataset is a JSON document with `corpus` (docs) and `queries`
(retrieval tests with relevance judgments). This module loads and
validates the file and exposes typed dataclasses.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class DatasetError(ValueError):
    """Raised when the dataset is structurally invalid."""


@dataclass(frozen=True)
class Corpus:
    id: str
    text: str
    lang: str
    tags: list[str]


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    lang: str
    relevant: list[str]
    hard_negatives: list[str]
    category: str


@dataclass(frozen=True)
class Dataset:
    version: str
    corpus: list[Corpus]
    queries: list[Query]
    corpus_by_id: dict[str, Corpus] = field(default_factory=dict)


def load_dataset(path: Path) -> Dataset:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    corpus = [Corpus(**d) for d in raw.get("corpus", [])]
    queries = [Query(**q) for q in raw.get("queries", [])]

    corpus_by_id: dict[str, Corpus] = {}
    for doc in corpus:
        if doc.id in corpus_by_id:
            raise DatasetError(f"duplicate corpus id: {doc.id}")
        corpus_by_id[doc.id] = doc

    for q in queries:
        for ref in [*q.relevant, *q.hard_negatives]:
            if ref not in corpus_by_id:
                raise DatasetError(
                    f"query {q.id}: unknown corpus id {ref}"
                )

    return Dataset(
        version=raw.get("version", "unknown"),
        corpus=corpus,
        queries=queries,
        corpus_by_id=corpus_by_id,
    )
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_dataset.py -v`
Expected: 4 passed.

- [ ] **Step 5: Notify user — Task 2 complete, ready for commit.**

---

## Task 3: Ranking utility (TDD)

**Files:**
- Create: `.benchmark/embedding/ranking.py`
- Create: `.benchmark/embedding/tests/test_ranking.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_ranking.py`:

```python
"""Tests for cosine similarity and top-k ranking."""
from __future__ import annotations

import numpy as np

from ranking import cosine_matrix, l2_normalize, rank_ids, truncate_and_renormalize


def test_l2_normalize_unit_length() -> None:
    v = np.array([[3.0, 4.0]])
    out = l2_normalize(v)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_cosine_matrix_identical_vectors_score_1() -> None:
    a = l2_normalize(np.array([[1.0, 0.0, 0.0]]))
    b = l2_normalize(np.array([[1.0, 0.0, 0.0]]))
    m = cosine_matrix(a, b)
    assert m.shape == (1, 1)
    assert np.isclose(m[0, 0], 1.0)


def test_cosine_matrix_orthogonal_score_0() -> None:
    a = l2_normalize(np.array([[1.0, 0.0]]))
    b = l2_normalize(np.array([[0.0, 1.0]]))
    m = cosine_matrix(a, b)
    assert np.isclose(m[0, 0], 0.0)


def test_rank_ids_returns_sorted_ids_descending_by_score() -> None:
    scores = np.array([0.1, 0.9, 0.5])
    ids = ["a", "b", "c"]
    ranked = rank_ids(scores, ids)
    assert ranked == ["b", "c", "a"]


def test_truncate_and_renormalize_preserves_unit_norm() -> None:
    v = l2_normalize(np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]))
    out = truncate_and_renormalize(v, 3)
    assert out.shape == (1, 3)
    assert np.isclose(np.linalg.norm(out, axis=1)[0], 1.0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cd .benchmark/embedding && pytest tests/test_ranking.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the minimal implementation**

Create `.benchmark/embedding/ranking.py`:

```python
"""Cosine similarity, L2 normalisation, and ranking utilities.

All functions are pure and accept numpy arrays. They do not touch I/O.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)


def cosine_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix of shape (Q, C).

    Assumes both inputs are L2-normalised — callers are responsible for
    normalisation. This keeps the math a single matrix multiply.
    """
    return queries @ corpus.T


def rank_ids(scores: np.ndarray, ids: list[str]) -> list[str]:
    """Return ids sorted from highest to lowest score.

    Stable with respect to the input order on ties.
    """
    order = np.argsort(-scores, kind="stable")
    return [ids[i] for i in order]


def truncate_and_renormalize(mat: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka-style slice to `dim` then L2-renormalise."""
    if dim > mat.shape[1]:
        raise ValueError(
            f"requested dim {dim} is larger than vector dim {mat.shape[1]}"
        )
    return l2_normalize(mat[:, :dim])
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_ranking.py -v`
Expected: 5 passed.

- [ ] **Step 5: Notify user — Task 3 complete, ready for commit.**

---

## Task 4: Metrics module (TDD)

**Files:**
- Create: `.benchmark/embedding/metrics.py`
- Create: `.benchmark/embedding/tests/test_metrics.py`

- [ ] **Step 1: Write the failing tests**

Write `tests/test_metrics.py`:

```python
"""Tests for retrieval metrics."""
from __future__ import annotations

from metrics import QueryEval, evaluate_query, aggregate


def make_eval(ranked: list[str], relevant: list[str], hns: list[str]) -> QueryEval:
    return evaluate_query(
        ranked_ids=ranked,
        relevant=set(relevant),
        hard_negatives=set(hns),
    )


def test_recall_at_1_hit() -> None:
    e = make_eval(ranked=["c1", "c2", "c3"], relevant=["c1"], hns=[])
    assert e.recall_at_1 == 1.0
    assert e.recall_at_5 == 1.0
    assert e.reciprocal_rank == 1.0


def test_recall_at_1_miss_but_at_5() -> None:
    e = make_eval(ranked=["c9", "c1", "c2"], relevant=["c1"], hns=[])
    assert e.recall_at_1 == 0.0
    assert e.recall_at_5 == 1.0
    assert e.reciprocal_rank == 0.5


def test_no_relevant_found_scores_zero() -> None:
    e = make_eval(ranked=["c9", "c8", "c7"], relevant=["c1"], hns=[])
    assert e.recall_at_1 == 0.0
    assert e.recall_at_5 == 0.0
    assert e.reciprocal_rank == 0.0


def test_hard_negative_rank_is_reported() -> None:
    e = make_eval(
        ranked=["c1", "c2", "c3", "c4", "c5"],
        relevant=["c1"],
        hns=["c2", "c5"],
    )
    # c2 is rank 2, c5 is rank 5 → mean 3.5
    assert e.hard_negative_mean_rank == 3.5


def test_hard_negative_missing_from_ranking_uses_worst_rank() -> None:
    # c2 is ranked, c9 is NOT → c9 gets len(ranked)+1 = 4
    e = make_eval(
        ranked=["c1", "c2", "c3"],
        relevant=["c1"],
        hns=["c2", "c9"],
    )
    # c2 rank 2, c9 rank 4 (penalty) → mean 3.0
    assert e.hard_negative_mean_rank == 3.0


def test_aggregate_averages_over_queries() -> None:
    evals = [
        make_eval(["c1"], ["c1"], []),
        make_eval(["c9", "c1"], ["c1"], []),
    ]
    agg = aggregate(evals)
    assert agg["recall_at_1"] == 0.5
    assert agg["recall_at_5"] == 1.0
    assert agg["mrr"] == 0.75
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cd .benchmark/embedding && pytest tests/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the minimal implementation**

Create `.benchmark/embedding/metrics.py`:

```python
"""Retrieval metrics: recall@k, MRR, hard-negative penalty.

All functions are pure. The caller supplies already-ranked doc ids.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class QueryEval:
    query_id: str
    recall_at_1: float
    recall_at_5: float
    reciprocal_rank: float
    hard_negative_mean_rank: float | None  # None if no HN on this query


def evaluate_query(
    *,
    ranked_ids: list[str],
    relevant: set[str],
    hard_negatives: set[str],
    query_id: str = "",
    k_list: tuple[int, ...] = (1, 5),
) -> QueryEval:
    r1 = 1.0 if _hit_at(ranked_ids, relevant, 1) else 0.0
    r5 = 1.0 if _hit_at(ranked_ids, relevant, 5) else 0.0
    rr = _reciprocal_rank(ranked_ids, relevant)
    hn_rank = _hard_negative_mean_rank(ranked_ids, hard_negatives)
    return QueryEval(
        query_id=query_id,
        recall_at_1=r1,
        recall_at_5=r5,
        reciprocal_rank=rr,
        hard_negative_mean_rank=hn_rank,
    )


def aggregate(evals: list[QueryEval]) -> dict[str, float]:
    if not evals:
        return {"recall_at_1": 0.0, "recall_at_5": 0.0, "mrr": 0.0, "hn_mean_rank": 0.0}
    hn_ranks = [e.hard_negative_mean_rank for e in evals if e.hard_negative_mean_rank is not None]
    return {
        "recall_at_1": mean(e.recall_at_1 for e in evals),
        "recall_at_5": mean(e.recall_at_5 for e in evals),
        "mrr": mean(e.reciprocal_rank for e in evals),
        "hn_mean_rank": mean(hn_ranks) if hn_ranks else 0.0,
    }


def _hit_at(ranked: list[str], relevant: set[str], k: int) -> bool:
    return any(doc_id in relevant for doc_id in ranked[:k])


def _reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    for i, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def _hard_negative_mean_rank(ranked: list[str], hns: set[str]) -> float | None:
    if not hns:
        return None
    worst_rank = len(ranked) + 1
    ranks: list[int] = []
    for hn in hns:
        try:
            ranks.append(ranked.index(hn) + 1)
        except ValueError:
            ranks.append(worst_rank)
    return mean(ranks)
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_metrics.py -v`
Expected: 6 passed.

- [ ] **Step 5: Notify user — Task 4 complete, ready for commit.**

---

## Task 5: Adapter protocol + fake adapter + contract test

**Files:**
- Create: `.benchmark/embedding/adapters/base.py`
- Create: `.benchmark/embedding/adapters/fake.py`
- Create: `.benchmark/embedding/tests/test_adapters_contract.py`

- [ ] **Step 1: Write the contract test first**

Create `tests/test_adapters_contract.py`:

```python
"""Shared contract every adapter must satisfy.

Concrete adapter tests (HTTP ones) will parametrize this contract with
mocked httpx transports. For now we validate the contract itself via the
deterministic FakeAdapter.
"""
from __future__ import annotations

import numpy as np
import pytest

from adapters.base import EmbedAdapter, EmbedResult
from adapters.fake import FakeAdapter


@pytest.fixture
def adapter() -> EmbedAdapter:
    return FakeAdapter(dim=8, seed=42)


def test_embed_returns_result_object(adapter: EmbedAdapter) -> None:
    result = adapter.embed(["hello", "world"])
    assert isinstance(result, EmbedResult)


def test_embed_vectors_shape_matches_input(adapter: EmbedAdapter) -> None:
    result = adapter.embed(["a", "b", "c"])
    assert result.vectors.shape == (3, 8)


def test_embed_vectors_are_l2_normalized(adapter: EmbedAdapter) -> None:
    result = adapter.embed(["a", "b", "c"])
    norms = np.linalg.norm(result.vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_embed_same_input_gives_same_vector_for_fake(adapter: EmbedAdapter) -> None:
    r1 = adapter.embed(["hello"]).vectors
    r2 = adapter.embed(["hello"]).vectors
    assert np.allclose(r1, r2)


def test_embed_records_per_item_latency(adapter: EmbedAdapter) -> None:
    result = adapter.embed(["a", "b", "c"])
    assert len(result.latencies_ms) == 3
    assert all(lat >= 0 for lat in result.latencies_ms)
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `cd .benchmark/embedding && pytest tests/test_adapters_contract.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `adapters/base.py`**

```python
"""Adapter protocol and result dataclass.

Every concrete embedding backend must satisfy `EmbedAdapter.embed(texts)`
and return an `EmbedResult` with L2-normalised vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class EmbedResult:
    vectors: np.ndarray  # shape (N, dim), L2-normalised
    latencies_ms: list[float]  # one per input text, measured end-to-end per batch and divided
    model: str  # canonical model id as reported by the backend


class EmbedAdapter(Protocol):
    name: str  # adapter slug

    def embed(self, texts: list[str]) -> EmbedResult: ...
```

- [ ] **Step 4: Write `adapters/fake.py`**

```python
"""Deterministic fake adapter for tests.

Embeds each input as an N-dim vector derived from its hash, then
L2-normalises. No network, no external deps beyond numpy.
"""
from __future__ import annotations

import hashlib
import time

import numpy as np

from .base import EmbedAdapter, EmbedResult


class FakeAdapter:
    name = "fake"

    def __init__(self, dim: int = 8, seed: int = 0, latency_ms: float = 0.0) -> None:
        self.dim = dim
        self.seed = seed
        self.latency_ms = latency_ms

    def embed(self, texts: list[str]) -> EmbedResult:
        t0 = time.perf_counter()
        vectors = np.stack([self._vec(t) for t in texts])
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-12, None)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_item = max(elapsed_ms / max(len(texts), 1), self.latency_ms)
        return EmbedResult(
            vectors=vectors.astype(np.float32),
            latencies_ms=[per_item] * len(texts),
            model=f"fake:{self.dim}",
        )

    def _vec(self, text: str) -> np.ndarray:
        # Deterministic vector from a sha256 hash seeded with self.seed
        h = hashlib.sha256(f"{self.seed}:{text}".encode()).digest()
        # Repeat hash bytes until we have at least dim * 4 bytes
        buf = (h * ((self.dim * 4 // len(h)) + 1))[: self.dim * 4]
        arr = np.frombuffer(buf, dtype=np.uint32).astype(np.float32)
        # Centre around 0
        return arr - arr.mean()
```

- [ ] **Step 5: Run the tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_adapters_contract.py -v`
Expected: 5 passed.

- [ ] **Step 6: Notify user — Task 5 complete, ready for commit.**

---

## Task 6: OpenAI-family HTTP adapters (OpenAI, Voyage, Mistral, Ollama)

These four share a near-identical request/response shape (`{"input": [...], "model": "..."}` → `{"data": [{"embedding": [...]}]}`). We implement them as thin wrappers over a shared helper, each with its own endpoint, auth header, and response-path. Cohere and Google have different shapes and are handled in Task 7.

**Files:**
- Create: `.benchmark/embedding/adapters/_http.py`
- Create: `.benchmark/embedding/adapters/openai_adapter.py`
- Create: `.benchmark/embedding/adapters/voyage_adapter.py`
- Create: `.benchmark/embedding/adapters/mistral_adapter.py`
- Create: `.benchmark/embedding/adapters/ollama_adapter.py`
- Create: `.benchmark/embedding/tests/test_http_adapters.py`

- [ ] **Step 1: Write the shared HTTP test using httpx MockTransport**

Create `tests/test_http_adapters.py`:

```python
"""HTTP adapter tests using httpx MockTransport — no real network."""
from __future__ import annotations

import httpx
import numpy as np
import pytest

from adapters.openai_adapter import OpenAIAdapter
from adapters.voyage_adapter import VoyageAdapter
from adapters.mistral_adapter import MistralAdapter
from adapters.ollama_adapter import OllamaAdapter


def _openai_style_response(request: httpx.Request) -> httpx.Response:
    body = request.read()
    # OpenAI/Voyage/Mistral all accept {"input": [...], "model": "..."}
    import json as _json
    payload = _json.loads(body)
    texts = payload["input"] if isinstance(payload["input"], list) else [payload["input"]]
    data = [
        {"embedding": [float((i + 1) * 0.1)] * 4, "index": i}
        for i in range(len(texts))
    ]
    return httpx.Response(200, json={"data": data, "model": payload["model"]})


def test_openai_adapter_embeds_batch() -> None:
    transport = httpx.MockTransport(_openai_style_response)
    client = httpx.Client(transport=transport)
    adapter = OpenAIAdapter(
        api_key="fake",
        model="text-embedding-3-small",
        client=client,
    )
    result = adapter.embed(["hello", "world"])
    assert result.vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
    assert result.model == "text-embedding-3-small"


def test_voyage_adapter_embeds_batch() -> None:
    transport = httpx.MockTransport(_openai_style_response)
    client = httpx.Client(transport=transport)
    adapter = VoyageAdapter(
        api_key="fake",
        model="voyage-3-large",
        client=client,
    )
    result = adapter.embed(["hello"])
    assert result.vectors.shape == (1, 4)


def test_mistral_adapter_embeds_batch() -> None:
    transport = httpx.MockTransport(_openai_style_response)
    client = httpx.Client(transport=transport)
    adapter = MistralAdapter(
        api_key="fake",
        model="mistral-embed",
        client=client,
    )
    result = adapter.embed(["hello"])
    assert result.vectors.shape == (1, 4)


def _ollama_response(request: httpx.Request) -> httpx.Response:
    # Ollama has a per-item endpoint: {"model": "...", "prompt": "..."}
    # → {"embedding": [...]}
    import json as _json
    payload = _json.loads(request.read())
    return httpx.Response(
        200,
        json={"embedding": [0.3, 0.4, 0.0, 0.0], "model": payload["model"]},
    )


def test_ollama_adapter_embeds_one_at_a_time() -> None:
    transport = httpx.MockTransport(_ollama_response)
    client = httpx.Client(transport=transport)
    adapter = OllamaAdapter(
        base_url="http://fake",
        model="qwen3-embedding-8b",
        client=client,
    )
    result = adapter.embed(["a", "b", "c"])
    assert result.vectors.shape == (3, 4)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
```

- [ ] **Step 2: Run the tests and verify they fail**

Run: `cd .benchmark/embedding && pytest tests/test_http_adapters.py -v`
Expected: ImportError.

- [ ] **Step 3: Write the shared HTTP helper `adapters/_http.py`**

```python
"""Shared HTTP utilities for embedding adapters.

Provides a default httpx.Client with sane timeouts and the L2-normalise
post-process that every adapter applies to its raw response.
"""
from __future__ import annotations

import httpx
import numpy as np

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def default_client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT)


def l2_normalise_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)
```

- [ ] **Step 4: Write `adapters/openai_adapter.py`**

```python
"""OpenAI `/v1/embeddings` adapter — works for text-embedding-3-large/small."""
from __future__ import annotations

import time

import httpx
import numpy as np

from ._http import default_client, l2_normalise_rows
from .base import EmbedResult


class OpenAIAdapter:
    name = "openai"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or default_client()

    def embed(self, texts: list[str]) -> EmbedResult:
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        payload = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        vecs = np.array(
            [item["embedding"] for item in payload["data"]], dtype=np.float32
        )
        vecs = l2_normalise_rows(vecs)
        per_item = elapsed_ms / max(len(texts), 1)
        return EmbedResult(
            vectors=vecs,
            latencies_ms=[per_item] * len(texts),
            model=payload.get("model", self._model),
        )
```

- [ ] **Step 5: Write `adapters/voyage_adapter.py`**

```python
"""Voyage AI `/v1/embeddings` adapter.

Voyage accepts OpenAI-shaped requests: `{"input": [...], "model": "..."}`
but also supports `input_type` ("query" or "document"). We pass
`input_type="document"` by default and let the runner override it if
needed via the constructor.
"""
from __future__ import annotations

import time

import httpx
import numpy as np

from ._http import default_client, l2_normalise_rows
from .base import EmbedResult


class VoyageAdapter:
    name = "voyage"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        input_type: str = "document",
        base_url: str = "https://api.voyageai.com/v1",
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._input_type = input_type
        self._base_url = base_url.rstrip("/")
        self._client = client or default_client()

    def embed(self, texts: list[str]) -> EmbedResult:
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "input": texts,
                "model": self._model,
                "input_type": self._input_type,
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        vecs = np.array(
            [item["embedding"] for item in payload["data"]], dtype=np.float32
        )
        vecs = l2_normalise_rows(vecs)
        per_item = elapsed_ms / max(len(texts), 1)
        return EmbedResult(
            vectors=vecs,
            latencies_ms=[per_item] * len(texts),
            model=payload.get("model", self._model),
        )
```

- [ ] **Step 6: Write `adapters/mistral_adapter.py`**

```python
"""Mistral `/v1/embeddings` adapter (OpenAI-compatible shape)."""
from __future__ import annotations

import time

import httpx
import numpy as np

from ._http import default_client, l2_normalise_rows
from .base import EmbedResult


class MistralAdapter:
    name = "mistral"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.mistral.ai/v1",
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._client = client or default_client()

    def embed(self, texts: list[str]) -> EmbedResult:
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self._base_url}/embeddings",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"input": texts, "model": self._model},
        )
        resp.raise_for_status()
        payload = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        vecs = np.array(
            [item["embedding"] for item in payload["data"]], dtype=np.float32
        )
        vecs = l2_normalise_rows(vecs)
        per_item = elapsed_ms / max(len(texts), 1)
        return EmbedResult(
            vectors=vecs,
            latencies_ms=[per_item] * len(texts),
            model=payload.get("model", self._model),
        )
```

- [ ] **Step 7: Write `adapters/ollama_adapter.py`**

```python
"""Ollama `/api/embeddings` adapter.

Ollama's embedding endpoint accepts one prompt per call, so we loop.
Latency is measured per call and reported per item.
"""
from __future__ import annotations

import time

import httpx
import numpy as np

from ._http import default_client, l2_normalise_rows
from .base import EmbedResult


class OllamaAdapter:
    name = "ollama"

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or default_client()

    def embed(self, texts: list[str]) -> EmbedResult:
        vecs: list[list[float]] = []
        latencies: list[float] = []
        for text in texts:
            t0 = time.perf_counter()
            resp = self._client.post(
                f"{self._base_url}/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            elapsed_ms = (time.perf_counter() - t0) * 1000
            vecs.append(resp.json()["embedding"])
            latencies.append(elapsed_ms)

        arr = l2_normalise_rows(np.array(vecs, dtype=np.float32))
        return EmbedResult(vectors=arr, latencies_ms=latencies, model=self._model)
```

- [ ] **Step 8: Run the tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_http_adapters.py -v`
Expected: 4 passed.

- [ ] **Step 9: Notify user — Task 6 complete, ready for commit.**

---

## Task 7: Cohere and Google adapters

These two don't fit the OpenAI shape.
- **Cohere v2 `/v2/embed`**: request `{"texts": [...], "model": "embed-v4.0", "input_type": "search_document", "embedding_types": ["float"]}` → response `{"embeddings": {"float": [[...], [...]]}}`.
- **Google Generative Language API**: endpoint `https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key=...`, request `{"requests": [{"model": "models/...", "content": {"parts": [{"text": "..."}]}, "task_type": "RETRIEVAL_DOCUMENT"}]}` → response `{"embeddings": [{"values": [...]}]}`.

**Files:**
- Create: `.benchmark/embedding/adapters/cohere_adapter.py`
- Create: `.benchmark/embedding/adapters/google_adapter.py`
- Create: `.benchmark/embedding/tests/test_cohere_google.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cohere_google.py`:

```python
"""Tests for the Cohere and Google adapters via httpx MockTransport."""
from __future__ import annotations

import json

import httpx
import numpy as np

from adapters.cohere_adapter import CohereAdapter
from adapters.google_adapter import GoogleAdapter


def _cohere_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.read())
    n = len(payload["texts"])
    return httpx.Response(
        200,
        json={
            "id": "cohere-mock",
            "embeddings": {
                "float": [[0.1, 0.2, 0.3, 0.4] for _ in range(n)],
            },
            "model": payload["model"],
        },
    )


def test_cohere_adapter_embeds_batch() -> None:
    transport = httpx.MockTransport(_cohere_response)
    client = httpx.Client(transport=transport)
    adapter = CohereAdapter(
        api_key="fake",
        model="embed-v4.0",
        client=client,
    )
    result = adapter.embed(["hello", "world"])
    assert result.vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
    assert result.model == "embed-v4.0"


def _google_response(request: httpx.Request) -> httpx.Response:
    payload = json.loads(request.read())
    n = len(payload["requests"])
    return httpx.Response(
        200,
        json={
            "embeddings": [
                {"values": [0.5, 0.5, 0.5, 0.5]} for _ in range(n)
            ]
        },
    )


def test_google_adapter_embeds_batch() -> None:
    transport = httpx.MockTransport(_google_response)
    client = httpx.Client(transport=transport)
    adapter = GoogleAdapter(
        api_key="fake",
        model="gemini-embedding-001",
        client=client,
    )
    result = adapter.embed(["hello", "world"])
    assert result.vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `cd .benchmark/embedding && pytest tests/test_cohere_google.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `adapters/cohere_adapter.py`**

```python
"""Cohere `/v2/embed` adapter."""
from __future__ import annotations

import time

import httpx
import numpy as np

from ._http import default_client, l2_normalise_rows
from .base import EmbedResult


class CohereAdapter:
    name = "cohere"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        input_type: str = "search_document",
        base_url: str = "https://api.cohere.com/v2",
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._input_type = input_type
        self._base_url = base_url.rstrip("/")
        self._client = client or default_client()

    def embed(self, texts: list[str]) -> EmbedResult:
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self._base_url}/embed",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "texts": texts,
                "model": self._model,
                "input_type": self._input_type,
                "embedding_types": ["float"],
            },
        )
        resp.raise_for_status()
        payload = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        vecs = np.array(payload["embeddings"]["float"], dtype=np.float32)
        vecs = l2_normalise_rows(vecs)
        per_item = elapsed_ms / max(len(texts), 1)
        return EmbedResult(
            vectors=vecs,
            latencies_ms=[per_item] * len(texts),
            model=payload.get("model", self._model),
        )
```

- [ ] **Step 4: Write `adapters/google_adapter.py`**

```python
"""Google Generative Language API batch embeddings adapter.

Endpoint pattern:
  POST {base}/models/{model}:batchEmbedContents?key={api_key}

Request:
  {"requests": [{"model": "models/{model}", "content": {"parts": [{"text": "..."}]},
                 "task_type": "RETRIEVAL_DOCUMENT"}, ...]}
"""
from __future__ import annotations

import time

import httpx
import numpy as np

from ._http import default_client, l2_normalise_rows
from .base import EmbedResult


class GoogleAdapter:
    name = "google"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._task_type = task_type
        self._base_url = base_url.rstrip("/")
        self._client = client or default_client()

    def embed(self, texts: list[str]) -> EmbedResult:
        requests = [
            {
                "model": f"models/{self._model}",
                "content": {"parts": [{"text": text}]},
                "task_type": self._task_type,
            }
            for text in texts
        ]
        t0 = time.perf_counter()
        resp = self._client.post(
            f"{self._base_url}/models/{self._model}:batchEmbedContents",
            params={"key": self._api_key},
            json={"requests": requests},
        )
        resp.raise_for_status()
        payload = resp.json()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        vecs = np.array(
            [item["values"] for item in payload["embeddings"]], dtype=np.float32
        )
        vecs = l2_normalise_rows(vecs)
        per_item = elapsed_ms / max(len(texts), 1)
        return EmbedResult(
            vectors=vecs,
            latencies_ms=[per_item] * len(texts),
            model=self._model,
        )
```

- [ ] **Step 5: Run tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_cohere_google.py -v`
Expected: 2 passed.

- [ ] **Step 6: Notify user — Task 7 complete, ready for commit.**

---

## Task 8: sentence-transformers adapter (in-process)

**Files:**
- Create: `.benchmark/embedding/adapters/st_adapter.py`
- Create: `.benchmark/embedding/tests/test_st_adapter.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_st_adapter.py`:

```python
"""Tests for the sentence-transformers in-process adapter.

Uses a monkeypatched fake SentenceTransformer to avoid downloading weights.
"""
from __future__ import annotations

import numpy as np
import pytest

from adapters import st_adapter as st_mod


class _FakeST:
    def __init__(self, name: str, **_: object) -> None:
        self.name = name

    def encode(
        self, texts: list[str], convert_to_numpy: bool = True, **_: object
    ) -> np.ndarray:
        return np.ones((len(texts), 4), dtype=np.float32)


def test_st_adapter_embeds_and_normalises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(st_mod, "SentenceTransformer", _FakeST)
    adapter = st_mod.SentenceTransformersAdapter(model="fake/model")
    result = adapter.embed(["hello", "world"])
    assert result.vectors.shape == (2, 4)
    assert np.allclose(np.linalg.norm(result.vectors, axis=1), 1.0)
    assert result.model == "fake/model"
    assert len(result.latencies_ms) == 2
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `cd .benchmark/embedding && pytest tests/test_st_adapter.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `adapters/st_adapter.py`**

```python
"""sentence-transformers adapter — loads HuggingFace models locally.

Imports sentence_transformers lazily so the rest of the suite can run
without torch installed. The tests monkeypatch `SentenceTransformer` at
module level to avoid downloading weights.
"""
from __future__ import annotations

import time

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover — lazily surfaced
    SentenceTransformer = None  # type: ignore[assignment]

from ._http import l2_normalise_rows
from .base import EmbedResult


class SentenceTransformersAdapter:
    name = "sentence-transformers"

    def __init__(
        self,
        *,
        model: str,
        trust_remote_code: bool = True,
        device: str | None = None,
    ) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed; "
                "run `pip install -r requirements.txt`"
            )
        self._model_name = model
        self._model = SentenceTransformer(
            model, trust_remote_code=trust_remote_code, device=device
        )

    def embed(self, texts: list[str]) -> EmbedResult:
        t0 = time.perf_counter()
        vecs = self._model.encode(texts, convert_to_numpy=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        arr = l2_normalise_rows(np.asarray(vecs, dtype=np.float32))
        per_item = elapsed_ms / max(len(texts), 1)
        return EmbedResult(
            vectors=arr,
            latencies_ms=[per_item] * len(texts),
            model=self._model_name,
        )
```

- [ ] **Step 4: Run tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_st_adapter.py -v`
Expected: 1 passed.

- [ ] **Step 5: Notify user — Task 8 complete, ready for commit.**

---

## Task 9: Runner core (TDD with fake adapter)

**Files:**
- Create: `.benchmark/embedding/runner.py`
- Create: `.benchmark/embedding/tests/test_runner.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_runner.py`:

```python
"""Tests for the benchmark runner, using the FakeAdapter end-to-end."""
from __future__ import annotations

import json
from pathlib import Path

from adapters.fake import FakeAdapter
from dataset import load_dataset
from runner import RunConfig, run_benchmark


def test_run_benchmark_produces_result_file(
    tiny_dataset_file: Path, tmp_path: Path
) -> None:
    ds = load_dataset(tiny_dataset_file)
    adapter = FakeAdapter(dim=8, seed=1)
    cfg = RunConfig(
        model_slug="fake-test",
        dim_native=8,
        dim_used=8,
        price_per_1m_usd=None,
        output_dir=tmp_path,
    )
    result_path = run_benchmark(adapter, ds, cfg)

    assert result_path.exists()
    data = json.loads(result_path.read_text(encoding="utf-8"))

    assert data["model_slug"] == "fake-test"
    assert data["dim_used"] == 8
    assert "metrics" in data
    assert "recall_at_1" in data["metrics"]
    assert "per_query" in data
    assert len(data["per_query"]) == len(ds.queries)
    assert "latency" in data
    assert "p50_ms" in data["latency"]


def test_run_benchmark_with_truncation(
    tiny_dataset_file: Path, tmp_path: Path
) -> None:
    ds = load_dataset(tiny_dataset_file)
    adapter = FakeAdapter(dim=8, seed=1)
    cfg = RunConfig(
        model_slug="fake-test",
        dim_native=8,
        dim_used=4,
        price_per_1m_usd=None,
        output_dir=tmp_path,
    )
    result_path = run_benchmark(adapter, ds, cfg)
    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert data["dim_used"] == 4
    assert data["dim_native"] == 8
```

- [ ] **Step 2: Run and verify failure**

Run: `cd .benchmark/embedding && pytest tests/test_runner.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `runner.py`**

```python
"""Benchmark runner: embed corpus + queries, compute metrics, dump JSON.

The runner receives an already-constructed adapter, so it is fully
testable with FakeAdapter. It does not touch environment variables or
the filesystem beyond the output directory.
"""
from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from adapters.base import EmbedAdapter
from dataset import Dataset
from metrics import QueryEval, aggregate, evaluate_query
from ranking import cosine_matrix, rank_ids, truncate_and_renormalize


@dataclass(frozen=True)
class RunConfig:
    model_slug: str
    dim_native: int
    dim_used: int
    price_per_1m_usd: float | None
    output_dir: Path
    dataset_version: str = ""
    warmup_texts: tuple[str, ...] = ("warmup one", "warmup two", "warmup three")


def run_benchmark(
    adapter: EmbedAdapter, dataset: Dataset, cfg: RunConfig
) -> Path:
    # --- warmup (not timed)
    adapter.embed(list(cfg.warmup_texts))

    # --- corpus embedding
    corpus_texts = [doc.text for doc in dataset.corpus]
    corpus_ids = [doc.id for doc in dataset.corpus]
    t0 = time.perf_counter()
    corpus_res = adapter.embed(corpus_texts)
    corpus_wall_ms = (time.perf_counter() - t0) * 1000

    # --- query embedding
    query_texts = [q.query for q in dataset.queries]
    t0 = time.perf_counter()
    query_res = adapter.embed(query_texts)
    query_wall_ms = (time.perf_counter() - t0) * 1000

    corpus_vecs = corpus_res.vectors
    query_vecs = query_res.vectors

    # --- Matryoshka truncation
    if cfg.dim_used < cfg.dim_native:
        corpus_vecs = truncate_and_renormalize(corpus_vecs, cfg.dim_used)
        query_vecs = truncate_and_renormalize(query_vecs, cfg.dim_used)

    # --- ranking + metrics
    sim = cosine_matrix(query_vecs, corpus_vecs)
    per_query: list[dict[str, object]] = []
    evals: list[QueryEval] = []
    for i, q in enumerate(dataset.queries):
        ranked = rank_ids(sim[i], corpus_ids)
        ev = evaluate_query(
            ranked_ids=ranked,
            relevant=set(q.relevant),
            hard_negatives=set(q.hard_negatives),
            query_id=q.id,
        )
        evals.append(ev)
        per_query.append(
            {
                "query_id": q.id,
                "category": q.category,
                "lang": q.lang,
                "top5": ranked[:5],
                "recall_at_1": ev.recall_at_1,
                "recall_at_5": ev.recall_at_5,
                "reciprocal_rank": ev.reciprocal_rank,
                "hard_negative_mean_rank": ev.hard_negative_mean_rank,
            }
        )

    agg = aggregate(evals)

    # --- latency stats (per-doc median and P95 across corpus+queries)
    all_latencies = list(corpus_res.latencies_ms) + list(query_res.latencies_ms)
    latency = {
        "p50_ms": _percentile(all_latencies, 50),
        "p95_ms": _percentile(all_latencies, 95),
        "corpus_wall_ms": corpus_wall_ms,
        "query_wall_ms": query_wall_ms,
    }

    # --- cost estimate (rough: 1 token ≈ 4 chars)
    total_chars = sum(len(t) for t in corpus_texts) + sum(
        len(t) for t in query_texts
    )
    est_tokens = total_chars / 4
    cost_usd = (
        (est_tokens / 1_000_000) * cfg.price_per_1m_usd
        if cfg.price_per_1m_usd is not None
        else None
    )

    payload = {
        "model_slug": cfg.model_slug,
        "model_reported": corpus_res.model,
        "dim_native": cfg.dim_native,
        "dim_used": cfg.dim_used,
        "dataset_version": cfg.dataset_version or dataset.version,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": agg,
        "per_query": per_query,
        "latency": latency,
        "cost": {
            "estimated_input_tokens": int(est_tokens),
            "estimated_cost_usd": cost_usd,
            "price_per_1m_usd": cfg.price_per_1m_usd,
        },
        "environment": _environment_snapshot(),
    }

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.output_dir / f"{cfg.model_slug}_dim{cfg.dim_used}.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def _percentile(values: list[float], p: int) -> float:
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100)
    lo, hi = int(k), min(int(k) + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _environment_snapshot() -> dict[str, str]:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True
        ).strip()
    except Exception:
        git_commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": git_commit,
    }
```

- [ ] **Step 4: Run and verify it passes**

Run: `cd .benchmark/embedding && pytest tests/test_runner.py -v`
Expected: 2 passed.

- [ ] **Step 5: Notify user — Task 9 complete, ready for commit.**

---

## Task 10: CLI entry point (`run_benchmark.py`) + `models.yaml`

**Files:**
- Create: `.benchmark/embedding/models.yaml`
- Create: `.benchmark/embedding/run_benchmark.py`
- Create: `.benchmark/embedding/tests/test_cli.py`

- [ ] **Step 1: Write `models.yaml` with all 12 model configs**

```yaml
# Embedding models under test for graph-mem.
# `enabled: false` → skip without running.
# Each entry maps to an adapter family in run_benchmark.py.

models:
  openai-3-large:
    enabled: true
    phase: hosted
    family: openai
    model: text-embedding-3-large
    api_key_env: OPENAI_API_KEY
    dim_native: 3072
    dim_runs: [3072, 1024]
    price_per_1m_usd: 0.13
    max_ctx: 8191

  openai-3-small:
    enabled: true
    phase: hosted
    family: openai
    model: text-embedding-3-small
    api_key_env: OPENAI_API_KEY
    dim_native: 1536
    dim_runs: [1536, 1024]
    price_per_1m_usd: 0.02
    max_ctx: 8191

  voyage-3-large:
    enabled: true
    phase: hosted
    family: voyage
    model: voyage-3-large
    api_key_env: VOYAGE_API_KEY
    dim_native: 2048
    dim_runs: [2048, 1024]
    price_per_1m_usd: 0.18
    max_ctx: 32000

  cohere-embed-v4:
    enabled: true
    phase: hosted
    family: cohere
    model: embed-v4.0
    api_key_env: COHERE_API_KEY
    dim_native: 1024
    dim_runs: [1024]
    price_per_1m_usd: 0.10
    max_ctx: 512

  gemini-embed-001:
    enabled: true
    phase: hosted
    family: google
    model: gemini-embedding-001
    api_key_env: GOOGLE_API_KEY
    dim_native: 3072
    dim_runs: [3072, 1024]
    price_per_1m_usd: 0.15
    max_ctx: 8192

  mistral-embed:
    enabled: true
    phase: hosted
    family: mistral
    model: mistral-embed
    api_key_env: MISTRAL_API_KEY
    dim_native: 1024
    dim_runs: [1024]
    price_per_1m_usd: 0.10
    max_ctx: 8000

  qwen3-embedding-8b:
    enabled: true
    phase: local
    family: ollama
    model: qwen3-embedding-8b
    base_url_env: OLLAMA_BASE_URL
    dim_native: 4096
    dim_runs: [1024]
    price_per_1m_usd: null
    max_ctx: 32000

  qwen3-embedding-4b:
    enabled: true
    phase: local
    family: ollama
    model: qwen3-embedding-4b
    base_url_env: OLLAMA_BASE_URL
    dim_native: 2560
    dim_runs: [1024]
    price_per_1m_usd: null
    max_ctx: 32000

  bge-m3:
    enabled: true
    phase: local
    family: sentence-transformers
    model: BAAI/bge-m3
    dim_native: 1024
    dim_runs: [1024]
    price_per_1m_usd: null
    max_ctx: 8192

  jina-v3:
    enabled: true
    phase: local
    family: sentence-transformers
    model: jinaai/jina-embeddings-v3
    dim_native: 1024
    dim_runs: [1024]
    price_per_1m_usd: null
    max_ctx: 8192

  nomic-v1.5:
    enabled: true
    phase: local
    family: sentence-transformers
    model: nomic-ai/nomic-embed-text-v1.5
    dim_native: 768
    dim_runs: [768]
    price_per_1m_usd: null
    max_ctx: 8192

  stella-en-1.5b:
    enabled: true
    phase: local
    family: sentence-transformers
    model: NovaSearch/stella_en_1.5B_v5
    dim_native: 1024
    dim_runs: [1024]
    price_per_1m_usd: null
    max_ctx: 512
```

- [ ] **Step 2: Write the CLI test**

Create `tests/test_cli.py`:

```python
"""Tests for the CLI glue: YAML parsing and adapter factory."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from run_benchmark import ModelEntry, build_adapter, load_models_yaml


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(yaml.safe_dump(data), encoding="utf-8")
    return p


def test_load_models_yaml_filters_disabled(tmp_path: Path) -> None:
    y = _write_yaml(
        tmp_path,
        {
            "models": {
                "a": {
                    "enabled": True,
                    "phase": "hosted",
                    "family": "openai",
                    "model": "m",
                    "api_key_env": "OPENAI_API_KEY",
                    "dim_native": 8,
                    "dim_runs": [8],
                    "price_per_1m_usd": 0.1,
                    "max_ctx": 100,
                },
                "b": {
                    "enabled": False,
                    "phase": "local",
                    "family": "ollama",
                    "model": "m",
                    "base_url_env": "OLLAMA_BASE_URL",
                    "dim_native": 8,
                    "dim_runs": [8],
                    "price_per_1m_usd": None,
                    "max_ctx": 100,
                },
            }
        },
    )
    entries = load_models_yaml(y)
    assert [e.slug for e in entries] == ["a"]


def test_build_adapter_unknown_family_raises() -> None:
    entry = ModelEntry(
        slug="x",
        phase="hosted",
        family="does-not-exist",
        model="m",
        config={},
        dim_native=8,
        dim_runs=[8],
        price_per_1m_usd=None,
        max_ctx=None,
    )
    with pytest.raises(ValueError, match="unknown adapter family"):
        build_adapter(entry, env={})


def test_build_adapter_missing_api_key_raises() -> None:
    entry = ModelEntry(
        slug="x",
        phase="hosted",
        family="openai",
        model="m",
        config={"api_key_env": "MISSING_KEY"},
        dim_native=8,
        dim_runs=[8],
        price_per_1m_usd=None,
        max_ctx=None,
    )
    with pytest.raises(RuntimeError, match="MISSING_KEY"):
        build_adapter(entry, env={})
```

- [ ] **Step 3: Run and verify failure**

Run: `cd .benchmark/embedding && pytest tests/test_cli.py -v`
Expected: ImportError.

- [ ] **Step 4: Write `run_benchmark.py`**

```python
"""CLI entry point for the embedding benchmark.

Usage:
    python run_benchmark.py                           # all enabled
    python run_benchmark.py --phase hosted            # only hosted
    python run_benchmark.py --phase local             # only local
    python run_benchmark.py --only qwen3-embedding-8b
    python run_benchmark.py --dry-run                 # canary only, 3 docs
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from adapters.base import EmbedAdapter
from adapters.cohere_adapter import CohereAdapter
from adapters.google_adapter import GoogleAdapter
from adapters.mistral_adapter import MistralAdapter
from adapters.ollama_adapter import OllamaAdapter
from adapters.openai_adapter import OpenAIAdapter
from adapters.st_adapter import SentenceTransformersAdapter
from adapters.voyage_adapter import VoyageAdapter
from dataset import load_dataset
from runner import RunConfig, run_benchmark


HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"
DATASET_FILE = HERE / "dataset.json"
MODELS_FILE = HERE / "models.yaml"


@dataclass(frozen=True)
class ModelEntry:
    slug: str
    phase: str               # "hosted" | "local"
    family: str              # "openai" | "voyage" | ...
    model: str               # canonical model id
    config: dict[str, Any]   # extra fields (api_key_env, base_url_env, ...)
    dim_native: int
    dim_runs: list[int]
    price_per_1m_usd: float | None
    max_ctx: int | None = None


def load_models_yaml(path: Path) -> list[ModelEntry]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    out: list[ModelEntry] = []
    for slug, cfg in raw["models"].items():
        if not cfg.get("enabled", True):
            continue
        out.append(
            ModelEntry(
                slug=slug,
                phase=cfg["phase"],
                family=cfg["family"],
                model=cfg["model"],
                config=cfg,
                dim_native=cfg["dim_native"],
                dim_runs=list(cfg["dim_runs"]),
                price_per_1m_usd=cfg.get("price_per_1m_usd"),
                max_ctx=cfg.get("max_ctx"),
            )
        )
    return out


def build_adapter(
    entry: ModelEntry, env: dict[str, str]
) -> EmbedAdapter:
    family = entry.family
    if family == "openai":
        return OpenAIAdapter(
            api_key=_require_env(env, entry.config["api_key_env"]),
            model=entry.model,
        )
    if family == "voyage":
        return VoyageAdapter(
            api_key=_require_env(env, entry.config["api_key_env"]),
            model=entry.model,
        )
    if family == "cohere":
        return CohereAdapter(
            api_key=_require_env(env, entry.config["api_key_env"]),
            model=entry.model,
        )
    if family == "google":
        return GoogleAdapter(
            api_key=_require_env(env, entry.config["api_key_env"]),
            model=entry.model,
        )
    if family == "mistral":
        return MistralAdapter(
            api_key=_require_env(env, entry.config["api_key_env"]),
            model=entry.model,
        )
    if family == "ollama":
        return OllamaAdapter(
            base_url=env.get(
                entry.config.get("base_url_env", "OLLAMA_BASE_URL"),
                "http://127.0.0.1:11434",
            ),
            model=entry.model,
        )
    if family == "sentence-transformers":
        return SentenceTransformersAdapter(model=entry.model)
    raise ValueError(f"unknown adapter family: {family}")


def _require_env(env: dict[str, str], key: str) -> str:
    val = env.get(key)
    if not val:
        raise RuntimeError(
            f"{key} is not set — add it to .env or the environment"
        )
    return val


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="graph-mem embedding benchmark")
    parser.add_argument("--only", help="run a single model slug")
    parser.add_argument(
        "--phase", choices=["hosted", "local"], help="filter by phase"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="embed 3 canary texts per model and print expected cost; no full run",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-run even if results already exist",
    )
    args = parser.parse_args(argv)

    load_dotenv(HERE / ".env")
    env = dict(os.environ)

    entries = load_models_yaml(MODELS_FILE)
    if args.only:
        entries = [e for e in entries if e.slug == args.only]
    if args.phase:
        entries = [e for e in entries if e.phase == args.phase]

    if not entries:
        print("No models matched the filters.", file=sys.stderr)
        return 1

    dataset = load_dataset(DATASET_FILE)

    total_cost_estimate = 0.0
    for entry in entries:
        print(f"\n=== {entry.slug} ({entry.family} / {entry.model}) ===")
        try:
            adapter = build_adapter(entry, env)
        except RuntimeError as err:
            print(f"  skipped: {err}")
            continue

        if args.dry_run:
            canary = ["hello", "monde", "le projet Atlas tourne sur Kubernetes"]
            result = adapter.embed(canary)
            print(
                f"  dry-run ok — canary shape {result.vectors.shape}, "
                f"p50 latency {result.latencies_ms[0]:.1f} ms"
            )
            if entry.price_per_1m_usd is not None:
                tokens = sum(len(c) for c in canary) / 4
                cost = (tokens / 1_000_000) * entry.price_per_1m_usd
                print(f"  full-run cost estimate: ${cost * 30:.5f}")
                total_cost_estimate += cost * 30
            continue

        for dim_used in entry.dim_runs:
            out_path = RESULTS_DIR / f"{entry.slug}_dim{dim_used}.json"
            if out_path.exists() and not args.force:
                print(f"  skipped {entry.slug}@{dim_used}d: result exists (use --force)")
                continue
            cfg = RunConfig(
                model_slug=entry.slug,
                dim_native=entry.dim_native,
                dim_used=dim_used,
                price_per_1m_usd=entry.price_per_1m_usd,
                output_dir=RESULTS_DIR,
            )
            print(f"  running @ dim={dim_used}...")
            path = run_benchmark(adapter, dataset, cfg)
            print(f"    → {path.name}")

    if args.dry_run and total_cost_estimate > 0:
        print(f"\nTotal estimated hosted cost for a full run: ${total_cost_estimate:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run the CLI tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_cli.py -v`
Expected: 3 passed.

- [ ] **Step 6: Run all tests to confirm nothing regressed**

Run: `cd .benchmark/embedding && pytest -v`
Expected: all tests pass.

- [ ] **Step 7: Notify user — Task 10 complete, ready for commit.**

---

## Task 11: Dataset authoring (human-in-the-loop)

This task produces `.benchmark/embedding/dataset.json` with 100 corpus docs and 40 queries per the spec.

**Files:**
- Create: `.benchmark/embedding/dataset.json`

**⚠ This is the only task that requires real content writing, not code.** Every item is hand-authored. No LLM-generated ground truth.

- [ ] **Step 1: Build the 5 personas outline**

In a scratch file (not committed), draft 5 personas and the tech/project/preference facts for each:

```
P1: Bruno — Python/Go dev, Paris, uses Neo4j, likes simple architectures
P2: Julie — data scientist, Lyon, uses Pandas/DuckDB, prefers notebooks
P3: Karim — SRE, Nantes, runs Kubernetes on bare metal, bash/Rust
P4: Léa — student, Python beginner, learning FastAPI
P5: Thomas — tech lead, TypeScript/Node, ships with pnpm, coaches the team
```

Each persona gets ~20 facts, cross-linked with others (e.g. "Julie works with Bruno on Atlas"). Mix FR/EN per the 60/30/10 split.

- [ ] **Step 2: Start from the extraction-benchmark seed**

Read `.benchmark/llm/dataset.json` and copy each `input` string as a standalone fact. Normalise to 5 personas by substituting names where needed. These become `c001`-`c025`.

- [ ] **Step 3: Author the remaining 75 corpus docs**

Add `c026`-`c100` to cover the 5 personas × ~15 additional facts each. Explicitly include:
- At least 10 **past** facts ("Bruno used to use Ruby before Python")
- At least 10 **current** facts that share surface features with a past fact (same person, same domain) → these become hard negatives
- At least 5 **negation** facts ("Léa does NOT work on Atlas")
- At least 5 **pronoun-first** facts ("Sylvie me coache sur C#")
- At least 10 **FR-only** facts and 3 **mixed-language** facts (code-switched)

- [ ] **Step 4: Author the 40 queries**

Structure:
- 15 `user_fact` queries (direct retrieval)
- 10 `project_fact` queries
- 5 `temporal` queries (past vs present, each with one HN)
- 4 `negation` queries (each with one HN)
- 3 `pronoun` queries
- 3 `mixed_lang` queries (query in FR against EN docs and vice versa)

For each query fill `relevant` (1-3 doc ids) and `hard_negatives` (0-3 ids).

- [ ] **Step 5: Write `dataset.json` with the complete structure**

Use this shape (the full content has 100+ entries — authored during this step, not copy-pasted here):

```json
{
  "version": "1.0",
  "description": "graph-mem embedding benchmark v1 — FR/EN short-fact retrieval with hard negatives.",
  "language_distribution": {"en": 60, "fr": 30, "mixed": 10},
  "corpus": [
    {"id": "c001", "text": "Bruno uses Python and likes Neo4j.", "lang": "en", "tags": ["user", "tech", "P1"]}
  ],
  "queries": [
    {"id": "q001", "query": "what tech does Bruno use?", "lang": "en", "relevant": ["c001"], "hard_negatives": [], "category": "user_fact"}
  ]
}
```

- [ ] **Step 6: Validate the dataset via the loader**

Run: `cd .benchmark/embedding && python -c "from dataset import load_dataset; ds = load_dataset('dataset.json'); print(f'{len(ds.corpus)} docs, {len(ds.queries)} queries')"`
Expected: `100 docs, 40 queries` and no validation error.

- [ ] **Step 7: USER REVIEW GATE**

**Stop and ask the user to review `dataset.json` before any model is run.** Wait for explicit approval. Do not proceed to Task 12 without it.

- [ ] **Step 8: Notify user — Task 11 complete, dataset ready for review.**

---

## Task 12: Analyzer (`analyze.py`)

**Files:**
- Create: `.benchmark/embedding/analyze.py`
- Create: `.benchmark/embedding/tests/test_analyze.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analyze.py`:

```python
"""Tests for the results aggregator / analysis writer."""
from __future__ import annotations

import json
from pathlib import Path

from analyze import aggregate_results, write_markdown_summary


def _make_result(slug: str, dim: int, r1: float, r5: float, mrr: float) -> dict:
    return {
        "model_slug": slug,
        "dim_used": dim,
        "dim_native": dim,
        "metrics": {
            "recall_at_1": r1,
            "recall_at_5": r5,
            "mrr": mrr,
            "hn_mean_rank": 4.2,
        },
        "latency": {"p50_ms": 120.0, "p95_ms": 300.0},
        "cost": {"estimated_cost_usd": 0.001, "price_per_1m_usd": 0.1},
        "per_query": [],
    }


def test_aggregate_results_sorts_by_mrr_desc(tmp_path: Path) -> None:
    (tmp_path / "a_dim1024.json").write_text(
        json.dumps(_make_result("a", 1024, 0.5, 0.9, 0.7))
    )
    (tmp_path / "b_dim1024.json").write_text(
        json.dumps(_make_result("b", 1024, 0.6, 0.95, 0.85))
    )
    rows = aggregate_results(tmp_path)
    assert [r["model_slug"] for r in rows] == ["b", "a"]


def test_write_markdown_summary(tmp_path: Path) -> None:
    (tmp_path / "a_dim1024.json").write_text(
        json.dumps(_make_result("a", 1024, 0.5, 0.9, 0.7))
    )
    rows = aggregate_results(tmp_path)
    md_path = tmp_path / "summary.md"
    write_markdown_summary(rows, md_path)
    content = md_path.read_text(encoding="utf-8")
    assert "| model_slug" in content or "| Model" in content
    assert "a" in content
```

- [ ] **Step 2: Run and verify failure**

Run: `cd .benchmark/embedding && pytest tests/test_analyze.py -v`
Expected: ImportError.

- [ ] **Step 3: Write `analyze.py`**

```python
"""Aggregate results/*.json into a comparative table and a markdown summary.

Usage:
    python analyze.py                         # writes results/summary.json + summary.md
    python analyze.py --out path/to/doc.md    # custom markdown output
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


def aggregate_results(results_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for fp in sorted(results_dir.glob("*.json")):
        if fp.name in ("summary.json",):
            continue
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "model_slug": data["model_slug"],
                "dim_used": data["dim_used"],
                "dim_native": data["dim_native"],
                "recall_at_1": data["metrics"]["recall_at_1"],
                "recall_at_5": data["metrics"]["recall_at_5"],
                "mrr": data["metrics"]["mrr"],
                "hn_mean_rank": data["metrics"]["hn_mean_rank"],
                "p50_ms": data["latency"]["p50_ms"],
                "p95_ms": data["latency"]["p95_ms"],
                "cost_usd": data["cost"]["estimated_cost_usd"],
                "price_per_1m_usd": data["cost"]["price_per_1m_usd"],
            }
        )
    rows.sort(key=lambda r: r["mrr"], reverse=True)
    return rows


def write_markdown_summary(rows: list[dict], out_path: Path) -> None:
    headers = [
        "Model", "Dim (used/native)", "R@1", "R@5", "MRR",
        "HN rank", "p50 ms", "p95 ms", "$/run", "$/1M",
    ]
    lines: list[str] = []
    lines.append("# Embedding benchmark — summary\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            f"| {r['model_slug']} | {r['dim_used']}/{r['dim_native']} | "
            f"{r['recall_at_1']:.3f} | {r['recall_at_5']:.3f} | {r['mrr']:.3f} | "
            f"{r['hn_mean_rank']:.2f} | {r['p50_ms']:.0f} | {r['p95_ms']:.0f} | "
            f"{_fmt_cost(r['cost_usd'])} | {_fmt_cost(r['price_per_1m_usd'])} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_cost(val: float | None) -> str:
    if val is None:
        return "local"
    return f"${val:.5f}" if val < 0.01 else f"${val:.4f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RESULTS_DIR / "summary.md")
    parser.add_argument("--json", type=Path, default=RESULTS_DIR / "summary.json")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    rows = aggregate_results(args.results_dir)
    args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    write_markdown_summary(rows, args.out)
    print(f"wrote {args.out} and {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd .benchmark/embedding && pytest tests/test_analyze.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `cd .benchmark/embedding && pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Notify user — Task 12 complete, ready for commit.**

---

## Task 13: Execution — dry run for all models

This is the mandatory dry-run validation before any paid API call.

- [ ] **Step 1: Ensure `.env` has every available API key**

Check that `.benchmark/embedding/.env` exists and contains the keys the user has access to. Models whose key is missing will skip with a warning — this is expected.

- [ ] **Step 2: Run the dry run**

Run: `cd .benchmark/embedding && python run_benchmark.py --dry-run`

Expected: for each model with a resolvable API key, a line like:
```
=== openai-3-large (openai / text-embedding-3-large) ===
  dry-run ok — canary shape (3, 3072), p50 latency 180.4 ms
  full-run cost estimate: $0.00073
```

For missing keys:
```
=== voyage-3-large (voyage / voyage-3-large) ===
  skipped: VOYAGE_API_KEY is not set — add it to .env or the environment
```

At the end:
```
Total estimated hosted cost for a full run: $0.00450
```

- [ ] **Step 3: USER GATE — ask to switch to max-reflexion mode**

**Stop.** Notify the user: dry run succeeded, dataset is reviewed, ready for the real hosted run. The user should now switch me to max reflexion mode per their earlier request before I proceed to Task 14.

- [ ] **Step 4: Notify user — Task 13 complete, awaiting go-ahead for live run.**

---

## Task 14: Execute hosted phase

- [ ] **Step 1: Run the hosted phase**

Run: `cd .benchmark/embedding && python run_benchmark.py --phase hosted`

Expected: one `results/<slug>_dim<N>.json` file per enabled hosted model × each dim in `dim_runs`. For models with a 2-entry `dim_runs` (openai-3-large/small, voyage, gemini), two files.

- [ ] **Step 2: Sanity-check the result files**

Run: `ls .benchmark/embedding/results/`
Expected: ~10 new JSON files (if all 5 hosted vendors are configured).

For each file, verify metrics are sane (recall@1 between 0 and 1, MRR > 0):

Run: `cd .benchmark/embedding && python -c "
import json, pathlib
for fp in sorted(pathlib.Path('results').glob('*.json')):
    if fp.name == 'summary.json': continue
    d = json.loads(fp.read_text())
    print(f\"{d['model_slug']}@{d['dim_used']}: R@1={d['metrics']['recall_at_1']:.3f} MRR={d['metrics']['mrr']:.3f}\")
"`

Expected: one line per result file with reasonable-looking numbers.

- [ ] **Step 3: Notify user — Task 14 complete, hosted results in hand.**

---

## Task 15: Execute local phase

- [ ] **Step 1: Ensure the local backends are ready**

- Ollama running at `http://127.0.0.1:11434` with models `qwen3-embedding-8b` and `qwen3-embedding-4b` pulled
- sentence-transformers models (`BAAI/bge-m3`, `jinaai/jina-embeddings-v3`, `nomic-ai/nomic-embed-text-v1.5`, `NovaSearch/stella_en_1.5B_v5`) will auto-download on first use — this may take time and disk

Run: `ollama list`
Expected: the Qwen3 embedding models listed.

- [ ] **Step 2: Run the local phase**

Run: `cd .benchmark/embedding && python run_benchmark.py --phase local`

Expected: one result file per local model. First run will download sentence-transformers weights — this is expected.

- [ ] **Step 3: Notify user — Task 15 complete, local results in hand.**

---

## Task 16: Analysis writeup

- [ ] **Step 1: Generate the comparative summary**

Run: `cd .benchmark/embedding && python analyze.py`
Expected: `results/summary.json` and `results/summary.md` created.

- [ ] **Step 2: Author the human analysis doc**

Create `.docs/benchmark/2026-04-11-embedding-benchmark-analysis.md` with the same structure as the existing extraction analysis document. Sections:

1. **TL;DR** — recommended hosted / recommended local / reference ceiling / models to avoid
2. **Comparative table** — reuse the markdown table from `summary.md`, add columns for short names / provider
3. **Per-category analysis** — breakdown by `user_fact` / `temporal` / `negation` / `pronoun` / `mixed_lang` — which models crater where
4. **Matryoshka truncation impact** — for each dual-dim model, delta in recall@1 / MRR between native and 1024
5. **Cost vs quality frontier** — simple scatter/commentary
6. **Recommendation for graph-mem** — concrete `.env` values for the production deployment (`EMBEDDING_MODEL_NAME`, `OPENAI_BASE_URL`, dim)
7. **Limitations and caveats** — single-author dataset, 100-doc corpus size, HN selection bias

- [ ] **Step 3: Cross-reference from `CLAUDE.md`**

Add a one-line pointer in `CLAUDE.md` under "Docker Stack" or a new "Embeddings" subsection mentioning the analysis file and the recommended default.

- [ ] **Step 4: Notify user — Task 16 complete, full benchmark done and documented.**

---

## Self-review (already performed during plan authoring)

**Spec coverage check (every section of the spec has a task):**
- Section 3 (methodology) → Tasks 3, 4 (ranking + metrics)
- Section 4 (dataset) → Task 11
- Section 5 (models) → Task 10 (models.yaml)
- Section 6 (harness file layout) → Tasks 1–10
- Section 6.3 (adapters) → Tasks 5, 6, 7, 8
- Section 6.4 (run flow) → Task 9 (runner)
- Section 6.5 (CLI) → Task 10
- Section 6.6 (analysis step) → Tasks 12, 16
- Section 7 (cost safeguards) → Task 13 (dry-run gate)
- Section 8 (reproducibility) → Task 9 (`_environment_snapshot`)
- Section 9 (deliverables) → Tasks 14, 15, 16
- Section 10 (execution gates) → Tasks 11 (dataset review), 13 (dry-run gate)
- Section 11 (open questions) → documented in final analysis (Task 16)

No gaps.

**Placeholder scan:** every code step has full code. Task 11 (dataset authoring) is the only task that intentionally defers content — it is a human authoring task, not a code task.

**Type / signature consistency checked:**
- `EmbedAdapter.embed(texts)` signature consistent across all adapters
- `EmbedResult` dataclass used consistently
- `QueryEval` fields used consistently between metrics and runner
- `RunConfig` fields used consistently between tests and runner
- `ModelEntry` fields used consistently between YAML loader, adapter factory, and tests

No inconsistencies found.
