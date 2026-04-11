"""Benchmark runner: embed corpus + queries, compute metrics, dump JSON.

The runner receives an already-constructed adapter, so it is fully
testable with any adapter. It does not touch environment variables.
"""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

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

    # --- latency stats
    all_latencies = list(corpus_res.latencies_ms) + list(query_res.latencies_ms)
    latency = {
        "p50_ms": _percentile(all_latencies, 50),
        "p95_ms": _percentile(all_latencies, 95),
        "corpus_wall_ms": corpus_wall_ms,
        "query_wall_ms": query_wall_ms,
    }

    # --- cost estimate (rough: 1 token ~ 4 chars)
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
