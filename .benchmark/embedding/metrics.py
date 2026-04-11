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
