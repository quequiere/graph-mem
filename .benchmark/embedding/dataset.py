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
