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
