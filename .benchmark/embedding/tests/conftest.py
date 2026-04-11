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
