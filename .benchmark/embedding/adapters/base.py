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
    latencies_ms: list[float]  # one per input text
    model: str  # canonical model id as reported by the backend


class EmbedAdapter(Protocol):
    name: str  # adapter slug

    def embed(self, texts: list[str]) -> EmbedResult: ...
