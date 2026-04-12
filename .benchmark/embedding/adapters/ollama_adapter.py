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
