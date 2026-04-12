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
