"""sentence-transformers adapter — loads HuggingFace models locally.

Imports sentence_transformers lazily so the rest of the suite can run
without torch installed. Consumers will hit the RuntimeError only when
they actually try to use this adapter.
"""
from __future__ import annotations

import time

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # lazily surfaced
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
