"""Shared HTTP utilities for embedding adapters.

Provides a default httpx.Client with sane timeouts and the L2-normalise
post-process that every adapter applies to its raw response.

SSL verification is on by default. Set the env var `BENCHMARK_SSL_VERIFY=false`
to disable it — required on corporate networks (e.g. Michelin proxy) that
intercept HTTPS traffic with a self-signed root.
"""
from __future__ import annotations

import os

import httpx
import numpy as np

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0)


def _ssl_verify() -> bool:
    return os.environ.get("BENCHMARK_SSL_VERIFY", "true").lower() not in (
        "0", "false", "no", "off",
    )


def default_client() -> httpx.Client:
    return httpx.Client(timeout=DEFAULT_TIMEOUT, verify=_ssl_verify())


def l2_normalise_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-12, None)).astype(np.float32)
