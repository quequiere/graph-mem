"""Cosine similarity, L2 normalisation, and ranking utilities.

All functions are pure and accept numpy arrays. They do not touch I/O.
"""
from __future__ import annotations

import numpy as np


def l2_normalize(mat: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.clip(norms, eps, None)


def cosine_matrix(queries: np.ndarray, corpus: np.ndarray) -> np.ndarray:
    """Cosine similarity matrix of shape (Q, C).

    Assumes both inputs are L2-normalised -- callers are responsible for
    normalisation. This keeps the math a single matrix multiply.
    """
    return queries @ corpus.T


def rank_ids(scores: np.ndarray, ids: list[str]) -> list[str]:
    """Return ids sorted from highest to lowest score.

    Stable with respect to the input order on ties.
    """
    order = np.argsort(-scores, kind="stable")
    return [ids[i] for i in order]


def truncate_and_renormalize(mat: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka-style slice to `dim` then L2-renormalise."""
    if dim > mat.shape[1]:
        raise ValueError(
            f"requested dim {dim} is larger than vector dim {mat.shape[1]}"
        )
    return l2_normalize(mat[:, :dim])
