"""Brute-force cosine index over chunk embeddings.

Exact search, no ANN: at ~2,700 vectors a full dot product is instant, and
exactness means recall numbers measure the EMBEDDINGS, not an index's
approximation error (per the design rule: no quantized/ANN index in v1).
"""
from __future__ import annotations

import numpy as np

from rag.chunk import Chunk


class DenseIndex:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray):
        if len(chunks) != matrix.shape[0]:
            raise ValueError("one embedding row per chunk required")
        self.chunks = chunks
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        self._unit = matrix / np.maximum(norms, 1e-12)  # cosine = dot of unit vectors

    def query(self, vector: list[float], top_k: int = 10) -> list[tuple[Chunk, float]]:
        q = np.asarray(vector, dtype=np.float64)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self._unit @ q
        order = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]
