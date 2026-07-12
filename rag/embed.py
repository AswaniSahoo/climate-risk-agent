"""Gemini embeddings with a disk cache (SDK seam: rag/gemini_client.py).

Asymmetric task types are the point: passages embed as RETRIEVAL_DOCUMENT and
questions as RETRIEVAL_QUERY — the model is trained to land paraphrased
questions near jargon-heavy passages (the "1-day rainfall" vs "Rx1day" gap that
lexical BM25 cannot cross).

Vectors are MRL-truncated to 768 dims (near-equal retrieval quality, small
cache, instant brute-force cosine at our corpus size). The disk cache keys on
(model | dims | task_type | text), so after one indexing pass the eval is
offline and deterministic.

Pacing: free-tier AI-Studio keys allow 100 embed-requests/min with every batch
item counted (measured), so defaults are batch-50 + 62 s pause. Paid/Vertex:
set EMBED_BATCH_PAUSE_S=0. Batches persist to the cache as they complete, so an
interrupted run resumes instead of re-paying.
"""
from __future__ import annotations

import hashlib
import os
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.gemini_client import EMBED_MODEL, GeminiError, embed_batch

MODEL = EMBED_MODEL
DIMS = 768
_BATCH_SIZE = 50
_BATCH_PAUSE_S = float(os.environ.get("EMBED_BATCH_PAUSE_S", "62"))

_sleep = time.sleep  # module-level so tests can stub the pacing out


class EmbeddingError(RuntimeError):
    """Raised when embedding fails (auth, quota after retries, API error)."""


def embed_texts(texts: list[str], *, task_type: str) -> list[list[float]]:
    """Embed texts in paced batches via the SDK seam."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        try:
            vectors.extend(embed_batch(batch, task_type=task_type, dims=DIMS))
        except GeminiError as exc:
            raise EmbeddingError(str(exc)) from exc
        if start + _BATCH_SIZE < len(texts):
            _sleep(_BATCH_PAUSE_S)
    return vectors


class DiskVectorCache:
    """One .npy file per vector, filename = sha256(cache key)."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / (hashlib.sha256(key.encode("utf-8")).hexdigest() + ".npy")

    def get(self, key: str) -> list[float] | None:
        path = self._path(key)
        return np.load(path).tolist() if path.exists() else None

    def put(self, key: str, vector: list[float]) -> None:
        np.save(self._path(key), np.asarray(vector, dtype=np.float32))


def cache_key(text: str, task_type: str) -> str:
    return f"{MODEL}|{DIMS}|{task_type}|{text}"


def cached_embed_texts(
    texts: list[str], *, task_type: str, cache: DiskVectorCache
) -> list[list[float]]:
    """Embed texts through the disk cache: only cache misses hit the network."""

    vectors: list[list[float] | None] = [cache.get(cache_key(t, task_type)) for t in texts]
    missing = [i for i, v in enumerate(vectors) if v is None]

    if not missing:
        print("All embeddings loaded from cache.")
        return vectors  # type: ignore[return-value]

    print(
        f"Cache hits: {len(texts) - len(missing)} | "
        f"Need to embed: {len(missing)}"
    )

    progress = tqdm(
        total=len(missing),
        desc=f"Embedding ({task_type})",
        unit="chunks",
    )

    for start in range(0, len(missing), _BATCH_SIZE):
        batch = missing[start : start + _BATCH_SIZE]
        try:
            fetched = embed_batch(
                [texts[i] for i in batch],
                task_type=task_type,
                dims=DIMS,
            )
        except GeminiError as exc:
            progress.close()
            raise EmbeddingError(str(exc)) from exc
        for i, vector in zip(batch, fetched):
            cache.put(cache_key(texts[i], task_type), vector)
            vectors[i] = vector

        progress.update(len(batch))
        if start + _BATCH_SIZE < len(missing):
            _sleep(_BATCH_PAUSE_S)
    progress.close()

    return vectors  # type: ignore[return-value]