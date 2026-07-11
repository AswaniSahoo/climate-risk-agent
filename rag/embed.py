"""Gemini embeddings (gemini-embedding-2) with a disk cache.

Asymmetric task types are the point: passages embed as RETRIEVAL_DOCUMENT and
questions as RETRIEVAL_QUERY — the model is trained to land paraphrased
questions near jargon-heavy passages (the "1-day rainfall" vs "Rx1day" gap that
lexical BM25 cannot cross).

Vectors are MRL-truncated to 768 dims (near-equal retrieval quality, small
cache, instant brute-force cosine at our corpus size). The disk cache keys on
(model | dims | task_type | text), so after one live indexing pass the eval is
offline and deterministic. Host is hardcoded (SSRF guard); key comes from the
GEMINI_API_KEY env var only.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
from pathlib import Path

import httpx
import numpy as np

MODEL = "gemini-embedding-2"
EMBED_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:batchEmbedContents"
DIMS = 768
# Free tier: 100 embed-content requests/minute, and EVERY text in a batch counts
# as one request (measured from the 429 quota metadata). A batch of 100 fills the
# whole window and gets rejected whenever anything else touched it (measured:
# batch-100 always 429, batch-50 passes) — so half-window batches, one per minute.
_BATCH_SIZE = 50
_BATCH_PAUSE_S = 62
_RETRY_MAX = 8
_RETRY_IN = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)

_sleep = time.sleep  # module-level so tests can stub the waiting out


class EmbeddingError(RuntimeError):
    """Raised when the embedding API request fails."""


def embed_texts(
    texts: list[str], *, task_type: str, api_key: str | None = None
) -> list[list[float]]:
    """Embed texts in batches of 100. task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY."""
    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise EmbeddingError("GEMINI_API_KEY is not set")

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        payload = {
            "requests": [
                {
                    "model": f"models/{MODEL}",
                    "content": {"parts": [{"text": text}]},
                    "taskType": task_type,
                    "outputDimensionality": DIMS,
                }
                for text in batch
            ]
        }
        response = _post_with_retry(payload, key)
        vectors.extend(item["values"] for item in response.json()["embeddings"])
        if start + _BATCH_SIZE < len(texts):
            _sleep(_BATCH_PAUSE_S)  # stay under 100 requests/minute
    return vectors


def _post_with_retry(payload: dict, key: str) -> httpx.Response:
    """POST, sleeping out 429s using the server-suggested delay."""
    for _ in range(_RETRY_MAX):
        try:
            response = httpx.post(
                EMBED_URL, headers={"x-goog-api-key": key}, json=payload, timeout=120
            )
            if response.status_code == 429:
                match = _RETRY_IN.search(response.text)
                _sleep(min(float(match.group(1)) + 1 if match else 60.0, 120.0))
                continue
            response.raise_for_status()
            return response
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc
    raise EmbeddingError("rate-limit retries exhausted (429)")


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
    """embed_texts through the disk cache: only cache misses hit the network.

    Misses are fetched AND persisted one batch at a time, so a mid-run failure
    (crash, daily quota) keeps every completed batch — the next run resumes
    from the cache instead of re-paying the quota.
    """
    vectors: list[list[float] | None] = [cache.get(cache_key(t, task_type)) for t in texts]
    missing = [i for i, v in enumerate(vectors) if v is None]
    for start in range(0, len(missing), _BATCH_SIZE):
        batch = missing[start : start + _BATCH_SIZE]
        fetched = embed_texts([texts[i] for i in batch], task_type=task_type)
        for i, vector in zip(batch, fetched):
            cache.put(cache_key(texts[i], task_type), vector)
            vectors[i] = vector
        print(f"  embedded {min(start + _BATCH_SIZE, len(missing))}/{len(missing)} misses", flush=True)
        if start + _BATCH_SIZE < len(missing):
            _sleep(_BATCH_PAUSE_S)
    return vectors  # type: ignore[return-value]
