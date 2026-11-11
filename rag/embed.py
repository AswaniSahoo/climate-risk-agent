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

The image also carries a PACK: the corpus vectors in one file beside the
per-vector folder, read before the files (see `pack_path`).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import threading
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from tqdm import tqdm

from rag.gemini_client import EMBED_MODEL, GeminiError, embed_batch

_log = logging.getLogger(__name__)

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


# The vector pack: the corpus vectors in ONE file beside the per-vector folder
# (data/cache/embeddings -> data/cache/embeddings.pack.npy), written at image
# build by scripts/pack_vectors.py. Why: Cloud Run gen2 pulls image files on
# first read, and the first live IPCC answer of a process spent 17.5 s reading
# the 2,730 vector files (revision 00031, 2026-11-11) that a local disk reads in
# 0.15 s. Each row is (sha256 hex of the cache key, float32 vector), the same
# content address the files use, so a stale pack can only miss, never answer
# for another text.
PACK_SUFFIX = ".pack.npy"
_DIGEST = re.compile(r"[0-9a-f]{64}")


def pack_path(directory: Path | str) -> Path:
    """Where the pack of a vector folder lives: beside the folder, named after it."""
    folder = Path(directory)
    if folder.name in {"", ".."}:  # "." or "..": name the pack after the folder they mean
        folder = folder.resolve()
    return folder.with_name(folder.name + PACK_SUFFIX)


def vector_digest(key: str) -> str:
    """The name a vector is stored under: the sha256 of its cache key, in hex."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def write_pack(path: Path | str, vectors: Mapping[str, np.ndarray]) -> int:
    """Write {digest: vector} as one structured .npy file, replaced atomically.

    Refuses rather than casts: every vector must already be 1-D float32 of one
    width, as `DiskVectorCache.put` writes them, so the pack holds the files'
    exact bits. Returns the number of rows written.
    """
    if not vectors:
        raise ValueError("nothing to pack")
    width: int | None = None
    for digest, vector in vectors.items():
        if not _DIGEST.fullmatch(digest):
            raise ValueError(f"pack keys are sha256 hex digests, got {digest!r}")
        if vector.dtype != np.dtype("<f4") or vector.ndim != 1:
            raise ValueError(f"{digest}: a 1-D float32 vector is required, "
                             f"got {vector.dtype} {vector.shape}")
        if width is None:
            width = vector.shape[0]
        elif vector.shape[0] != width:
            raise ValueError(f"{digest}: width {vector.shape[0]}, the other vectors have {width}")
    assert width is not None
    table = np.empty(len(vectors), dtype=[("key", "S64"), ("vector", "<f4", (width,))])
    table["key"] = [digest.encode("ascii") for digest in vectors]
    table["vector"] = np.stack(list(vectors.values()))
    target = Path(path)
    partial = target.with_name(target.name + ".partial")
    try:
        with partial.open("wb") as handle:  # a file object: np.save appends no ".npy"
            np.save(handle, table, allow_pickle=False)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return len(table)


def read_pack(path: Path | str) -> dict[str, np.ndarray]:
    """The pack at `path` as {digest: vector}, read from disk in one go. Raises on a bad file."""
    table = np.load(Path(path), allow_pickle=False)
    if not isinstance(table, np.ndarray):  # an .npz archive, not a pack
        table.close()
        raise ValueError("not a vector pack: an .npz archive")
    layout = table.dtype
    if (table.ndim != 1 or layout.names != ("key", "vector")
            or layout["key"] != np.dtype("S64") or layout["vector"].base != np.dtype("<f4")
            or len(layout["vector"].shape) != 1):
        raise ValueError(f"not a vector pack: dtype {layout}, shape {table.shape}")
    vectors = np.ascontiguousarray(table["vector"])
    return {key.decode("ascii"): row for key, row in zip(table["key"].tolist(), vectors)}


_packs: dict[Path, tuple[tuple[int, int, int, int], Mapping[str, np.ndarray]]] = {}
_packs_lock = threading.Lock()


def load_pack(path: Path | str) -> Mapping[str, np.ndarray]:
    """`read_pack` once per process for each version of the file, and never raising.

    No file: empty, so every lookup goes to the vector files. A file that does
    not read as a pack: one warning and empty, the same fallback, so a damaged
    pack costs the slow path, never a wrong vector or a crash.
    """
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        return {}
    stamp = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    with _packs_lock:
        held = _packs.get(target)
        if held is not None and held[0] == stamp:
            return held[1]
        packed: Mapping[str, np.ndarray]
        try:
            packed = read_pack(target)
        except Exception as exc:  # noqa: BLE001 - a bad pack must degrade, never crash
            _log.warning("vector pack %s unreadable (%s): reading the vector files instead",
                         target, exc)
            packed = {}
        _packs[target] = (stamp, packed)
        return packed


class DiskVectorCache:
    """One .npy file per vector (named by `vector_digest`), read through the pack.

    `get` answers from the pack (`pack_path(directory)`) when it holds the key,
    and from the vector file otherwise, so a folder without a pack reads as it
    always has. `put` only ever writes a vector file: a newly embedded question
    never touches the pack.
    """

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.pack = pack_path(self.directory)
        self._packed: Mapping[str, np.ndarray] | None = None  # loaded on the first get

    def _path(self, key: str) -> Path:
        return self.directory / (vector_digest(key) + ".npy")

    def get(self, key: str) -> list[float] | None:
        if self._packed is None:
            self._packed = load_pack(self.pack)
        digest = vector_digest(key)
        packed = self._packed.get(digest)
        if packed is not None:
            return packed.tolist()
        path = self.directory / (digest + ".npy")
        return np.load(path).tolist() if path.exists() else None

    def file_vector(self, key: str) -> np.ndarray | None:
        """The vector file's own array, never the pack's (the build check compares the two)."""
        path = self._path(key)
        return np.load(path, allow_pickle=False) if path.exists() else None

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
        _log.info("all %d embeddings loaded from cache", len(texts))
        return vectors  # type: ignore[return-value]

    _log.info("cache hits: %d | need to embed: %d", len(texts) - len(missing), len(missing))

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