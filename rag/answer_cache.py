"""Cache for cited answers: a repeat query costs zero tokens and zero latency.

MEASURED CAVEAT (2026-07-22, evals/determinism_probe.py): temperature 0 does
NOT make generation deterministic. Repeating one fixed (question, chunks) pair
produced different answers from BOTH model families — gemini-2.5-flash gave 4
vs 5 citations across 5 runs, and gemini-3.6-flash varied on every run once it
stopped abstaining. So (question, chunk texts, model) does not *determine* the
CitedAnswer; it selects ONE VALID answer and pins it.

That is still the behaviour we want here — a user re-asking the same question
should get a stable reply rather than a reshuffled one — but it is a
consistency choice, not a correctness guarantee, and it is why the e2e eval
must never use this cache (it would freeze one sample of a distribution and
hide model drift). Run the probe again after any model change.

STORAGE, since Cache v2: this used to be a private directory of JSON files,
with a note saying Redis would earn its place only once multiple replicas
needed shared state. That trigger fired — Cloud Run, up to 2 replicas, an
ephemeral per-replica disk, and a redeploy on every push — so the entries now
go through `tools/cache_backend.py`: Upstash Redis when it is configured, the
same local disk when it is not. The key logic is untouched, and a corrupt entry
is still a LOUD miss rather than a crash.

The key hashes the chunk TEXTS, not just ids: a re-chunk that changes content
under a stable id invalidates naturally.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from rag.answer import CitedAnswer
from rag.chunk import Chunk
from rag.gemini_client import GENERATE_MODEL
from tools.cache_backend import CacheBackend, DiskCache, JsonCache

_log = logging.getLogger(__name__)

# The corpus is frozen, but a cached answer still embeds one model version and
# one prompt. 30 days bounds how long a superseded pairing can keep answering.
_ANSWER_TTL_S = 30 * 24 * 3600


class AnswerCache:
    """sha256-keyed entries, one per (question, evidence, model).

    `directory` pins the cache to a local folder (what the tests and offline
    runs want); passing nothing takes the process-wide backend from the
    environment, which is how the deployed app gets the shared Redis tier.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        backend: CacheBackend | None = None,
    ) -> None:
        if backend is None and directory is not None:
            backend = DiskCache(directory)
        self._cache = JsonCache("answers", backend=backend)

    def key(self, question: str, chunks: Sequence[Chunk]) -> str:
        hasher = hashlib.sha256()
        hasher.update(GENERATE_MODEL.encode())
        hasher.update(question.encode())
        for chunk in chunks:
            hasher.update(chunk.chunk_id.encode())
            hasher.update(hashlib.sha256(chunk.text.encode()).digest())
        return hasher.hexdigest()

    def get(self, key: str) -> CitedAnswer | None:
        # ONE telemetry event per read, and the shared cache layer already emits
        # it: `JsonCache.get_model` records op "cache:answers" with cached=True
        # on a hit and the tier that served it. This used to add a second
        # `generate`/cached=True event on top, which double-counted every hit in
        # `Span.summary()["cache_hits"]` and in the UI's cache badge.
        return self._cache.get_model(key, CitedAnswer)

    def put(self, key: str, answer: CitedAnswer) -> None:
        self._cache.set_model(key, answer, ttl_s=_ANSWER_TTL_S)
