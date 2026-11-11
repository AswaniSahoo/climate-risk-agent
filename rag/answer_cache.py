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
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, model_validator

from agent.contracts import Hazard, ProjectedChange
from rag.answer import CitedAnswer, _instructions
from rag.chunk import Chunk
from rag.corpus import _fingerprint as _corpus_fingerprint
from rag.embed import DIMS
from rag.embed import MODEL as EMBED_MODEL
from rag.gemini_client import GENERATE_MODEL
from tools.ar6_regions import AR6Region
from tools.cache_backend import CacheBackend, DiskCache, JsonCache, source_bytes

_log = logging.getLogger(__name__)

# The corpus is frozen, but a cached answer still embeds one model version and
# one prompt. 30 days bounds how long a superseded pairing can keep answering.
_ANSWER_TTL_S = 30 * 24 * 3600

# A grounding or projection entry carries every version that decides it in its
# KEY (corpus, ranking code, prompt, model), so a change to any of them asks
# again instead of replaying. What is left for the TTL to bound is hygiene,
# the same reasoning as the 365-day ERA5 fit cache in tools/climatology.py.
_GROUNDING_TTL_S = 365 * 24 * 3600

# Files whose content decides which chunks come back for a question, and files
# whose content decides what the answerer is asked. Hashed from source, the
# same way rag/corpus.py fingerprints the chunker: nobody has to remember to
# bump a version constant.
_RETRIEVAL_SOURCES = ("retrieve.py", "bm25.py", "dense.py", "hybrid.py")
_PROMPT_SOURCES = ("answer.py",)
_PROJECTION_SOURCES = ("cid.py",)
_SOURCE_DIR = Path(__file__).resolve().parent


@lru_cache(maxsize=8)
def _source_digest(names: tuple[str, ...]) -> str:
    """Hash of these rag/ sources, read with LF line endings (a CRLF checkout agrees)."""
    digest = hashlib.sha256()
    for name in names:
        digest.update(source_bytes(_SOURCE_DIR / name))
    return digest.hexdigest()[:16]


def corpus_version() -> str:
    """The chunked corpus: PDF names and sizes plus the chunker's source."""
    return _corpus_fingerprint()


def retrieval_version() -> str:
    """The ranking code plus the embedding model and width the index was built with."""
    return f"{_source_digest(_RETRIEVAL_SOURCES)}|{EMBED_MODEL}|{DIMS}"


def prompt_version() -> str:
    """The answer prompt as this process would send it.

    The source hash covers the rules, the response schema and the excerpt
    framing. The rendered rule block is added because CRG_REGIONAL_GROUNDING
    changes the prompt at runtime without changing a byte of source.
    """
    digest = hashlib.sha256(_source_digest(_PROMPT_SOURCES).encode())
    digest.update(_instructions().encode())
    return digest.hexdigest()[:16]


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


class Grounding(BaseModel):
    """What the research step produced for one (hazard, AR6 area).

    Only the chunks the answer CITES are kept: they are all the report needs
    (agent/graph.py maps each citation to its page, and the Jev check reads the
    cited text), and a full eight-chunk retrieval would triple the size of every
    entry that ships in the image. `answer.allowed_ids` still lists everything
    that was retrieved, so the citation-integrity rule keeps its full meaning.
    """

    hazard: str
    area: str
    question: str
    answer: CitedAnswer
    cited_chunks: list[Chunk]

    @classmethod
    def from_research(
        cls, *, hazard: str, area: str, question: str, answer: CitedAnswer,
        chunks: Sequence[Chunk],
    ) -> "Grounding":
        by_id = {c.chunk_id: c for c in chunks}
        cited = [by_id[chunk_id] for chunk_id in dict.fromkeys(answer.citations)]
        return cls(hazard=hazard, area=area, question=question, answer=answer, cited_chunks=cited)

    @model_validator(mode="after")
    def _check(self) -> "Grounding":
        if self.answer.abstain:
            # One abstention is one sample of a non-deterministic model
            # (module docstring): pinning it would refuse a whole region.
            raise ValueError("an abstention is never stored as a grounding")
        stored = {c.chunk_id for c in self.cited_chunks}
        missing = [c for c in self.answer.citations if c not in stored]
        if missing:
            raise ValueError(f"cited chunks {missing} are not stored with the grounding")
        return self


class GroundingCache:
    """The research step's result, one entry per (hazard, area) and version.

    agent/graph.py asks `_ipcc_question(hazard, _ipcc_area(state))`, and the
    area is the AR6 region, never the city, so every city in a region shares
    one entry. On Cloud Run at min-instances 0 the shared backend is the only
    thing between a cold visitor and a retrieval plus a model call; with no
    Redis configured, the committed seed (tools/cache_seed/) is.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        backend: CacheBackend | None = None,
    ) -> None:
        if backend is None and directory is not None:
            backend = DiskCache(directory)
        self._cache = JsonCache("grounding", backend=backend)

    @property
    def json_cache(self) -> JsonCache:
        return self._cache

    def key(self, *, hazard: str, area: str, question: str, top_k: int) -> str:
        raw = "|".join([
            hazard, area, question, f"top_k={top_k}",
            f"corpus={corpus_version()}", f"retrieval={retrieval_version()}",
            f"prompt={prompt_version()}", f"model={GENERATE_MODEL}",
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Grounding | None:
        return self._cache.get_model(key, Grounding)

    def put(self, key: str, grounding: Grounding) -> None:
        self._cache.set_model(key, grounding, ttl_s=_GROUNDING_TTL_S)


class ProjectionResult(BaseModel):
    """The project step's result for one (region, hazard): a projection, or None
    when the corpus holds no qualifying Ch.12 sentence. Both are answers."""

    change: ProjectedChange | None


class ProjectionCache:
    """rag/cid.py's projection, one entry per (AR6 region, hazard) and version.

    The projection is retrieval plus regex, no generation, so it is
    deterministic for a given corpus, ranking code and rag/cid.py: caching the
    "not found" outcome is as safe as caching a found one.
    """

    def __init__(
        self,
        directory: Path | str | None = None,
        *,
        backend: CacheBackend | None = None,
    ) -> None:
        if backend is None and directory is not None:
            backend = DiskCache(directory)
        self._cache = JsonCache("projection", backend=backend)

    @property
    def json_cache(self) -> JsonCache:
        return self._cache

    def key(self, region: AR6Region, hazard: Hazard) -> str:
        from rag.cid import DEFAULT_TOP_K, cid_question

        raw = "|".join([
            region.acronym, region.name, hazard.value, cid_question(region, hazard),
            f"top_k={DEFAULT_TOP_K}", f"corpus={corpus_version()}",
            f"retrieval={retrieval_version()}", f"cid={_source_digest(_PROJECTION_SOURCES)}",
        ])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> ProjectionResult | None:
        return self._cache.get_model(key, ProjectionResult)

    def put(self, key: str, result: ProjectionResult) -> None:
        self._cache.set_model(key, result, ttl_s=_GROUNDING_TTL_S)
