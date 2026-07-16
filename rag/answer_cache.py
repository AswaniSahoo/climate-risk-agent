"""Disk cache for cited answers: same question + same evidence = same answer, free.

Why this exists (and why not Redis): the corpus is frozen and generation runs
at temperature 0, so (question, retrieved chunk texts, model) fully determines
the CitedAnswer — a repeat query should cost zero tokens and zero latency.
This is a single-process app (Streamlit / stdio MCP), so the right cache is a
directory of JSON files, not a cache server. Redis earns its place only when
multiple replicas need shared state (recorded in DEBT with that trigger).

The key hashes the chunk TEXTS, not just ids: a re-chunk that changes content
under a stable id invalidates naturally. Corrupt entries are a LOUD miss.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

from rag.answer import CitedAnswer
from rag.chunk import Chunk
from rag.gemini_client import GENERATE_MODEL

_log = logging.getLogger(__name__)


class AnswerCache:
    """sha256-keyed JSON files, one per (question, evidence, model)."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def key(self, question: str, chunks: Sequence[Chunk]) -> str:
        hasher = hashlib.sha256()
        hasher.update(GENERATE_MODEL.encode())
        hasher.update(question.encode())
        for chunk in chunks:
            hasher.update(chunk.chunk_id.encode())
            hasher.update(hashlib.sha256(chunk.text.encode()).digest())
        return hasher.hexdigest()

    def get(self, key: str) -> CitedAnswer | None:
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            answer = CitedAnswer.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # corrupt entry -> loud miss, never a crash
            _log.warning("corrupt cache entry %s ignored (%s)", path.name, exc)
            return None
        from obs.telemetry import record

        # a hit is a generate call that cost nothing — visible in the data
        record(op="generate", model=GENERATE_MODEL, latency_ms=0.0,
               tokens_in=0, tokens_out=0, retries=0, ok=True, cached=True)
        return answer

    def put(self, key: str, answer: CitedAnswer) -> None:
        path = self.directory / f"{key}.json"
        path.write_text(answer.model_dump_json(), encoding="utf-8")
