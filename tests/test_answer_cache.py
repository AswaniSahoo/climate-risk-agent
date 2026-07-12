"""Tests for the disk answer cache (rag/answer_cache.py).

Why a cache is CORRECT here (and Redis is not): the corpus is frozen and
generation runs at temperature 0, so (question, retrieved chunks, model) fully
determines the answer — same key, same CitedAnswer, forever. A repeat query
should cost zero tokens. Single-process app -> disk file, not a cache server.
"""
import pytest

from rag.answer import CitedAnswer
from rag.answer_cache import AnswerCache
from rag.chunk import Chunk

_CHUNKS = (
    Chunk(chunk_id="doc.pdf#p5#0", source="doc.pdf", page=5, text="heavy precipitation +7%"),
    Chunk(chunk_id="doc.pdf#p9#0", source="doc.pdf", page=9, text="glaciers keep melting"),
)

_ANSWER = CitedAnswer(
    answer="It intensifies about 7% per degree.",
    citations=["doc.pdf#p5#0"], abstain=False,
    allowed_ids=[c.chunk_id for c in _CHUNKS],
)


def test_miss_then_hit_roundtrips_the_answer(tmp_path):
    cache = AnswerCache(tmp_path)
    key = cache.key("how much?", _CHUNKS)

    assert cache.get(key) is None  # cold
    cache.put(key, _ANSWER)
    hit = cache.get(key)
    assert hit == _ANSWER
    assert hit.citations == ["doc.pdf#p5#0"]


def test_key_depends_on_question_chunks_and_text(tmp_path):
    cache = AnswerCache(tmp_path)
    base = cache.key("how much?", _CHUNKS)

    assert cache.key("how much??", _CHUNKS) != base  # question changes key
    assert cache.key("how much?", _CHUNKS[:1]) != base  # chunk set changes key
    edited = (_CHUNKS[0].model_copy(update={"text": "edited"}), _CHUNKS[1])
    assert cache.key("how much?", edited) != base  # same ids, new TEXT -> new key


def test_corrupt_cache_file_is_a_loud_miss(tmp_path, capsys):
    cache = AnswerCache(tmp_path)
    key = cache.key("q", _CHUNKS)
    cache.put(key, _ANSWER)
    (tmp_path / f"{key}.json").write_text("{not json", encoding="utf-8")

    assert cache.get(key) is None  # degrade to a miss, never crash
    assert "answer cache" in capsys.readouterr().out.lower()  # loud, not silent
