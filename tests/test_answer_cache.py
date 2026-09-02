"""Tests for the disk answer cache (rag/answer_cache.py).

Why a cache is CORRECT here (and Redis is not): the corpus is frozen and
generation runs at temperature 0, so (question, retrieved chunks, model) fully
determines the answer — same key, same CitedAnswer, forever. A repeat query
should cost zero tokens. Single-process app -> disk file, not a cache server.
"""
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


def test_corrupt_cache_file_is_a_loud_miss(tmp_path, caplog):
    cache = AnswerCache(tmp_path)
    key = cache.key("q", _CHUNKS)
    cache.put(key, _ANSWER)
    # Since Cache v2 the on-disk name is sha256 of the NAMESPACED key, so find
    # the entry rather than guessing its filename.
    [entry] = list(tmp_path.glob("*.json"))
    entry.write_text("{not json", encoding="utf-8")

    assert cache.get(key) is None  # degrade to a miss, never crash
    assert "corrupt cache entry" in caplog.text  # loud (WARNING), not silent


def test_cache_accepts_a_shared_backend_and_sets_a_bounded_ttl():
    """Cache v2: the same entries can live in the replica-shared tier.

    A cached answer pins one model version, so it must not live forever — the
    30-day TTL is what retires a superseded (prompt, model) pairing.
    """
    from rag.answer_cache import _ANSWER_TTL_S

    class _RecordingBackend:
        name = "redis"

        def __init__(self):
            self.store, self.ttls = {}, []

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, ttl_s=None):
            self.store[key] = value
            self.ttls.append(ttl_s)

    backend = _RecordingBackend()
    cache = AnswerCache(backend=backend)
    key = cache.key("how much?", _CHUNKS)

    cache.put(key, _ANSWER)

    assert cache.get(key) == _ANSWER
    assert backend.ttls == [_ANSWER_TTL_S] and _ANSWER_TTL_S == 30 * 24 * 3600
    assert list(backend.store)[0].startswith("crg:v1:answers:")  # namespaced, not bare


def test_a_hit_records_exactly_one_telemetry_event(tmp_path):
    """A hit used to book TWO events: "cache:answers" from the shared cache
    layer plus a bespoke "generate"/cached=True on top. Both carry cached=True,
    so every cached answer counted twice in Span.summary()["cache_hits"] and in
    the UI's cache badge."""
    from obs.telemetry import Span, snapshot

    cache = AnswerCache(tmp_path)
    key = cache.key("how much?", _CHUNKS)
    cache.put(key, _ANSWER)

    with Span("report") as span:
        assert cache.get(key) == _ANSWER

    events = [e for e in snapshot() if e.get("cached")]
    assert len(events) == 1
    assert events[0]["op"] == "cache:answers"  # the tier is named; the badge works
    assert span.summary()["cache_hits"] == 1

