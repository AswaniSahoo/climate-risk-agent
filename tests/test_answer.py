"""Tests for the cited LLM answerer (rag/answer.py). SDK seam mocked; validators pure.

The trust core: a citation that doesn't reference a retrieved chunk_id cannot
even be constructed — fabricated citations are a schema violation, not a vibe.
"""
import json

import pytest
from pydantic import ValidationError

import rag.answer as answer_mod
from rag.answer import AnswerError, CitedAnswer, answer_question, answer_with_guard
from rag.chunk import Chunk
from rag.gemini_client import GeminiError

ALLOWED = ["a.pdf#p1#0", "a.pdf#p2#0"]


def _chunk(cid: str, text: str = "some evidence") -> Chunk:
    source, page, idx = cid.split("#")
    return Chunk(chunk_id=cid, source=source, page=int(page[1:]), text=text)


def _llm_json(**overrides) -> str:
    payload = {
        "answer": "Heavy precipitation intensifies about 7% per 1°C.",
        "citations": ["a.pdf#p1#0"],
        "abstain": False,
        "abstain_reason": "",
    }
    payload.update(overrides)
    return json.dumps(payload)


# --- validation invariants (pure) ---


def test_valid_cited_answer_parses():
    answer = CitedAnswer.from_llm_json(_llm_json(), allowed_ids=ALLOWED)
    assert answer.citations == ["a.pdf#p1#0"]
    assert answer.abstain is False


def test_citation_outside_retrieved_set_is_rejected():
    raw = _llm_json(citations=["b.pdf#p9#0"])  # fabricated locator
    with pytest.raises(ValidationError, match="not among the retrieved chunks"):
        CitedAnswer.from_llm_json(raw, allowed_ids=ALLOWED)


def test_answer_without_citations_is_rejected():
    raw = _llm_json(citations=[])
    with pytest.raises(ValidationError, match="must cite"):
        CitedAnswer.from_llm_json(raw, allowed_ids=ALLOWED)


def test_grounded_abstention_may_carry_citations():
    raw = _llm_json(abstain=True, abstain_reason="premise contradicts the report",
                    citations=["a.pdf#p2#0"], answer="")
    answer = CitedAnswer.from_llm_json(raw, allowed_ids=ALLOWED)
    assert answer.abstain is True
    assert answer.citations == ["a.pdf#p2#0"]


def test_malformed_llm_json_is_a_typed_error():
    with pytest.raises(AnswerError):
        CitedAnswer.from_llm_json("not json at all", allowed_ids=ALLOWED)


# --- the seam edge (mocked) ---


def test_answer_question_round_trip(monkeypatch):
    seen = {}

    def fake_generate(prompt, *, schema):
        seen["prompt"] = prompt
        return _llm_json()

    monkeypatch.setattr(answer_mod, "generate_json", fake_generate)
    chunks = [_chunk(cid) for cid in ALLOWED]

    answer = answer_question("How fast does heavy rain intensify?", chunks)

    assert answer.citations == ["a.pdf#p1#0"]
    assert "a.pdf#p1#0" in seen["prompt"]  # chunk ids are offered to the model
    assert "How fast does heavy rain intensify?" in seen["prompt"]


def test_answer_question_seam_error_is_typed(monkeypatch):
    def broken(prompt, *, schema):
        raise GeminiError("backend down")

    monkeypatch.setattr(answer_mod, "generate_json", broken)
    with pytest.raises(AnswerError, match="backend down"):
        answer_question("q", [_chunk(ALLOWED[0])])


def test_scope_guard_refuses_without_any_model_call(monkeypatch):
    monkeypatch.setattr(
        answer_mod, "generate_json",
        lambda *a, **k: pytest.fail("LLM must not be called for out-of-scope questions"),
    )
    result = answer_with_guard("Are tropical cyclones intensifying?", [_chunk(ALLOWED[0])])
    assert result.abstain is True
    assert "tropical cyclone" in result.abstain_reason


# --- caching: answers are pinned, abstentions are not ---

_CACHED_Q = "How fast does heavy rain intensify over South Asia?"


def _counting_model(monkeypatch, raw: str) -> list[str]:
    calls: list[str] = []

    def fake_generate(prompt, *, schema):
        calls.append(prompt)
        return raw

    monkeypatch.setattr(answer_mod, "generate_json", fake_generate)
    return calls


def test_an_answer_is_cached_so_a_repeat_costs_no_model_call(monkeypatch, tmp_path):
    from rag.answer_cache import AnswerCache

    calls = _counting_model(monkeypatch, _llm_json())
    cache = AnswerCache(tmp_path)
    chunks = [_chunk(cid) for cid in ALLOWED]

    first = answer_with_guard(_CACHED_Q, chunks, cache=cache)
    second = answer_with_guard(_CACHED_Q, chunks, cache=cache)

    assert first == second and first.abstain is False
    assert len(calls) == 1


def test_an_abstention_is_not_cached_so_the_next_ask_reaches_the_model(
    monkeypatch, tmp_path
):
    """Temperature 0 is not deterministic: one abstention is one sample, and
    caching it would replay that refusal to every identical query."""
    from rag.answer_cache import AnswerCache

    calls = _counting_model(
        monkeypatch, _llm_json(answer="", citations=[], abstain=True,
                               abstain_reason="no region-level excerpt"),
    )
    cache = AnswerCache(tmp_path)
    chunks = [_chunk(cid) for cid in ALLOWED]

    answer_with_guard(_CACHED_Q, chunks, cache=cache)
    answer_with_guard(_CACHED_Q, chunks, cache=cache)

    assert len(calls) == 2
    assert cache.get(cache.key(_CACHED_Q, chunks)) is None  # nothing was written


def test_an_abstention_cached_before_the_rule_is_read_as_a_miss(monkeypatch, tmp_path):
    from rag.answer_cache import AnswerCache

    calls = _counting_model(monkeypatch, _llm_json())
    cache = AnswerCache(tmp_path)
    chunks = [_chunk(cid) for cid in ALLOWED]
    key = cache.key(_CACHED_Q, chunks)
    cache.put(key, CitedAnswer(answer="", citations=[], abstain=True, allowed_ids=ALLOWED))

    result = answer_with_guard(_CACHED_Q, chunks, cache=cache)

    assert result.abstain is False and len(calls) == 1
    assert cache.get(key) == result  # the fresh answer replaced the stale refusal
