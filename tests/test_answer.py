"""Tests for the cited LLM answerer (rag/answer.py). HTTP mocked; validators pure.

The trust core: a citation that doesn't reference a retrieved chunk_id cannot
even be constructed — fabricated citations are a schema violation, not a vibe.
"""
import json

import pytest
from pydantic import ValidationError

from rag.answer import AnswerError, CitedAnswer, answer_question
from rag.chunk import Chunk

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


def _gemini_response(text: str) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": text}]}}]}


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


# --- the network edge (mocked) ---


def test_answer_question_round_trip(httpx_mock):
    httpx_mock.add_response(json=_gemini_response(_llm_json()))
    chunks = [_chunk(cid) for cid in ALLOWED]

    answer = answer_question("How fast does heavy rain intensify?", chunks, api_key="k")

    assert answer.citations == ["a.pdf#p1#0"]
    sent = json.loads(httpx_mock.get_requests()[0].content)
    prompt = json.dumps(sent)
    assert "a.pdf#p1#0" in prompt  # chunk ids are offered to the model
    assert "How fast does heavy rain intensify?" in prompt


def test_answer_question_http_error_is_typed(httpx_mock):
    httpx_mock.add_response(status_code=500)
    with pytest.raises(AnswerError):
        answer_question("q", [_chunk(ALLOWED[0])], api_key="k")


def test_answer_question_retries_429_with_server_delay(httpx_mock, monkeypatch):
    slept = []
    monkeypatch.setattr("rag.answer._sleep", slept.append)
    httpx_mock.add_response(status_code=429, text="Please retry in 11s.")
    httpx_mock.add_response(json=_gemini_response(_llm_json()))
    answer = answer_question("q", [_chunk(ALLOWED[0])], api_key="k")
    assert answer.abstain is False
    assert slept == [12.0]


def test_scope_guard_refuses_without_any_network_call(httpx_mock):
    # no responses registered: any HTTP attempt would fail this test loudly
    from rag.answer import answer_with_guard

    result = answer_with_guard(
        "Are tropical cyclones intensifying?", [_chunk(ALLOWED[0])], api_key="k"
    )
    assert result.abstain is True
    assert "tropical cyclone" in result.abstain_reason
    assert httpx_mock.get_requests() == []
