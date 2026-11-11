"""Tests for rag/jev_verify.py: per-citation evidence strength from Jev.

One request per (sentence, passage) pair, each answered by a pytest-httpx
callback that reads the pair back out of the request, so the parallel fan-out
cannot pair an answer with the wrong request.
"""
from __future__ import annotations

import json

import httpx
import pytest

import rag.jev_client as jc
import rag.jev_verify as ver
from rag.chunk import Chunk
from tests.jev_fakes import FAKE_KEY, enable_jev, relation_reply

P124 = Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#2", source="IPCC_AR6_WGI_Chapter11.pdf",
             page=124, text="East Central Asia (ECA): median increase of more than 3.5°C.")
P124B = Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#1", source="IPCC_AR6_WGI_Chapter11.pdf",
              page=124, text="ECA: significant increases in hot extremes since 1950.")
P16 = Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p16#0", source="IPCC_AR6_WGI_SPM.pdf",
            page=16, text="Glaciers are projected to continue to lose mass.")


def _judge(rules):
    """A callback: `rules(claim, passage) -> (supports, contradicts)` per request."""
    def handle(request: httpx.Request) -> httpx.Response:
        state = json.loads(request.content)["state"]
        supports, contradicts = rules(state["claim"], state["passage"])
        return httpx.Response(200, json=relation_reply(supports, contradicts))
    return handle


# --- splitting and labels -----------------------------------------------------------

def test_statements_split_on_sentences_not_on_decimals_or_figure_numbers():
    text = ("Hot extremes rise by 1.5 °C per decade in ECA. See Fig. 11.3 for detail. "
            "Heavy rain also intensifies (high confidence).")
    assert ver.split_statements(text) == [
        "Hot extremes rise by 1.5 °C per decade in ECA.",
        "See Fig. 11.3 for detail.",
        "Heavy rain also intensifies (high confidence).",
    ]


def test_statements_are_capped_in_count_and_length():
    many = " ".join(f"Sentence number {i} is long enough." for i in range(10))
    assert len(ver.split_statements(many)) == ver.MAX_STATEMENTS
    assert len(ver.split_statements("A" * 2000)[0]) == ver.STATEMENT_CHARS


@pytest.mark.parametrize(("supports", "contradicts", "label"), [
    (0.95, 0.0, "strong"), (0.8, 0.0, "strong"), (0.65, 0.1, "moderate"),
    (0.2, 0.1, "weak"), (0.05, 0.9, "contradicted"), (0.85, 0.85, "strong"),
])
def test_strength_labels_follow_the_fixed_thresholds(supports, contradicts, label):
    assert ver.strength(supports, contradicts) == label


# --- the check -----------------------------------------------------------------------

def test_one_request_per_sentence_and_passage_with_the_cookbook_question(httpx_mock):
    httpx_mock.add_callback(_judge(lambda c, p: (0.9, 0.0)), is_reusable=True)
    client = jc.JevClient(FAKE_KEY)
    try:
        ver.verify_answer("First claim about heat. Second claim about rain.", [P124, P16],
                          client=client)
    finally:
        client.close()

    requests = httpx_mock.get_requests()
    assert len(requests) == 4  # 2 sentences x 2 passages
    body = json.loads(requests[0].content)
    assert set(body["state"]) == {"claim", "passage"}
    assert body["questions"] == ver.RELATION_QUESTION
    assert list(body["questions"]["relation"]["criteria"]) == [
        "supports", "contradicts", "says_nothing"]


def test_each_cited_page_gets_its_strongest_reading(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(
        _judge(lambda c, p: (0.95, 0.0) if "3.5" in p else (0.04, 0.0)), is_reusable=True)

    check = ver.verify_answer(
        "Hot extremes in ECA are projected to intensify by more than 3.5°C.", [P124, P124B, P16])

    assert check.status == "ok" and check.model == "jev-1.14.0"
    assert [(e.locator, e.strength) for e in check.citations] == [("p124", "strong"),
                                                                  ("p16", "weak")]
    best = next(e for e in check.citations if e.locator == "p124")
    assert best.chunk_id == P124.chunk_id  # the chunk that carried the support
    assert check.unsupported_statements == []


def test_a_sentence_no_page_supports_is_listed(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(
        _judge(lambda c, p: (0.9, 0.0) if c.startswith("Heat") else (0.1, 0.0)), is_reusable=True)

    check = ver.verify_answer("Heat extremes increase in ECA. Monsoon winds weaken sharply.",
                              [P124])

    assert check.unsupported_statements == ["Monsoon winds weaken sharply."]


def test_a_contradicting_page_is_flagged(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(_judge(lambda c, p: (0.02, 0.94)), is_reusable=True)

    check = ver.verify_answer("Glaciers are projected to grow.", [P16])

    assert check.citations[0].strength == "contradicted"
    assert check.citations[0].contradicts == pytest.approx(0.94)


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_some_failures_make_the_check_partial_not_broken(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    monkeypatch.setenv("CRG_JEV_VERIFY_BUDGET_S", "0.05")

    def handle(request: httpx.Request) -> httpx.Response:
        passage = json.loads(request.content)["state"]["passage"]
        if "Glaciers" in passage:
            return httpx.Response(529)
        return httpx.Response(200, json=relation_reply(0.9))

    httpx_mock.add_callback(handle, is_reusable=True)

    check = ver.verify_answer("Hot extremes intensify in ECA.", [P124, P16])

    assert check.status == "partial"
    assert {e.locator: e.strength for e in check.citations} == {"p124": "strong",
                                                                 "p16": "unchecked"}


def test_no_key_is_an_unavailable_check_not_an_exception(monkeypatch):
    monkeypatch.setenv("CRG_JEV", "verify")
    check = ver.verify_answer("Hot extremes intensify.", [P124])
    assert check.status == "unavailable" and "TYPESAFE_API_KEY" in check.detail


def test_nothing_to_check_sends_nothing(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    assert ver.verify_answer("", [P124]).status == "unavailable"
    assert ver.verify_answer("A long enough sentence here.", []).status == "unavailable"
    assert httpx_mock.get_requests() == []


def test_long_passages_are_cut_to_the_budget(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(_judge(lambda c, p: (0.9, 0.0)), is_reusable=True)
    huge = Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p1#0", source="IPCC_AR6_WGI_SPM.pdf",
                 page=1, text="word " * 5000)

    ver.verify_answer("A sentence that is long enough.", [huge])

    sent = json.loads(httpx_mock.get_requests()[0].content)["state"]["passage"]
    assert len(sent) == ver.PASSAGE_CHARS


def _pages(n: int) -> list[Chunk]:
    return [Chunk(chunk_id=f"IPCC_AR6_WGI_SPM.pdf#p{p}#0", source="IPCC_AR6_WGI_SPM.pdf",
                  page=p, text=f"Passage on page {p}.") for p in range(1, n + 1)]


def test_all_eight_retrieved_pages_are_checked(httpx_mock, monkeypatch):
    """Retrieval returns 8 chunks (agent/graph.py), so the check must cover 8."""
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(
        _judge(lambda c, p: (0.9, 0.0) if "page 8." in p else (0.05, 0.0)), is_reusable=True)

    check = ver.verify_answer("Only the last page backs this sentence.", _pages(8))

    assert len(check.citations) == 8 and check.status == "ok"
    assert check.citations[-1].strength == "strong"
    assert check.unsupported_statements == []


def test_pages_past_the_cap_are_unchecked_and_the_check_is_partial(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(_judge(lambda c, p: (0.05, 0.0)), is_reusable=True)

    check = ver.verify_answer("A sentence no checked page supports.", _pages(10))

    assert len(check.citations) == 10 and check.status == "partial"
    assert [e.strength for e in check.citations[-2:]] == ["unchecked", "unchecked"]
    # a dropped page might have supported it, so it is not called unsupported
    assert check.unsupported_statements == []


def test_sentences_past_the_cap_make_the_check_partial(httpx_mock, monkeypatch):
    enable_jev(monkeypatch, "verify")
    httpx_mock.add_callback(_judge(lambda c, p: (0.9, 0.0)), is_reusable=True)
    many = " ".join(f"Sentence number {i} is long enough." for i in range(6))

    check = ver.verify_answer(many, [P124])

    assert check.status == "partial" and "4 of 6 sentences" in check.detail


@pytest.mark.httpx_mock(assert_all_responses_were_requested=False)
def test_the_whole_check_stops_at_its_deadline(httpx_mock, monkeypatch):
    import time as _time

    enable_jev(monkeypatch, "verify")
    monkeypatch.setenv("CRG_JEV_VERIFY_BUDGET_S", "0.05")
    monkeypatch.setattr(ver, "DEADLINE_GRACE_S", 0.0)

    def slow(request: httpx.Request) -> httpx.Response:
        _time.sleep(0.4)  # longer than the whole check may take
        return httpx.Response(200, json=relation_reply(0.9))

    httpx_mock.add_callback(slow, is_reusable=True)
    started = _time.perf_counter()

    check = ver.verify_answer("A sentence that is long enough.", [P124, P16])

    assert _time.perf_counter() - started < 0.35
    assert check.status == "unavailable" and "time budget" in check.detail
