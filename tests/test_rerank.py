"""Tests for the optional reranker + query-rewrite stages.

Both stages call Gemini, so every test here mocks the SINGLE seam
(`generate_json`, imported into rag.rerank / rag.rewrite) rather than HTTP —
same convention as tests/test_answer.py. The cross-encoder is mocked at the
fastembed class, so the suite never downloads an ONNX model.
"""
import json
import sys

import numpy as np
import pytest

import rag.rerank as rerank_mod
import rag.rewrite as rewrite_mod
from evals.run_retrieval_eval import arm_label, parse_args
from rag.chunk import Chunk
from rag.rerank import (
    CrossEncoderReranker,
    GeminiListwiseReranker,
    NoReranker,
    _apply_order,
    build_reranker,
)
from rag.retrieve import HybridRetriever
from rag.rewrite import neutral_query, parse_query


def _chunk(i: int, text: str) -> Chunk:
    return Chunk(chunk_id=f"d.pdf#p{i+1}#0", source="d.pdf", page=i + 1, text=text)


CHUNKS = [
    _chunk(0, "heavy precipitation intensifies with warming rx1day"),
    _chunk(1, "glaciers melting committed centuries"),
    _chunk(2, "surface wind stilling tropics"),
]


# ---------------------------------------------------------------- protocol


def test_no_reranker_is_the_identity_up_to_top_k():
    assert NoReranker().rerank("q", CHUNKS, top_k=2) == CHUNKS[:2]


def test_build_reranker_none_means_none():
    assert build_reranker("none") is None
    assert isinstance(build_reranker("gemini"), GeminiListwiseReranker)
    with pytest.raises(ValueError):
        build_reranker("nope")


def test_apply_order_survives_duplicate_missing_and_out_of_range_indices():
    # model says: 3, 3 again, an index that does not exist, then 1. Candidate 2
    # is never mentioned -> it must still be present, after the ranked ones.
    ranked = _apply_order([3, 3, 99, 1], CHUNKS, top_k=3)
    assert [c.chunk_id for c in ranked] == [
        CHUNKS[2].chunk_id, CHUNKS[0].chunk_id, CHUNKS[1].chunk_id
    ]


# ---------------------------------------------------------------- arm A


def test_gemini_reranker_applies_the_returned_order(monkeypatch):
    seen = {}

    def fake_generate_json(prompt, *, schema):
        seen["prompt"] = prompt
        seen["schema"] = schema
        return json.dumps({"ranking": [2, 1, 3]})

    monkeypatch.setattr(rerank_mod, "generate_json", fake_generate_json)
    ranked = GeminiListwiseReranker().rerank("why do glaciers melt", CHUNKS, top_k=2)

    assert [c.chunk_id for c in ranked] == [CHUNKS[1].chunk_id, CHUNKS[0].chunk_id]
    assert 'number="1"' in seen["prompt"] and 'number="3"' in seen["prompt"]
    assert seen["schema"]["properties"]["ranking"]["items"]["type"] == "INTEGER"


def test_gemini_reranker_caps_excerpt_length(monkeypatch):
    long_chunk = _chunk(0, "x" * 5000)
    monkeypatch.setattr(rerank_mod, "generate_json",
                        lambda prompt, *, schema: json.dumps({"ranking": [1]}))
    reranker = GeminiListwiseReranker(excerpt_chars=50)
    prompt = reranker._prompt("q", [long_chunk])
    assert "x" * 50 in prompt and "x" * 51 not in prompt


def test_gemini_reranker_falls_back_loudly_when_the_call_fails(monkeypatch, caplog):
    def boom(prompt, *, schema):
        raise rerank_mod.GeminiError("quota")

    monkeypatch.setattr(rerank_mod, "generate_json", boom)
    ranked = GeminiListwiseReranker().rerank("q", CHUNKS, top_k=3)

    assert ranked == CHUNKS  # fused order preserved, nothing dropped
    assert "keeping the fused RRF order" in caplog.text


def test_gemini_reranker_falls_back_on_malformed_json(monkeypatch, caplog):
    monkeypatch.setattr(rerank_mod, "generate_json", lambda prompt, *, schema: "not json")
    assert GeminiListwiseReranker().rerank("q", CHUNKS, top_k=3) == CHUNKS
    assert "keeping the fused RRF order" in caplog.text


# ---------------------------------------------------------------- arm B


class _FakeCrossEncoder:
    """Stands in for fastembed's TextCrossEncoder: scores, higher is better."""

    def __init__(self, model_name):
        self.model_name = model_name

    def rerank(self, query, documents):
        # score = position of the word "wind", so chunk 3 wins
        return [10.0 if "wind" in d else -1.0 for d in documents]


def test_cross_encoder_reranker_sorts_by_score(monkeypatch):
    # fastembed lives in the optional 'eval' group: skip rather than fail when a
    # runtime-only install (the deployed image) runs the suite.
    module = pytest.importorskip("fastembed.rerank.cross_encoder")
    monkeypatch.setattr(module, "TextCrossEncoder", _FakeCrossEncoder)

    reranker = CrossEncoderReranker()
    ranked = reranker.rerank("stilling", CHUNKS, top_k=2)

    assert ranked[0].chunk_id == CHUNKS[2].chunk_id
    assert reranker.model_name == "Xenova/ms-marco-MiniLM-L-6-v2"


def test_cross_encoder_ties_keep_the_fused_order(monkeypatch):
    class _Flat(_FakeCrossEncoder):
        def rerank(self, query, documents):
            return [1.0] * len(documents)

    module = pytest.importorskip("fastembed.rerank.cross_encoder")
    monkeypatch.setattr(module, "TextCrossEncoder", _Flat)
    assert CrossEncoderReranker().rerank("q", CHUNKS, top_k=3) == CHUNKS


# ---------------------------------------------------------------- wiring


def test_retriever_without_a_reranker_is_unchanged():
    retriever = HybridRetriever(CHUNKS, doc_matrix=None)
    assert retriever.retrieve("glaciers melting", top_k=1)[0].chunk_id == CHUNKS[1].chunk_id


def test_retriever_hands_the_reranker_a_wider_pool_than_top_k(monkeypatch):
    seen = {}

    class _Spy:
        def rerank(self, question, candidates, top_k):
            seen["n_candidates"] = len(candidates)
            seen["question"] = question
            return list(reversed(candidates))[:top_k]

    matrix = np.eye(3)
    retriever = HybridRetriever(CHUNKS, doc_matrix=matrix, reranker=_Spy())
    monkeypatch.setattr(retriever, "_embed_query", lambda q: [1.0, 0.0, 0.0])

    top = retriever.retrieve("precipitation glaciers wind", top_k=1)

    assert seen["n_candidates"] == 3  # everything the corpus has, > top_k=1
    assert len(top) == 1
    assert seen["question"] == "precipitation glaciers wind"


def test_rewriter_changes_the_search_query_not_the_callers_question(monkeypatch):
    seen = {}

    class _Spy:
        def rerank(self, question, candidates, top_k):
            seen["question"] = question
            return candidates[:top_k]

    retriever = HybridRetriever(
        CHUNKS, doc_matrix=None, reranker=_Spy(),
        rewriter=lambda q: "glaciers melting",
    )
    top = retriever.retrieve("given that glaciers are growing, why?", top_k=1)

    assert top[0].chunk_id == CHUNKS[1].chunk_id  # searched with the NEUTRAL query
    assert seen["question"] == "glaciers melting"  # reranker sees it too


# ---------------------------------------------------------------- arm C


@pytest.mark.parametrize(
    "raw",
    ["not json", '["a list"]', '{"query": ""}', '{"query": 7}', '{"other": "x"}', None],
)
def test_parse_query_falls_back_to_the_original_on_any_bad_output(raw):
    assert parse_query(raw, fallback="original question") == "original question"


def test_parse_query_takes_the_rewritten_query():
    assert parse_query('{"query": "  Rx1day South Asia  "}', fallback="orig") == "Rx1day South Asia"


def test_neutral_query_calls_the_seam_and_strips_the_premise(monkeypatch):
    seen = {}

    def fake_generate_json(prompt, *, schema):
        seen["prompt"] = prompt
        return json.dumps({"query": "Rx1day South Asia projection"})

    monkeypatch.setattr(rewrite_mod, "generate_json", fake_generate_json)
    out = neutral_query("Given that Rx1day is falling over South Asia, by how much?")

    assert out == "Rx1day South Asia projection"
    assert "<question>" in seen["prompt"]


def test_neutral_query_returns_the_original_when_gemini_fails(monkeypatch, caplog):
    def boom(prompt, *, schema):
        raise rewrite_mod.GeminiError("no auth")

    monkeypatch.setattr(rewrite_mod, "generate_json", boom)
    assert neutral_query("original") == "original"
    assert "retrieving with the original question" in caplog.text


# ---------------------------------------------------------------- CLI flags


def test_cli_defaults_are_todays_behaviour(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_retrieval_eval"])
    args = parse_args()
    assert (args.reranker, args.rewrite) == ("none", "off")
    assert arm_label(args.reranker, args.rewrite) == ""  # no extra column


@pytest.mark.parametrize(
    ("reranker", "rewrite", "expected"),
    [
        ("none", "off", ""),
        ("gemini", "off", "hybrid+gemini"),
        ("minilm", "off", "hybrid+minilm"),
        ("none", "on", "hybrid+rewrite"),
        ("minilm", "on", "hybrid+minilm+rewrite"),
    ],
)
def test_arm_label_names_the_extra_column(reranker, rewrite, expected):
    assert arm_label(reranker, rewrite) == expected


def test_cli_accepts_the_new_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["run_retrieval_eval", "--reranker", "minilm", "--rewrite", "on"])
    args = parse_args()
    assert (args.reranker, args.rewrite) == ("minilm", "on")


def test_cli_rejects_an_unknown_reranker(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_retrieval_eval", "--reranker", "bm42"])
    with pytest.raises(SystemExit):
        parse_args()
