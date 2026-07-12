"""Tests for the claim-level judge (evals/claim_judge.py) — seam-mocked, offline.

Evaluator gap #5: page-level citation checks can't catch a correctly-cited page
being misquoted. The judge decomposes an answer into factual claims and tests
each against ONLY the cited excerpt texts.
"""
import pytest

import evals.claim_judge as judge_mod
from evals.claim_judge import ClaimJudgment, judge_claims


def _fake_seam(monkeypatch, payload):
    captured = {}

    def fake_generate_json(prompt, *, schema):
        captured["prompt"] = prompt
        return payload

    monkeypatch.setattr(judge_mod, "generate_json", fake_generate_json)
    return captured


def test_judgment_parses_and_scores(monkeypatch):
    _fake_seam(monkeypatch, {
        "claims": [
            {"claim": "heavy precipitation intensifies ~7%/°C", "supported": True},
            {"claim": "this holds for all regions", "supported": False},
        ]
    })
    result = judge_claims("answer text", ["excerpt one", "excerpt two"])

    assert isinstance(result, ClaimJudgment)
    assert result.n_claims == 2
    assert result.n_supported == 1
    assert result.all_supported is False


def test_prompt_carries_excerpts_and_answer_as_data(monkeypatch):
    captured = _fake_seam(monkeypatch, {"claims": []})
    judge_claims("THE ANSWER", ["EXCERPT A", "EXCERPT B"])

    assert "THE ANSWER" in captured["prompt"]
    assert "EXCERPT A" in captured["prompt"] and "EXCERPT B" in captured["prompt"]


def test_no_claims_counts_as_vacuously_supported(monkeypatch):
    _fake_seam(monkeypatch, {"claims": []})
    result = judge_claims("…", ["e"])
    assert result.all_supported is True and result.n_claims == 0
