"""Tests for evals/run_jev_eval.py: the Jev eval reads the dev set only, scores
against labels the dev set already carries, and runs offline in a dry run."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import evals.run_jev_eval as ev
from evals.gold_set import load_gold_set

TREE = ast.parse(Path(ev.__file__).read_text(encoding="utf-8"))


def test_the_held_out_set_is_unreachable_from_this_script():
    """No name, import or string in the code (the docstring aside) reaches it."""
    names = {node.id for node in ast.walk(TREE) if isinstance(node, ast.Name)}
    names |= {node.attr for node in ast.walk(TREE) if isinstance(node, ast.Attribute)}
    names |= {alias.name for node in ast.walk(TREE) if isinstance(node, ast.ImportFrom)
              for alias in node.names}
    assert not names & {"load_test_set", "TEST_SET_PATH", "TEST_SET_FROZEN_HASH_PATH"}
    docstring = ast.get_docstring(TREE)
    strings = [node.value for node in ast.walk(TREE)
               if isinstance(node, ast.Constant) and isinstance(node.value, str)
               and node.value.strip() != (docstring or "").strip()]
    assert not any("gold_set_v2" in s or "EVAL_SET" in s for s in strings)


def test_labels_come_from_the_dev_set_slices():
    rows = ev.dev_rows()
    assert len(rows) == 60
    refuse = sorted(r["id"] for r in rows if r["should_refuse"])
    assert refuse == ["OOC-01", "OOC-02", "OOC-03", "OSH-01", "OSH-02", "OSH-03", "OSH-04"]
    # premise items must reach the answering model for a grounded refusal
    assert not any(r["should_refuse"] for r in rows if r["slice"] == "premise_injection")
    assert sum(1 for r in rows if r["expected_hazard"]) == 42


def test_the_probes_are_the_stage_two_paraphrases():
    rows = ev.probe_rows()
    assert len(rows) == 14
    assert sum(r["should_refuse"] for r in rows) == 7
    assert {r["expected_hazard"] for r in rows if not r["should_refuse"]} == set(ev.OUR_HAZARDS)


def test_todays_baseline_is_measured_offline():
    rows = ev.dev_rows()
    for row in rows:
        ev.lexical_baseline(row)
    scores = ev.scope_scores(rows, "stage1_refused")
    assert scores["false_refusals"] == []
    assert scores["false_accepts"] == ["OOC-01", "OOC-03"]  # refused only by the LLM today
    assert sum(r["silent_bucket"] for r in rows) == 16


def test_scores_count_both_error_kinds():
    rows = [
        {"id": "a", "should_refuse": True, "refused": True},
        {"id": "b", "should_refuse": True, "refused": False},
        {"id": "c", "should_refuse": False, "refused": True},
        {"id": "d", "should_refuse": False, "refused": False},
    ]
    scores = ev.scope_scores(rows, "refused")
    assert (scores["correct"], scores["false_accepts"], scores["false_refusals"]) == (2, ["b"], ["c"])


def test_hazard_scores_separate_missed_from_wrong():
    rows = [{"id": "a", "expected_hazard": "wind", "got": "wind"},
            {"id": "b", "expected_hazard": "wind", "got": None},
            {"id": "c", "expected_hazard": "heatwave", "got": "wind"},
            {"id": "d", "expected_hazard": None, "got": "wind"}]
    scores = ev.hazard_scores(rows, "got")
    assert (scores["n"], scores["right"], scores["missed"], scores["wrong"]) == (3, 1, ["b"], ["c"])


def test_percentile_and_auc():
    assert ev.percentile([], 0.5) is None
    assert ev.percentile([100.0, 200.0, 300.0, 400.0], 0.95) == 400.0
    assert ev.auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert ev.auc([0.5], [0.5]) == 0.5


def test_every_premise_and_cid_item_parses_into_a_claim():
    questions = load_gold_set().questions
    premises = [q for q in questions if q.slice.value == "premise_injection"]
    cids = [q for q in questions if q.slice.value == "cid-table"]
    assert all(ev._PREMISE.match(q.question) for q in premises)
    assert all(ev._CID_QUESTION.match(q.question) for q in cids)
    assert all(ev._CID_NOTE.search(q.notes) and ev._CID_CHUNK.search(q.notes) for q in cids)


def test_the_corrected_cid_labels_build_the_claims_their_chunks_state():
    """CID-06's sentence states river flooding, and CID-11's ESAF clause states cyclone
    landfalls, not mean wind speed (evidence in their notes in evals/gold_set.json)."""
    from rag.chunk import Chunk

    chunks = []
    for q in load_gold_set().questions:
        if q.slice.value == "cid-table":
            chunk_id = ev._CID_CHUNK.search(q.notes).group("chunk")
            source, page, _ = chunk_id.split("#")
            chunks.append(Chunk(chunk_id=chunk_id, source=source, page=int(page[1:]),
                                text=q.supporting_quote))

    pairs, skipped = ev.verify_pairs(chunks)

    claims = {(p["id"], p["kind"]): p["claim"] for p in pairs if p["kind"].startswith("cid")}
    assert claims[("CID-06", "cid_true")] == (
        "River flooding is projected to increase in New Zealand (NZ).")
    assert claims[("CID-06", "cid_flipped")] == (
        "River flooding is projected to decrease in New Zealand (NZ).")
    assert ("CID-11", "cid_true") not in claims and ("CID-11", "cid_flipped") not in claims
    assert skipped["cid_no_statement"] == 1
    assert claims[("CID-04", "cid_true")] == (
        "Heavy precipitation is projected to increase in Northern Australia (NAU).")
    assert sum(kind == "cid_true" for _, kind in claims) == 14


def test_a_dry_run_writes_both_artifacts_without_any_request(tmp_path, monkeypatch, httpx_mock):
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRG_SCOPE_STAGE2", "off")
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)

    assert ev.main(["--dry-run"]) == 0

    [json_path] = tmp_path.glob("jev-dev-*.json")
    [md_path] = tmp_path.glob("jev-dev-*.md")
    report = md_path.read_text(encoding="utf-8")
    assert "not run (dry run)" in report and "held-out set was not read" in report
    assert "58/60" in report  # today's stage 1 on the dev set
    assert httpx_mock.get_requests() == []
    assert "\u2014" not in report and "\u2013" not in report  # house style: no em or en dashes


def _fake_jev(request):
    """A stand-in TypeSafe: models list, and triage readings keyed on the words."""
    import json as _json

    import httpx

    from tests.jev_fakes import triage_reply

    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"models": [
            {"name": "jev-latest", "description": "x", "release_date": "2026-10-14"}]})
    question = _json.loads(request.content)["state"]["question"].lower()
    if any(word in question for word in ("gdp", "nationally determined", "promised", "smoke")):
        reply = triage_reply(topic={"policy_or_economics": 0.95, "climate_science": 0.05},
                             about=0.03, hazard={"heatwave": 1.0}, other={"policy": 1.0})
    elif any(word in question for word in ("hot", "heat", "scorcher")):
        reply = triage_reply(topic={"assessed_hazard": 0.97, "climate_science": 0.03},
                             about=0.96, hazard={"heatwave": 0.94, "wind": 0.06})
    else:
        reply = triage_reply(topic={"climate_science": 0.9, "assessed_hazard": 0.1},
                             about=0.2, hazard={"wind": 0.5, "heatwave": 0.5})
    return httpx.Response(200, json=reply)


def test_the_live_path_scores_every_arm_against_a_fake_jev(tmp_path, monkeypatch, httpx_mock):
    from tests.jev_fakes import FAKE_KEY

    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRG_SCOPE_STAGE2", "off")
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)
    httpx_mock.add_callback(_fake_jev, is_reusable=True)

    assert ev.main(["--order-check", "--repeats", "2", "--max-rpm", "0"]) == 0

    import json as _json

    artifact = _json.loads(next(tmp_path.glob("jev-dev-*.json")).read_text(encoding="utf-8"))
    assert artifact["jev"]["status"] == "ok" and artifact["jev"]["live_calls"] > 0
    assert artifact["jev"]["models_listed"] == ["jev-latest"]
    assert artifact["jev"]["spend_usd"] > 0 and artifact["jev"]["latency_ms_p50"] is not None
    dev = artifact["scope"]["dev"]
    # the four lexical refusals, OOC-02 (lexical, "sea-level"), and the two the
    # fake reads as policy: OOC-01 (NDC) and OOC-03 (GDP), now refused at once
    assert set(dev["shipped"]["correct_refusals"]) == {
        "OSH-01", "OSH-02", "OSH-03", "OSH-04", "OOC-01", "OOC-02", "OOC-03"}
    assert dev["shipped"]["false_refusals"] == []
    assert artifact["hazard"]["dev"]["front_door"]["n"] == 42
    # robustness runs cover every silent-bucket row: 16 dev items and all 14 probes
    assert artifact["order_check"]["n"] == 30 and artifact["repeats"]["runs"] == 2
    report = next(tmp_path.glob("jev-dev-*.md")).read_text(encoding="utf-8")
    assert "| OOC-03 | out_of_corpus | True | refuse |" in report
    assert FAKE_KEY not in report and FAKE_KEY not in _json.dumps(artifact)


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_a_nonsense_budget_override_keeps_the_default(monkeypatch, value):
    from rag.jev_client import triage_policy

    monkeypatch.setenv("CRG_JEV_TRIAGE_BUDGET_S", value)
    assert triage_policy().budget_s == 2.5


def _clean_env(monkeypatch, tmp_path):
    from tests.jev_fakes import FAKE_KEY

    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CRG_SCOPE_STAGE2", "off")
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)


def test_a_verify_only_run_scores_verify_and_leaves_triage_unmeasured(
    tmp_path, monkeypatch, httpx_mock
):
    """Review finding: --task verify used to crash after spending, before writing."""
    import json as _json

    import httpx

    from rag.chunk import Chunk
    from tests.jev_fakes import relation_reply

    _clean_env(monkeypatch, tmp_path)
    chunk = Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p9#0", source="IPCC_AR6_WGI_SPM.pdf",
                  page=9, text="Hot extremes have become more frequent.")
    pairs = [{"id": "X-1", "kind": "quote", "expected": "supports",
              "claim": "Hot extremes have become more frequent.", "chunk": chunk},
             {"id": "X-1>Y-1", "kind": "cross", "expected": "says_nothing",
              "claim": "Glaciers are shrinking.", "chunk": chunk}]
    monkeypatch.setattr(ev, "verify_pairs", lambda chunks: (pairs, {}))
    monkeypatch.setattr("rag.corpus.load_corpus_chunks", lambda: ())

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": []})
        claim = _json.loads(request.content)["state"]["claim"]
        return httpx.Response(200, json=relation_reply(0.95 if "Hot" in claim else 0.02))

    httpx_mock.add_callback(handle, is_reusable=True)

    assert ev.main(["--task", "verify", "--max-rpm", "0"]) == 0

    artifact = _json.loads(next(tmp_path.glob("jev-dev-*.json")).read_text(encoding="utf-8"))
    assert artifact["jev"]["verify_measured"] and not artifact["jev"]["triage_measured"]
    assert artifact["scope"]["dev"]["shipped"]["accuracy"] is None
    assert artifact["verify"]["auc_supports_vs_cross"] == 1.0


def test_rows_jev_could_not_answer_are_left_out_and_listed(tmp_path, monkeypatch, httpx_mock):
    """Review finding: an unavailable reading used to be scored as a decision."""
    import json as _json

    import httpx

    _clean_env(monkeypatch, tmp_path)
    monkeypatch.setenv("CRG_JEV_TRIAGE_BUDGET_S", "0.05")
    failed: list[int] = []

    def handle(request):
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"models": []})
        # Fail only the first ask about GDP, the guard's. Three failures in a
        # row would (correctly) open the circuit and blank out every later row.
        if "gdp" in _json.loads(request.content)["state"]["question"].lower() and not failed:
            failed.append(1)
            return httpx.Response(529)
        return _fake_jev(request)

    httpx_mock.add_callback(handle, is_reusable=True)

    assert ev.main(["--max-rpm", "0"]) == 0

    artifact = _json.loads(next(tmp_path.glob("jev-dev-*.json")).read_text(encoding="utf-8"))
    shipped = artifact["scope"]["dev"]["shipped"]
    assert shipped["excluded_unavailable"] == ["OOC-03"]  # the GDP question
    assert "OOC-03" not in shipped["false_accepts"] and shipped["n"] == 59
    report = next(tmp_path.glob("jev-dev-*.md")).read_text(encoding="utf-8")
    assert "left out because Jev was unavailable for them: OOC-03" in report


def test_the_e2e_runner_refuses_a_held_out_run_with_jev_on(monkeypatch):
    """Review finding: held-out questions must never reach TypeSafe."""
    import sys

    import evals.run_e2e_eval as e2e

    monkeypatch.setattr(sys, "argv", ["run_e2e_eval"])
    monkeypatch.setattr(e2e, "_EVAL_SET", "test")
    monkeypatch.setenv("CRG_JEV", "triage")
    monkeypatch.setattr(e2e, "load_test_set", lambda: pytest.fail("held-out set was loaded"))

    with pytest.raises(SystemExit, match="CRG_JEV"):
        e2e.main()


def test_a_run_leaves_no_environment_behind(tmp_path, monkeypatch, httpx_mock):
    """main() paces and widens budgets for itself only; nothing may leak."""
    import os

    for name in ev._SCOPED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(ev, "RESULTS_DIR", tmp_path)

    ev.main(["--dry-run"])

    assert not [name for name in ev._SCOPED_ENV if name in os.environ]
