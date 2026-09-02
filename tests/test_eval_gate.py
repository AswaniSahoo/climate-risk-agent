"""The release gate must fail the build, not just print a number.

Every test here builds tiny artifacts with the same SHAPE the real runners emit
(evals/run_retrieval_eval.py, evals/run_e2e_eval.py), including the two shapes
that are easy to get wrong: `false_answer` is OMITTED when the cell is empty,
and the earliest committed artifacts predate the `eval_set` key entirely.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.eval_gate import (
    GateError,
    arm_of,
    evaluate,
    false_answer_ids,
    headline_r_at_3,
    load_runs,
    main,
    summary_markdown,
)


def retrieval_artifact(r3: float, *, run_utc: str, eval_set: str | None = "dev",
                       retrievers: tuple[str, ...] = ("bm25", "hybrid")) -> dict:
    payload: dict = {"run_utc": run_utc, "retrievers": {}}
    if eval_set is not None:
        payload["eval_set"] = eval_set
    for name in retrievers:
        payload["retrievers"][name] = [
            {"label": "single_page", "n": 4, "recall": {"@3": {"rate": 1.0}}, "mrr": 1.0},
            {
                "label": "HEADLINE (answer)",
                "n": 34,
                "recall": {"@3": {"rate": r3}, "@5": {"rate": r3}, "@10": {"rate": r3}},
                "mrr": 0.8,
            },
        ]
    return payload


def e2e_artifact(*, run_utc: str, false_answers: list[str] | None = None,
                 eval_set: str | None = "dev") -> dict:
    matrix: dict[str, list[str]] = {"correct_answer": ["SP-01"], "correct_refuse": ["OSH-01"]}
    if false_answers:  # the runner omits the key when the cell is empty
        matrix["false_answer"] = false_answers
    payload: dict = {
        "run_utc": run_utc,
        "generate_model": "gemini-2.5-flash",
        "top_k": 8,
        "matrix": matrix,
        "citation_validity": {"passed": 27, "total": 28},
        "numeric_provenance": {"passed": 22, "total": 28},
    }
    if eval_set is not None:
        payload["eval_set"] = eval_set
    return payload


def write(path: Path, payload: dict) -> tuple[Path, dict]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path, payload


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------

def test_false_answer_absent_key_means_zero():
    assert false_answer_ids(e2e_artifact(run_utc="2026-09-01T00:00:00+00:00")) == []


def test_false_answer_ids_returned():
    payload = e2e_artifact(run_utc="2026-09-01T00:00:00+00:00", false_answers=["SP-02", "RT-07"])
    assert false_answer_ids(payload) == ["SP-02", "RT-07"]


def test_missing_matrix_is_malformed_not_zero():
    with pytest.raises(GateError, match="no `matrix`"):
        false_answer_ids({"run_utc": "2026-09-01T00:00:00+00:00"})


def test_headline_r_at_3_reads_the_headline_row_only():
    payload = retrieval_artifact(0.8824, run_utc="2026-09-01T00:00:00+00:00")
    assert headline_r_at_3(payload, "hybrid") == pytest.approx(0.8824)
    assert headline_r_at_3(payload, "dense") is None


# --------------------------------------------------------------------------
# artifact selection
# --------------------------------------------------------------------------

def test_load_runs_is_newest_first_and_split_by_eval_set(tmp_path):
    write(tmp_path / "retrieval-2026-07-16.json",
          retrieval_artifact(0.88, run_utc="2026-07-16T18:27:14+00:00", eval_set=None))
    write(tmp_path / "retrieval-2026-09-01.json",
          retrieval_artifact(0.90, run_utc="2026-09-01T10:00:00+00:00"))
    write(tmp_path / "retrieval-test-2026-07-18.json",
          retrieval_artifact(0.87, run_utc="2026-07-18T06:40:15+00:00", eval_set="test"))

    dev = load_runs(tmp_path, "retrieval", "dev")
    assert [p.name for p, _ in dev] == ["retrieval-2026-09-01.json", "retrieval-2026-07-16.json"]

    test = load_runs(tmp_path, "retrieval", "test")
    assert [p.name for p, _ in test] == ["retrieval-test-2026-07-18.json"]


# --------------------------------------------------------------------------
# the three checks
# --------------------------------------------------------------------------

def test_clean_run_passes(tmp_path):
    result = evaluate(
        retrieval=write(tmp_path / "r-new.json",
                        retrieval_artifact(0.90, run_utc="2026-09-01T10:00:00+00:00")),
        e2e=write(tmp_path / "e-new.json", e2e_artifact(run_utc="2026-09-01T11:00:00+00:00")),
        baseline=write(tmp_path / "r-old.json",
                       retrieval_artifact(0.88, run_utc="2026-07-16T18:00:00+00:00")),
    )
    assert result.ok
    assert result.failures == []


def test_one_false_answer_blocks_release(tmp_path):
    result = evaluate(
        retrieval=write(tmp_path / "r-new.json",
                        retrieval_artifact(0.90, run_utc="2026-09-01T10:00:00+00:00")),
        e2e=write(tmp_path / "e-new.json",
                  e2e_artifact(run_utc="2026-09-01T11:00:00+00:00", false_answers=["SP-02"])),
        baseline=None,
    )
    assert not result.ok
    assert "false_answer = 1" in result.failures[0]
    assert "SP-02" in result.failures[0]


@pytest.mark.parametrize(
    "current, regressed",
    [
        (0.90, False),   # improved
        (0.88, False),   # unchanged
        (0.85, False),   # exactly 3 points down: the boundary is allowed
        (0.8499, True),  # a hair past the boundary
        (0.70, True),    # collapse
    ],
)
def test_r_at_3_regression_boundary(tmp_path, current, regressed):
    result = evaluate(
        retrieval=write(tmp_path / "r-new.json",
                        retrieval_artifact(current, run_utc="2026-09-01T10:00:00+00:00")),
        e2e=write(tmp_path / "e-new.json", e2e_artifact(run_utc="2026-09-01T11:00:00+00:00")),
        baseline=write(tmp_path / "r-old.json",
                       retrieval_artifact(0.88, run_utc="2026-07-16T18:00:00+00:00")),
        max_r3_drop=0.03,
    )
    assert result.ok is not regressed
    if regressed:
        assert "headline R@3 fell" in result.failures[0]


def test_missing_hybrid_column_fails_with_the_cache_command(tmp_path):
    """No hybrid column = the dense path was skipped = the embedding cache was
    missing. The run measured BM25-only, so its numbers are not the release."""
    result = evaluate(
        retrieval=write(tmp_path / "r-new.json",
                        retrieval_artifact(0.82, run_utc="2026-09-01T10:00:00+00:00",
                                           retrievers=("bm25",))),
        e2e=write(tmp_path / "e-new.json", e2e_artifact(run_utc="2026-09-01T11:00:00+00:00")),
        baseline=None,
    )
    assert not result.ok
    assert "scripts.pack_eval_cache" in result.failures[0]


def test_no_baseline_skips_the_regression_check(tmp_path):
    result = evaluate(
        retrieval=write(tmp_path / "r-new.json",
                        retrieval_artifact(0.50, run_utc="2026-09-01T10:00:00+00:00")),
        e2e=write(tmp_path / "e-new.json", e2e_artifact(run_utc="2026-09-01T11:00:00+00:00")),
        baseline=None,
    )
    assert result.ok
    assert "no earlier committed run" in result.notes[0]


# --------------------------------------------------------------------------
# step summary + CLI
# --------------------------------------------------------------------------

def test_summary_markdown_has_a_table_and_the_verdict(tmp_path):
    retrieval = write(tmp_path / "r-new.json",
                      retrieval_artifact(0.90, run_utc="2026-09-01T10:00:00+00:00"))
    e2e = write(tmp_path / "e-new.json", e2e_artifact(run_utc="2026-09-01T11:00:00+00:00"))
    result = evaluate(retrieval=retrieval, e2e=e2e, baseline=None)

    markdown = summary_markdown(result, eval_set="dev", retrieval=retrieval, e2e=e2e)
    assert "| check | value | baseline / rule | verdict |" in markdown
    assert "PASS" in markdown.splitlines()[0]
    assert "| false_answer | 0 |" in markdown
    assert "| hybrid | 90.0% |" in markdown
    assert "gemini-2.5-flash" in markdown


def test_cli_returns_zero_and_writes_the_step_summary(tmp_path):
    results = tmp_path / "results"
    write(results / "retrieval-2026-07-16.json",
          retrieval_artifact(0.88, run_utc="2026-07-16T18:00:00+00:00", eval_set=None))
    write(results / "retrieval-2026-09-01.json",
          retrieval_artifact(0.90, run_utc="2026-09-01T10:00:00+00:00"))
    write(results / "e2e-gemini-2.5-flash-2026-09-01.json",
          e2e_artifact(run_utc="2026-09-01T11:00:00+00:00"))
    summary = tmp_path / "summary.md"

    code = main(["--results-dir", str(results), "--eval-set", "dev", "--summary", str(summary)])

    assert code == 0
    assert "Eval gate" in summary.read_text(encoding="utf-8")


def test_cli_returns_one_when_the_gate_fails(tmp_path):
    results = tmp_path / "results"
    write(results / "retrieval-2026-09-01.json",
          retrieval_artifact(0.90, run_utc="2026-09-01T10:00:00+00:00"))
    write(results / "e2e-gemini-2.5-flash-2026-09-01.json",
          e2e_artifact(run_utc="2026-09-01T11:00:00+00:00", false_answers=["SP-02"]))

    assert main(["--results-dir", str(results), "--eval-set", "dev"]) == 1


def test_cli_returns_two_when_an_artifact_is_missing(tmp_path):
    assert main(["--results-dir", str(tmp_path), "--eval-set", "dev"]) == 2


# --- arm matching: an experimental arm is not a baseline for the default ------

def test_arm_of_reads_both_runner_shapes_and_defaults_when_absent():
    assert arm_of({"arm": {"reranker": "gemini", "rewrite": "on"}}) == "reranker=gemini,rewrite=on"
    assert arm_of({"arm": {"reranker": "none", "rewrite": "off"}}) == "reranker=none,rewrite=off"
    assert arm_of({"scope_stage2": {"arm": "embed"}}) == "scope_stage2=embed"
    assert arm_of({"run_utc": "2026-07-16T18:00:00+00:00"}) == "default"


def test_a_reranked_arm_is_not_the_baseline_for_the_default_arm(tmp_path):
    """The rerank/rewrite arms land in the same directory and score higher.

    Grading the shipped default against one of them fails the build for a
    configuration nobody ships, which is exactly the false alarm check 3 exists
    to avoid.
    """
    results = tmp_path / "results"
    default_older = retrieval_artifact(0.82, run_utc="2026-09-01T08:00:00+00:00")
    default_older["arm"] = {"reranker": "none", "rewrite": "off"}
    reranked = retrieval_artifact(0.95, run_utc="2026-09-01T09:00:00+00:00")
    reranked["arm"] = {"reranker": "gemini", "rewrite": "on"}
    default_latest = retrieval_artifact(0.81, run_utc="2026-09-01T10:00:00+00:00")
    default_latest["arm"] = {"reranker": "none", "rewrite": "off"}

    write(results / "retrieval-2026-09-01-old.json", default_older)
    write(results / "retrieval-gemini-rewriteon-2026-09-01.json", reranked)
    write(results / "retrieval-2026-09-01.json", default_latest)
    write(results / "e2e-gemini-2.5-flash-2026-09-01.json",
          e2e_artifact(run_utc="2026-09-01T11:00:00+00:00"))

    # 81% vs the 82% default baseline is a 1-point drop (allowed); vs the 95%
    # reranked arm it would be 14 points and would fail.
    assert main(["--results-dir", str(results), "--eval-set", "dev"]) == 0
