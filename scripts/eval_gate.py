"""The release gate, as code: read the eval artifacts and decide pass/fail.

DEPLOY.md has always stated the rule in prose ("a false_answer > 0 is a release
blocker, full stop"). Prose does not fail a build. This turns the rule into an
exit code, and adds the second half a human would otherwise eyeball: silent
retrieval decay.

Three checks, all against the committed artifacts in evals/results/:

  1. false answers    : e2e `matrix.false_answer` must be empty. Confabulation
                        blocks a release. Note the runners OMIT the key when the
                        cell is empty, so "missing" means zero, but a payload
                        with no `matrix` at all is malformed and fails.
  2. retriever present : the retrieval artifact must contain the hybrid column.
                        Its absence means the dense path was skipped, i.e. the
                        embedding cache was missing and the run silently
                        measured BM25-only. That is a cost/validity guard, not
                        a quality one.
  3. R@3 regression   : headline R@3 must not fall more than `--max-r3-drop`
                        below the last committed run of the SAME eval set with
                        the SAME retriever and the SAME arm. Comparing
                        hybrid-to-bm25, dev-to-test, or the shipped default to a
                        reranked experiment would manufacture false alarms.

"Latest" and "baseline" are chosen by the `run_utc` INSIDE each artifact, not by
filename or mtime: the run that just finished is by definition the newest, and
everything older is committed history.

Run:  uv run python -m scripts.eval_gate --eval-set dev --summary "$GITHUB_STEP_SUMMARY"
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

RESULTS_DIR = Path("evals/results")
HEADLINE_LABEL = "HEADLINE (answer)"
DEFAULT_RETRIEVER = "hybrid"
DEFAULT_MAX_R3_DROP = 0.03  # 3 percentage points
_EPS = 1e-9

_PACK_HINT = (
    "the retrieval run has no `hybrid` column, which happens when the embedding cache is "
    "absent and the dense path is skipped. Re-pack and re-upload the cache: "
    "`uv run python -m scripts.pack_eval_cache`, then "
    "`gh release upload eval-cache-v1 eval-cache-v1.tar.gz --clobber`."
)


class GateError(RuntimeError):
    """Raised when an artifact is missing or malformed (as opposed to failing)."""


@dataclass
class GateResult:
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    rows: list[tuple[str, str, str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def _run_ts(payload: dict) -> datetime:
    """Sortable UTC timestamp for an artifact; missing/odd values sort oldest."""
    raw = payload.get("run_utc")
    if not isinstance(raw, str):
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def eval_set_of(path: Path, payload: dict, prefix: str) -> str:
    """The set an artifact belongs to. Early runs predate the `eval_set` key, so
    fall back to the `-test` filename suffix the runners have always written."""
    declared = payload.get("eval_set")
    if isinstance(declared, str) and declared:
        return declared
    return "test" if path.name.startswith(f"{prefix}-test-") else "dev"


def arm_of(payload: dict) -> str:
    """The experiment ARM an artifact measured, as a comparable key.

    The optional stages write their own artifacts into the same directory: the
    retrieval runner records `arm` (reranker, rewrite), the e2e runner records
    `scope_stage2.arm`. An arm is a DIFFERENT configuration, so its numbers are
    not a baseline for the shipped default — grading one against the other
    manufactures the same false alarm as comparing hybrid to bm25. Artifacts
    written before the arms existed carry no arm field and read as the default.
    """
    arm = payload.get("arm")
    if isinstance(arm, dict):
        return f"reranker={arm.get('reranker') or 'none'},rewrite={arm.get('rewrite') or 'off'}"
    stage2 = payload.get("scope_stage2")
    if isinstance(stage2, dict):
        return f"scope_stage2={stage2.get('arm') or 'off'}"
    return "default"


def load_runs(results_dir: Path, prefix: str, eval_set: str) -> list[tuple[Path, dict]]:
    """Every artifact of one kind and one eval set, NEWEST FIRST by run_utc."""
    runs: list[tuple[Path, dict]] = []
    for path in sorted(results_dir.glob(f"{prefix}*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GateError(f"{path} is not readable JSON: {exc}") from exc
        if not isinstance(payload, dict):
            continue
        if eval_set_of(path, payload, prefix) == eval_set:
            runs.append((path, payload))
    runs.sort(key=lambda item: _run_ts(item[1]), reverse=True)
    return runs


def false_answer_ids(payload: dict) -> list[str]:
    """The confabulation cell. Absent key = empty cell; absent matrix = malformed."""
    matrix = payload.get("matrix")
    if not isinstance(matrix, dict):
        raise GateError("e2e artifact has no `matrix`: it is not a run of evals.run_e2e_eval")
    return list(matrix.get("false_answer", []))


def headline_r_at_3(payload: dict, retriever: str) -> float | None:
    """Headline (answerable-question) R@3 for one retriever, or None if absent."""
    rows = payload.get("retrievers", {}).get(retriever)
    if not isinstance(rows, list):
        return None
    for row in rows:
        if row.get("label") == HEADLINE_LABEL:
            rate = row.get("recall", {}).get("@3", {}).get("rate")
            return float(rate) if rate is not None else None
    return None


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def evaluate(
    *,
    retrieval: tuple[Path, dict],
    e2e: tuple[Path, dict],
    baseline: tuple[Path, dict] | None,
    retriever: str = DEFAULT_RETRIEVER,
    max_r3_drop: float = DEFAULT_MAX_R3_DROP,
) -> GateResult:
    result = GateResult()
    retrieval_path, retrieval_payload = retrieval
    e2e_path, e2e_payload = e2e

    # --- check 1: false answers ------------------------------------------------
    bad = false_answer_ids(e2e_payload)
    verdict = "FAIL" if bad else "PASS"
    detail = ", ".join(bad) if bad else "none"
    result.rows.append(("false answers (e2e)", str(len(bad)), "must be 0", verdict))
    if bad:
        result.failures.append(
            f"false_answer = {len(bad)} in {e2e_path.name} ({detail}). Confabulation blocks release"
        )

    # --- check 2: the hybrid column exists (cache/cost guard) ------------------
    current = headline_r_at_3(retrieval_payload, retriever)
    if current is None:
        result.rows.append((f"headline R@3 ({retriever})", "missing", "required", "FAIL"))
        result.failures.append(f"{retrieval_path.name}: {_PACK_HINT}")
        return result

    # --- check 3: R@3 regression vs the last committed run of the same set -----
    if baseline is None:
        result.rows.append((f"headline R@3 ({retriever})", _pct(current), "no baseline", "PASS"))
        result.notes.append("no earlier committed run for this eval set, regression check skipped")
        return result

    baseline_path, baseline_payload = baseline
    previous = headline_r_at_3(baseline_payload, retriever)
    if previous is None:
        result.rows.append(
            (f"headline R@3 ({retriever})", _pct(current), f"n/a ({baseline_path.name})", "PASS")
        )
        result.notes.append(
            f"baseline {baseline_path.name} has no `{retriever}` column, regression check skipped"
        )
        return result

    drop = previous - current
    regressed = drop > max_r3_drop + _EPS
    result.rows.append(
        (
            f"headline R@3 ({retriever})",
            f"{_pct(current)} ({current - previous:+.1%} vs baseline)",
            f"{_pct(previous)} ({baseline_path.name})",
            "FAIL" if regressed else "PASS",
        )
    )
    if regressed:
        result.failures.append(
            f"headline R@3 fell {drop:.1%} ({_pct(previous)} -> {_pct(current)}), more than the "
            f"{max_r3_drop:.0%} allowed, vs {baseline_path.name}"
        )
    return result


def summary_markdown(
    result: GateResult,
    *,
    eval_set: str,
    retrieval: tuple[Path, dict],
    e2e: tuple[Path, dict],
) -> str:
    _, retrieval_payload = retrieval
    e2e_path, e2e_payload = e2e
    matrix = e2e_payload.get("matrix", {})
    lines = [
        f"## Eval gate: `{eval_set}` set: {'PASS' if result.ok else 'FAIL'}",
        "",
        "| check | value | baseline / rule | verdict |",
        "| --- | --- | --- | --- |",
    ]
    lines += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in result.rows]

    lines += ["", "### Refusal matrix", "", "| cell | n |", "| --- | --- |"]
    for cell in ("correct_answer", "correct_refuse", "false_refuse", "false_answer"):
        lines.append(f"| {cell} | {len(matrix.get(cell, []))} |")

    def _rate(key: str) -> str:
        block = e2e_payload.get(key) or {}
        total = block.get("total") or 0
        return f"{block.get('passed', 0)}/{total}" + (
            f" ({block['passed'] / total:.0%})" if total else ""
        )

    lines += [
        "",
        "### Retrieval: headline R@3 by retriever",
        "",
        "| retriever | R@3 |",
        "| --- | --- |",
    ]
    for name in sorted(retrieval_payload.get("retrievers", {})):
        lines.append(f"| {name} | {_pct(headline_r_at_3(retrieval_payload, name))} |")

    lines += [
        "",
        f"citation validity {_rate('citation_validity')} · "
        f"numeric provenance {_rate('numeric_provenance')} · "
        f"model `{e2e_payload.get('generate_model', 'unrecorded')}` · "
        f"top_k {e2e_payload.get('top_k', 'n/a')}",
        "",
        f"artifacts: `{retrieval[0].as_posix()}`, `{e2e_path.as_posix()}`",
    ]
    for note in result.notes:
        lines.append(f"> note: {note}")
    for failure in result.failures:
        lines.append(f"> **FAIL** {failure}")
    return "\n".join(lines) + "\n"


def _resolve(
    explicit: Path | None, results_dir: Path, prefix: str, eval_set: str
) -> tuple[tuple[Path, dict], tuple[Path, dict] | None]:
    """(latest, baseline) for one artifact kind."""
    runs = load_runs(results_dir, prefix, eval_set)
    if explicit is not None:
        payload = json.loads(explicit.read_text(encoding="utf-8"))
        arm = arm_of(payload)
        others = [
            r for r in runs
            if r[0].resolve() != explicit.resolve() and arm_of(r[1]) == arm
        ]
        return (explicit, payload), (others[0] if others else None)
    if not runs:
        raise GateError(
            f"no `{prefix}` artifact for the {eval_set} set in {results_dir}. Did the eval run?"
        )
    latest = runs[0]
    arm = arm_of(latest[1])
    same_arm = [r for r in runs[1:] if arm_of(r[1]) == arm]
    return latest, (same_arm[0] if same_arm else None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fail the build on eval regressions.")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--eval-set", default="dev", choices=["dev", "test"])
    parser.add_argument("--retriever", default=DEFAULT_RETRIEVER)
    parser.add_argument("--max-r3-drop", type=float, default=DEFAULT_MAX_R3_DROP)
    parser.add_argument("--retrieval", type=Path, help="override the retrieval artifact")
    parser.add_argument("--e2e", type=Path, help="override the e2e artifact")
    parser.add_argument("--summary", type=Path, help="append the markdown table to this file")
    args = parser.parse_args(argv)

    try:
        retrieval, baseline = _resolve(args.retrieval, args.results_dir, "retrieval", args.eval_set)
        e2e, _ = _resolve(args.e2e, args.results_dir, "e2e", args.eval_set)
    except GateError as exc:
        print(f"EVAL GATE ERROR: {exc}", file=sys.stderr)
        return 2

    try:
        result = evaluate(
            retrieval=retrieval,
            e2e=e2e,
            baseline=baseline,
            retriever=args.retriever,
            max_r3_drop=args.max_r3_drop,
        )
    except GateError as exc:
        print(f"EVAL GATE ERROR: {exc}", file=sys.stderr)
        return 2

    markdown = summary_markdown(result, eval_set=args.eval_set, retrieval=retrieval, e2e=e2e)
    print(markdown)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as handle:
            handle.write(markdown)

    if not result.ok:
        print("EVAL GATE FAILED:", file=sys.stderr)
        for failure in result.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("eval gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
