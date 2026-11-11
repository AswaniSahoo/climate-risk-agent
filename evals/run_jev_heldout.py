"""One-time held-out check of Jev question triage (adr/0002, criterion 6).

ADR 0002 lets `CRG_JEV=triage` become a deployment default only after "one
held-out run at the next release gate, counted as an exposure". This is that
run, and it is the only code that sends held-out questions to Jev:
evals/run_jev_eval.py cannot reach the held-out set at all, and
evals/run_e2e_eval.py refuses a held-out run while CRG_JEV is on.

It sends what the shipped path sends and nothing more. A held-out question
goes to Jev only when `CRG_JEV=triage` itself would send it: stage 1 sees no
hazard word in it (the silent bucket), or stage 1 lets it in but the front
door cannot name its hazard. The dev eval's "Jev alone" arm is not run here,
because it would send every question and the pass conditions below are about
the shipped path. No Gemini or Vertex call can happen: their credentials are
unset before anything else runs, and scope stage 2 is held off.

Pass conditions, fixed before the run from the ADR's criteria, with the ADR's
thresholds unchanged (rag/jev_triage.py):

1. No new false refusal at scope. Every question the shipped path refuses at
   the guard that stage 1 alone does not refuse must be one the set labels for
   refusal (`out_of_scope_hazard` or `out_of_corpus`), so premise-injection
   questions still reach the answering model.
2. No wrong hazard named by the front door. A missed hazard is allowed: it
   falls back to today's refusal.
3. Stability. At least 90% of the questions Jev was asked keep their decision
   with the topic options reversed, and across two uncached runs; every
   refusal that flips is listed.

Latency (criterion 5) is measured and reported but does not decide: the dev
run missed the 1,000 ms p95 gate through the gateway, and Aswani accepted that
trade-off on 2026-11-11. Usefulness (criterion 3) was shown on the dev set and
is reported here, not gated, because the held-out questions were not written
to dodge the lexical stage.

A live run needs --spend-exposure, and it refuses to run again once a live
artifact exists, unless --new-exposure says a second exposure is intended.

Run:
  uv run python -m evals.run_jev_heldout                  # offline: stage 1 and what would be sent
  JEV_BASE_URL=https://api.experientiallabs.ai uv run python -m evals.run_jev_heldout --spend-exposure
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals import run_jev_eval as dev
from evals.gold_set import TEST_SET_FROZEN_HASH_PATH, load_test_set

RESULTS_DIR = Path("evals/results")
ARTIFACT_PREFIX = "jev-triage-test-"
STABLE_SHARE = 0.9  # criterion 4 in the ADR, criterion 3 here
GATEWAY_PRIVACY = (
    "The Experiential Labs privacy notice (last updated September 16, 2026, read "
    "2026-11-11 at https://www.experientiallabs.ai/privacy) says: \"The standard hosted "
    "service records prompts, responses and related activity.\" and \"Stored prompts and "
    "responses in the primary capture table are automatically removed after 30 days. "
    "That period does not describe every trace, file, backup or provider-held copy.\""
)


def held_out_rows() -> list[dict[str, Any]]:
    """One row per held-out question, labelled the way the dev eval labels its rows."""
    rows = []
    for q in load_test_set().questions:
        rows.append({
            "id": q.id, "set": "test", "slice": q.slice.value, "question": q.question,
            "should_refuse": q.slice.value in dev.SCOPE_REFUSE_SLICES,
            "expected_hazard": q.hazard if q.hazard in dev.OUR_HAZARDS else None,
        })
    return rows


def mark_what_is_sent(row: dict[str, Any]) -> None:
    """Stage 1 and the lexical parser, offline, and whether the shipped path asks Jev."""
    dev.lexical_baseline(row)
    # agent/nl.py's second use of Jev: stage 1 let the question in by its own
    # vocabulary, but the front door's patterns cannot say which hazard.
    row["front_door_gap"] = (not row["stage1_refused"] and not row["silent_bucket"]
                             and row["lexical_hazard"] is None)
    row["sent_to_jev"] = row["silent_bucket"] or row["front_door_gap"]


def _reading(reading: Any, *, may_refuse: bool) -> dict[str, Any]:
    return {"decision": reading.decision, "may_refuse": may_refuse,
            "out_of_scope": reading.out_of_scope, "hazard": reading.hazard,
            "hazard_p": reading.hazard_probability, "topic": reading.topic,
            "refuse_p": reading.refuse_probability, "noul": reading.about_assessed_hazard,
            "status": reading.status}


def run_shipped(rows: list[dict[str, Any]], *, budget_usd: float, repeats: int) -> dict[str, Any]:
    """The path CRG_JEV=triage ships, on every row, then the two stability checks."""
    from agent.nl import parse_query
    from rag.jev_triage import TOPIC_OPTIONS, triage
    from rag.scope import clear_stage2_cache, scope_verdict

    os.environ["CRG_JEV"] = "triage"
    clear_stage2_cache()
    for row in rows:
        decision = scope_verdict(row["question"])  # stage 1, then Jev on the silent bucket
        row["shipped_refused"] = decision.out_of_scope is not None
        row["shipped_stage"] = decision.stage
        row["shipped_out_of_scope"] = decision.out_of_scope
        row["shipped_unavailable"] = (decision.jev is not None
                                      and decision.jev.decision == "unavailable")
        parsed = parse_query(row["question"])  # the front door, as a person would reach it
        row["front_door_hazard"] = parsed.hazard.value if parsed.hazard else None
        row["front_door_source"] = parsed.triage.hazard_source if parsed.triage else None
        row["front_door_unavailable"] = bool(
            parsed.triage and parsed.triage.jev_status.startswith("unavailable"))
        if row["silent_bucket"] and decision.jev is not None:
            row["jev"] = _reading(decision.jev, may_refuse=True)
        elif row["front_door_gap"]:
            # The same request the front door just made, so the client's cache answers it.
            row["jev"] = _reading(triage(row["question"], may_refuse=False), may_refuse=False)
        dev._check_budget(budget_usd)

    asked = [r for r in rows if r.get("jev") and r["jev"]["decision"] != "unavailable"]
    flips, flip_skipped = [], []
    for row in asked:
        flipped = triage(row["question"], may_refuse=row["jev"]["may_refuse"],
                         topic_order=tuple(reversed(TOPIC_OPTIONS)), use_cache=False)
        row["reversed_decision"] = flipped.decision
        if flipped.decision == "unavailable":
            flip_skipped.append(row["id"])
        elif flipped.decision != row["jev"]["decision"]:
            flips.append(row["id"])
        dev._check_budget(budget_usd)
    unstable, repeat_skipped = [], []
    for row in asked:
        seen = {row["jev"]["decision"]}
        for _ in range(max(1, repeats) - 1):
            seen.add(triage(row["question"], may_refuse=row["jev"]["may_refuse"],
                            use_cache=False).decision)
            dev._check_budget(budget_usd)
        if "unavailable" in seen:
            repeat_skipped.append(row["id"])
        elif len(seen) > 1:
            unstable.append(row["id"])
    refusing = {r["id"] for r in asked if r["jev"]["decision"] == "refuse"}
    return {
        "order_check": {"n": len(asked) - len(flip_skipped), "decision_flips": flips,
                        "refusals_that_flipped": sorted(refusing & set(flips)),
                        "skipped_unavailable": flip_skipped},
        "repeats": {"runs": max(1, repeats), "n": len(asked) - len(repeat_skipped),
                    "unstable": unstable, "refusals_that_flipped": sorted(refusing & set(unstable)),
                    "skipped_unavailable": repeat_skipped},
    }


def _kept_share(check: dict[str, Any], changed_key: str) -> float | None:
    if not check["n"]:
        return None
    return round(1 - len(check[changed_key]) / check["n"], 4)


def gates(artifact: dict[str, Any]) -> dict[str, Any]:
    """The three pass conditions in the module docstring, read off the scores."""
    scope, hazard = artifact["scope"], artifact["hazard"]
    new_false_refusals = sorted(set(scope["shipped"]["false_refusals"])
                                - set(scope["stage1"]["false_refusals"]))
    order = _kept_share(artifact["order_check"], "decision_flips")
    repeat = _kept_share(artifact["repeats"], "unstable")
    return {
        "1_no_new_false_refusal": {"met": not new_false_refusals,
                                   "new_false_refusals": new_false_refusals},
        "2_no_wrong_hazard": {"met": not hazard["front_door"]["wrong"],
                              "wrong": hazard["front_door"]["wrong"]},
        "3_stability": {"met": (order is None or order >= STABLE_SHARE)
                        and (repeat is None or repeat >= STABLE_SHARE),
                        "kept_with_options_reversed": order, "kept_across_runs": repeat},
    }


def write_report(a: dict[str, Any], path: Path) -> None:
    """The dated markdown report, rendered from the artifact so no number is retyped."""
    j = a["jev"]
    sent = [r for r in a["rows"] if r["sent_to_jev"]]
    lines = [
        f"# Jev triage, held-out check, {a['date']}",
        "",
        f"Run {a['run_utc']}. Held-out set (`evals/gold_set_v2.json`, {a['questions']} "
        f"questions, frozen hash in `{a['sha256_file']}`: {a['sha256_ok']}). Triage only: no "
        "Gemini or Vertex call, no citation check, scope stage 2 off.",
        "",
        f"- Exposure: {a['exposure']}",
        f"- Sent to Jev: {len(sent)} of {a['questions']} questions, the ones `CRG_JEV=triage` "
        f"itself sends ({sum(r['silent_bucket'] for r in sent)} in the silent bucket, "
        f"{sum(r['front_door_gap'] for r in sent)} where the front door cannot name the "
        "hazard stage 1 found). The other questions never left this machine.",
        f"- Privacy: {a['privacy']}",
        f"- Jev status: **{j['status']}**, through provider `{j['provider']}`, model requested "
        f"`{j['model_requested']}`, models that answered: "
        f"{', '.join(j['models_answered']) or 'none'}",
        f"- Live Jev calls: {j['live_calls']}, failures: {j['failures']}, retries: "
        f"{j['retries']}, input tokens: {j['input_tokens']}",
        f"- Spend: ${j['spend_usd']:.6f} at TypeSafe's list price (cap ${a['budget_usd']:.2f}); "
        "reported by the provider: " + (f"${j['reported_spend_usd']:.6f}"
                                         if j["reported_spend_usd"] is not None
                                         else "nothing reported"),
        "- Latency per live call: " + (
            f"p50 {j['latency_ms_p50']} ms, p95 {j['latency_ms_p95']} ms "
            f"({j['calls_over_1000_ms']} of {j['live_calls'] - j['failures']} calls over 1,000 ms)"
            if j["latency_ms_p50"] is not None else "not measured"),
        f"- Requests paced {j['pacing_s_between_requests']:g} s apart; pacing waits are not "
        "counted as latency. The per-call budget was widened for the run so a rate-limited "
        "call could back off; production keeps the provider's own budget.",
        f"- Decision thresholds (unchanged from the dev run): refuse at summed refusal "
        f"probability >= {a['thresholds']['refuse_p']} with the companion noul <= "
        f"{a['thresholds']['noul_no']}; hint at >= {a['thresholds']['hint_p']} with the noul "
        f">= {a['thresholds']['noul_yes']}",
        "",
        "## Pass conditions (fixed before the run)",
        "",
        "| condition | result | detail |",
        "| --- | --- | --- |",
    ]
    if a.get("gates"):
        g = a["gates"]
        lines += [
            "| 1. no new false refusal at scope | "
            f"{'met' if g['1_no_new_false_refusal']['met'] else 'NOT met'} | new false "
            f"refusals: {dev._fmt_ids(g['1_no_new_false_refusal']['new_false_refusals'])} |",
            "| 2. no wrong hazard named by the front door | "
            f"{'met' if g['2_no_wrong_hazard']['met'] else 'NOT met'} | wrong: "
            f"{dev._fmt_ids(g['2_no_wrong_hazard']['wrong'])} |",
            f"| 3. stability, at least {STABLE_SHARE:.0%} kept | "
            f"{'met' if g['3_stability']['met'] else 'NOT met'} | kept with the options "
            f"reversed: {g['3_stability']['kept_with_options_reversed']}; kept across two "
            f"uncached runs: {g['3_stability']['kept_across_runs']} |",
        ]
    else:
        lines.append("| all | not measured | Jev was not called |")
    s, h = a["scope"], a["hazard"]
    lines += [
        "",
        "## Scope: refuse at the guard or not",
        "",
        "| arm | correct | false refusals | false accepts |",
        "| --- | --- | --- | --- |",
        f"| stage 1 today | {dev._pct(s['stage1'])} | {dev._fmt_ids(s['stage1']['false_refusals'])} "
        f"| {dev._fmt_ids(s['stage1']['false_accepts'])} |",
    ]
    if s["shipped"].get("accuracy") is not None:
        lines.append(
            f"| stage 1 + Jev (`CRG_JEV=triage`) | {dev._pct(s['shipped'])} | "
            f"{dev._fmt_ids(s['shipped']['false_refusals'])} | "
            f"{dev._fmt_ids(s['shipped']['false_accepts'])} |")
        if s["shipped"].get("excluded_unavailable"):
            lines.append(f"\nLeft out because Jev was unavailable for them: "
                         f"{dev._fmt_ids(s['shipped']['excluded_unavailable'])}")
    lines += [
        "",
        "A false accept is a question that should be refused at scope reaching the answering "
        "model, which refuses it itself today (held-out e2e, 2026-07-22: all 35 questions built "
        "to be refused were refused). It is a missed instant refusal, not a false answer.",
        "",
        "## Hazard named by the front door",
        "",
        "| arm | right | missed | wrong |",
        "| --- | --- | --- | --- |",
        f"| lexical parser today | {h['lexical']['right']}/{h['lexical']['n']} | "
        f"{dev._fmt_ids(h['lexical']['missed'])} | {dev._fmt_ids(h['lexical']['wrong'])} |",
    ]
    if h["front_door"].get("accuracy") is not None:
        lines.append(
            f"| front door with Jev | {h['front_door']['right']}/{h['front_door']['n']} | "
            f"{dev._fmt_ids(h['front_door']['missed'])} | "
            f"{dev._fmt_ids(h['front_door']['wrong'])} |")
    if a.get("usefulness"):
        u = a["usefulness"]
        lines += ["", "## Usefulness (reported, not gated)", "",
                  f"- Refused at the guard with Jev that stage 1 lets through: "
                  f"{dev._fmt_ids(u['refusals_added'])}",
                  f"- Hazards the front door names with Jev that the lexical parser misses: "
                  f"{dev._fmt_ids(u['hazards_added'])}"]
    if a.get("order_check"):
        oc, rp = a["order_check"], a["repeats"]
        lines += ["", "## Stability", "",
                  f"- Topic options reversed: {len(oc['decision_flips'])} of {oc['n']} decisions "
                  f"changed ({dev._fmt_ids(oc['decision_flips'])}); refusals among them: "
                  f"{dev._fmt_ids(oc['refusals_that_flipped'])}; left out, Jev unavailable: "
                  f"{dev._fmt_ids(oc['skipped_unavailable'])}.",
                  f"- {rp['runs']} uncached runs per question: {len(rp['unstable'])} of {rp['n']} "
                  f"decisions were not the same every time ({dev._fmt_ids(rp['unstable'])}); "
                  f"refusals among them: {dev._fmt_ids(rp['refusals_that_flipped'])}; left out, "
                  f"Jev unavailable: {dev._fmt_ids(rp['skipped_unavailable'])}."]
    lines += ["", "## Questions sent to Jev, as it read them", "",
              "| id | slice | should refuse | asked as | decision | refuse p | noul | hazard (p) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in sent:
        r = row.get("jev")
        asked_as = "silent bucket" if row["silent_bucket"] else "front-door gap"
        if r is None:
            lines.append(f"| {row['id']} | {row['slice']} | {row['should_refuse']} | {asked_as} "
                         "| not run | | | |")
            continue
        lines.append(f"| {row['id']} | {row['slice']} | {row['should_refuse']} | {asked_as} | "
                     f"{r['decision']} | {r['refuse_p']} | {r['noul']} | "
                     f"{r['hazard']} ({r['hazard_p']}) |")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in a["caveats"]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-time held-out check of Jev triage.")
    parser.add_argument("--spend-exposure", action="store_true",
                        help="send what the shipped path sends to Jev; without it, offline only")
    parser.add_argument("--new-exposure", action="store_true",
                        help="run live even though a live held-out artifact already exists")
    parser.add_argument("--budget-usd", type=float, default=1.0,
                        help="hard stop on Jev spend at the list price (default 1.00)")
    parser.add_argument("--repeats", type=int, default=2,
                        help="uncached runs per question Jev was asked (default 2)")
    parser.add_argument("--max-rpm", type=float, default=8.0,
                        help="client-side cap on Jev requests a minute (default 8)")
    parser.add_argument("--max-tokens", type=int, default=300_000,
                        help="hard stop on input plus output tokens (default 300,000)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    saved = {name: os.environ.get(name) for name in dev._SCOPED_ENV}
    try:
        return _run(_parse_args(argv))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _live_artifacts() -> list[Path]:
    found = []
    for path in sorted(RESULTS_DIR.glob(f"{ARTIFACT_PREFIX}*.json")):
        try:
            if json.loads(path.read_text(encoding="utf-8")).get("exposure_spent"):
                found.append(path)
        except (OSError, ValueError):
            continue
    return found


def _run(args: argparse.Namespace) -> int:
    from obs.log import configure

    configure()
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        os.environ.pop(name, None)  # nothing here may reach Gemini or Vertex
    os.environ["CRG_SCOPE_STAGE2"] = "off"
    live = args.spend_exposure
    if live and not args.new_exposure and (earlier := _live_artifacts()):
        print(f"a live held-out run already exists ({earlier[-1]}); pass --new-exposure to "
              "spend another exposure on purpose", file=sys.stderr)
        return 3
    dev._TOKEN_CAP = args.max_tokens
    if args.max_rpm > 0:
        os.environ["CRG_JEV_MIN_INTERVAL_S"] = f"{60.0 / args.max_rpm:g}"
    os.environ.setdefault("CRG_JEV_TRIAGE_BUDGET_S", "40")

    from rag.jev_client import JevError, get_client, get_provider
    from rag.jev_triage import HINT_P, NOUL_NO, NOUL_YES, REFUSE_P

    test_set = load_test_set()
    rows = held_out_rows()
    for row in rows:
        mark_what_is_sent(row)
    started = time.perf_counter()
    status = "ok" if live else "not run (offline: pass --spend-exposure to run it)"
    provider_name, model_requested = "not resolved", "not resolved"
    try:
        provider = get_provider()
        provider_name, model_requested = provider.name, provider.default_model
        if live:
            model_requested = get_client().model
    except JevError as exc:
        status = f"unavailable: {exc.reason}"
    extra: dict[str, Any] = {}
    try:
        if status == "ok":
            extra = run_shipped(rows, budget_usd=args.budget_usd, repeats=args.repeats)
    except dev.BudgetExceeded as exc:
        status = f"stopped: {exc}"
    finally:
        os.environ.pop("CRG_JEV", None)

    events = dev._jev_events()
    calls = [e for e in events if not e["cached"]]
    ok_calls = [e for e in calls if e["ok"]]
    if status == "ok" and calls and not ok_calls:
        status = f"unavailable: {calls[0].get('detail', 'every call failed')}"
    measured = status == "ok" and bool(ok_calls)
    tokens, spend = dev._jev_spend_usd()

    def scored(scorer, key: str, gone: str) -> dict[str, Any]:
        kept = [r for r in rows if not r.get(gone)]
        return {**scorer(kept, key), "excluded_unavailable": [r["id"] for r in rows if r.get(gone)]}

    scope: dict[str, dict[str, Any]] = {"stage1": dev.scope_scores(rows, "stage1_refused"),
             "shipped": (scored(dev.scope_scores, "shipped_refused", "shipped_unavailable")
                         if measured else {"n": len(rows), "accuracy": None})}
    hazard: dict[str, dict[str, Any]] = {"lexical": dev.hazard_scores(rows, "lexical_hazard"),
              "front_door": (scored(dev.hazard_scores, "front_door_hazard", "front_door_unavailable")
                             if measured else {"n": 0, "accuracy": None})}
    today = datetime.now(timezone.utc)
    sent = [r for r in rows if r["sent_to_jev"]]
    artifact: dict[str, Any] = {
        "date": f"{today:%Y-%m-%d}",
        "run_utc": today.isoformat(timespec="seconds"),
        "eval_set": "test",
        "questions": len(test_set.questions),
        "sha256_file": "evals/gold_set_v2.sha256",
        "sha256_ok": test_set.content_hash()
        == TEST_SET_FROZEN_HASH_PATH.read_text(encoding="utf-8").strip(),
        "exposure_spent": measured,
        "exposure": ("one exposure of the held-out set, the first for Jev; earlier dated runs on "
                     "it: 2026-07-18 (retrieval, e2e), 2026-07-22 (e2e), 2026-09-07 (retrieval)"
                     if measured else "none spent: Jev was not called"),
        "privacy": GATEWAY_PRIVACY if provider_name == "experientiallabs" else (
            "Requests went to TypeSafe directly; see LIMITATIONS.md for its terms."),
        "budget_usd": args.budget_usd,
        "wall_s": round(time.perf_counter() - started, 1),
        "sent_ids": [r["id"] for r in sent],
        "jev": {
            "status": status,
            "provider": provider_name,
            "model_requested": model_requested,
            "models_answered": sorted({e["model"] for e in ok_calls}),
            "live_calls": len(calls),
            "cached_calls": len(events) - len(calls),
            "failures": len(calls) - len(ok_calls),
            "retries": sum(e["retries"] for e in calls),
            "input_tokens": tokens,
            "input_plus_output_tokens": dev._jev_tokens_all(),
            "spend_usd": round(spend, 6),
            "reported_spend_usd": dev._jev_reported_usd(),
            "pacing_s_between_requests": float(os.environ.get("CRG_JEV_MIN_INTERVAL_S", 0) or 0),
            "latency_ms_p50": dev.percentile([e["latency_ms"] for e in ok_calls], 0.50),
            "latency_ms_p95": dev.percentile([e["latency_ms"] for e in ok_calls], 0.95),
            "calls_over_1000_ms": sum(1 for e in ok_calls if e["latency_ms"] > 1000),
            "failure_details": sorted({e.get("detail", "") for e in calls if not e["ok"]}),
        },
        "thresholds": {"refuse_p": REFUSE_P, "hint_p": HINT_P, "noul_yes": NOUL_YES,
                       "noul_no": NOUL_NO},
        "scope": scope,
        "hazard": hazard,
        **extra,
        "rows": rows,
        "caveats": [
            "The thresholds and question wording are the dev run's, unchanged; nothing was "
            "tuned on this set, and a failure here is diagnosed on the dev set, never by "
            "iterating against this one.",
            "Counts are small: few held-out questions lack every hazard word, so most of the "
            "set never reaches Jev, and a difference of one item is not evidence of a difference.",
            "The questions sent are now held by the gateway and by TypeSafe under their terms "
            "(LIMITATIONS.md), so a later held-out run cannot assume those providers never saw them.",
            "No prompt-injection probes were sent: TypeSafe's terms bar testing their systems.",
        ],
    }
    if measured:
        artifact["gates"] = gates(artifact)
        artifact["usefulness"] = {
            "refusals_added": sorted(set(scope["shipped"]["correct_refusals"])
                                     - set(scope["stage1"]["correct_refusals"])),
            "hazards_added": sorted(
                r["id"] for r in rows
                if r["expected_hazard"] and r.get("front_door_hazard") == r["expected_hazard"]
                and r["lexical_hazard"] != r["expected_hazard"]),
        }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"{ARTIFACT_PREFIX}{artifact['date']}.json"
    md_path = RESULTS_DIR / f"{ARTIFACT_PREFIX}{artifact['date']}.md"
    json_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    write_report(artifact, md_path)
    print(f"jev status: {status} | sent {len(sent)} of {len(rows)} | live calls {len(calls)} "
          f"| spend ${spend:.6f}")
    print(f"artifacts: {json_path} and {md_path}")
    if measured:
        print("gates:", json.dumps({k: v["met"] for k, v in artifact["gates"].items()}))
    return 0 if (measured or not live) else 2


if __name__ == "__main__":
    raise SystemExit(main())
