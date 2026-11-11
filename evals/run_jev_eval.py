"""Dev-set eval for Jev (adr/0002): triage against today's stage-1 scope guard,
and the citation support check, on the DEV set only.

The held-out test set (`evals/gold_set_v2.json`) is never read here: it runs at
release gates only, and this script has no switch that reaches it
(tests/test_jev_eval.py pins that). Everything below is diagnosed on
`evals/gold_set.json`, the 60-question dev set, plus the 14 paraphrase probes
that were authored for scope stage 2 in 2026-09 (evals/scope_stage2_calibration.py),
reported separately because they were written to dodge the lexical stage.

Triage labels come from the dev set itself, not from judgement made here:
- should refuse at scope: the `out_of_scope_hazard` and `out_of_corpus` slices.
  Everything else should pass the guard, INCLUDING `premise_injection`: those
  must reach the answering model so the refusal can cite the page that refutes
  the premise. A scope refusal there is the right cell for the wrong reason,
  and it is counted as a false refusal.
- expected hazard: the item's `hazard` when it is one of ours.
A "false accept" below means a question that should have been refused at
scope reached the answering model, which still refuses it on its own today
(e2e dev matrix 48/11/1/0). It is a missed instant refusal, not a false answer.

Citation-support pairs are derived mechanically from the dev set:
- `cid_true`: each `cid-table` item's recorded parse ("Parsed direction:
  increase; high confidence") written as a claim against its recorded source
  chunk. Expected: supports.
- `cid_flipped`: the same claim with the direction reversed. Expected: contradicts.
- `premise`: each `premise_injection` item's false premise against the gold
  chunk that refutes it. Expected: contradicts.
- `quote`: an item's hand-verified supporting quote against its own gold chunk.
  Expected: supports. Easy by construction: the claim is verbatim in the passage.
- `cross`: the same quote against another item's gold chunk on a different
  hazard and page. Expected: says nothing.

Run (no Gemini call anywhere; Jev calls need TYPESAFE_API_KEY):
  uv run python -m evals.run_jev_eval --dry-run          # offline: baseline and request sizes
  uv run python -m evals.run_jev_eval --task all --order-check --repeats 2
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from evals.gold_set import FROZEN_HASH_PATH, load_gold_set
from evals.schema import EvalQuestion

SCOPE_REFUSE_SLICES = frozenset({"out_of_scope_hazard", "out_of_corpus"})
OUR_HAZARDS = ("heatwave", "extreme_precip", "wind")
_HINT_NAME_TO_ENUM = {"heatwave": "heatwave", "extreme precipitation": "extreme_precip",
                      "wind": "wind"}
RESULTS_DIR = Path("evals/results")


# --- labels and baselines (pure, offline) ------------------------------------------

def dev_rows() -> list[dict[str, Any]]:
    """One row per dev question, with the labels this eval scores against."""
    rows = []
    for q in load_gold_set().questions:
        rows.append({
            "id": q.id, "set": "dev", "slice": q.slice.value, "question": q.question,
            "should_refuse": q.slice.value in SCOPE_REFUSE_SLICES,
            "expected_hazard": q.hazard if q.hazard in OUR_HAZARDS else None,
        })
    return rows


def probe_rows() -> list[dict[str, Any]]:
    """The stage-2 paraphrase probes (authored 2026-09, before this work)."""
    from evals.scope_stage2_calibration import PARAPHRASE_IN, PARAPHRASE_OUT

    rows = []
    for i, (text, hazard) in enumerate(PARAPHRASE_IN, 1):
        rows.append({"id": f"PIN-{i:02d}", "set": "probe", "slice": "paraphrase_in",
                     "question": text, "should_refuse": False,
                     "expected_hazard": _HINT_NAME_TO_ENUM.get(hazard or "")})
    for i, (text, _topic) in enumerate(PARAPHRASE_OUT, 1):
        rows.append({"id": f"POUT-{i:02d}", "set": "probe", "slice": "paraphrase_out",
                     "question": text, "should_refuse": True, "expected_hazard": None})
    return rows


def lexical_baseline(row: dict[str, Any]) -> None:
    """What today's code decides with every model off, written onto the row."""
    from agent.nl import _HAZARD_PATTERNS
    from rag.scope import has_supported_signal, out_of_scope_hazard

    out = out_of_scope_hazard(row["question"])
    row["stage1_out_of_scope"] = out
    row["stage1_refused"] = out is not None
    row["silent_bucket"] = out is None and not has_supported_signal(row["question"])
    row["lexical_hazard"] = next(
        (h.value for h, p in _HAZARD_PATTERNS.items() if p.search(row["question"])), None)


def scope_scores(rows: list[dict[str, Any]], refused_key: str) -> dict[str, Any]:
    """Accuracy, false refusals and false accepts for one way of refusing."""
    false_refusals = [r["id"] for r in rows if r[refused_key] and not r["should_refuse"]]
    false_accepts = [r["id"] for r in rows if not r[refused_key] and r["should_refuse"]]
    correct = len(rows) - len(false_refusals) - len(false_accepts)
    return {
        "n": len(rows),
        "correct": correct,
        "accuracy": round(correct / len(rows), 4) if rows else None,
        "correct_refusals": [r["id"] for r in rows if r[refused_key] and r["should_refuse"]],
        "false_refusals": false_refusals,
        "false_accepts": false_accepts,
    }


def hazard_scores(rows: list[dict[str, Any]], hazard_key: str) -> dict[str, Any]:
    """How often a reader named the right hazard, among items that have one."""
    labelled = [r for r in rows if r["expected_hazard"]]
    right = [r["id"] for r in labelled if r.get(hazard_key) == r["expected_hazard"]]
    missed = [r["id"] for r in labelled if r.get(hazard_key) is None]
    wrong = [r["id"] for r in labelled if r.get(hazard_key) not in (None, r["expected_hazard"])]
    return {"n": len(labelled), "right": len(right),
            "accuracy": round(len(right) / len(labelled), 4) if labelled else None,
            "missed": missed, "wrong": wrong}


def percentile(values: list[float], q: float) -> float | None:
    """Nearest-rank percentile, the same rule obs/telemetry.py uses."""
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(int(q * len(ordered)), len(ordered) - 1)], 1)


# --- citation-support pairs (needs the corpus chunks, no model) ---------------------

_PREMISE = re.compile(
    r"^Given that the IPCC says (?P<premise>.+?), (?:how|what|why|is|are|should|does|do|can|will)\b",
    re.IGNORECASE)
_CID_NOTE = re.compile(r"Parsed direction: (?P<direction>\w+)")
_CID_CHUNK = re.compile(r"Source chunk (?P<chunk>\S+?#p\d+#\d+)")
_CID_QUESTION = re.compile(
    r"^How is (?P<hazard>.+?) projected to change in (?P<region>.+?) under AR6\?$")
_FLIP = {"increase": "decrease", "decrease": "increase"}


def _norm(text: str) -> str:
    table = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'", "­": ""})
    return re.sub(r"\s+", " ", text.translate(table)).strip().lower()


def _gold_chunk(q: EvalQuestion, chunks: list) -> Any | None:
    """The chunk on a gold page that holds the start of the item's supporting quote."""
    if not q.supporting_quote:
        return None
    needle = _norm(q.supporting_quote)[:60]
    pages = {(ref.source, ref.page) for ref in q.gold_pages}
    for chunk in chunks:
        if (chunk.source, chunk.page) in pages and needle in _norm(chunk.text):
            return chunk
    return None


def verify_pairs(chunks: list) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Mechanically labelled (claim, passage, expected) pairs from the dev set."""
    by_id = {c.chunk_id: c for c in chunks}
    questions = load_gold_set().questions
    pairs: list[dict[str, Any]] = []
    skipped = {"cid_chunk_missing": 0, "quote_not_found": 0, "premise_unparsed": 0}

    for q in questions:
        if q.slice.value != "cid-table":
            continue
        direction = _CID_NOTE.search(q.notes)
        source = _CID_CHUNK.search(q.notes)
        asked = _CID_QUESTION.match(q.question)
        chunk = by_id.get(source.group("chunk")) if source else None
        if not (direction and asked and chunk):
            skipped["cid_chunk_missing"] += 1
            continue
        hazard, region = asked.group("hazard"), asked.group("region")
        verb = direction.group("direction")
        claim = f"{hazard[0].upper()}{hazard[1:]} is projected to {verb} in {region}."
        pairs.append({"id": q.id, "kind": "cid_true", "expected": "supports",
                      "claim": claim, "chunk": chunk})
        if verb in _FLIP:
            pairs.append({"id": q.id, "kind": "cid_flipped", "expected": "contradicts",
                          "claim": claim.replace(f" to {verb} ", f" to {_FLIP[verb]} "),
                          "chunk": chunk})

    quoted: list[tuple[EvalQuestion, Any]] = []
    for q in questions:
        if q.slice.value == "cid-table" or not q.answerable:
            continue
        chunk = _gold_chunk(q, chunks)
        if chunk is None:
            skipped["quote_not_found"] += 1
            continue
        if q.slice.value == "premise_injection":
            match = _PREMISE.match(q.question)
            if not match:
                skipped["premise_unparsed"] += 1
                continue
            premise = match.group("premise")
            pairs.append({"id": q.id, "kind": "premise", "expected": "contradicts",
                          "claim": f"{premise[0].upper()}{premise[1:]}.", "chunk": chunk})
            continue
        quoted.append((q, chunk))
        pairs.append({"id": q.id, "kind": "quote", "expected": "supports",
                      "claim": q.supporting_quote[:600], "chunk": chunk})

    # Each quote against the next item, in a fixed rotation, on another page and on
    # the other side of the hazard / background-science line: a heat quote against a
    # CO2-concentration page, a sea-ice quote against a rainfall page. Two hazard
    # items are never paired, because one SPM page often assesses several hazards
    # and "says nothing" would then be a label to argue about.
    for i, (q, chunk) in enumerate(quoted):
        for step in range(1, len(quoted)):
            other_q, other_chunk = quoted[(i + step) % len(quoted)]
            crosses_line = (q.hazard is None) != (other_q.hazard is None)
            if crosses_line and (other_chunk.source, other_chunk.page) != (chunk.source, chunk.page):
                pairs.append({"id": f"{q.id}>{other_q.id}", "kind": "cross",
                              "expected": "says_nothing", "claim": q.supporting_quote[:600],
                              "chunk": other_chunk})
                break
    return pairs, skipped


def auc(positives: list[float], negatives: list[float]) -> float | None:
    """Probability a random positive outranks a random negative (ties count half)."""
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return round(wins / (len(positives) * len(negatives)), 4)


# --- the live run --------------------------------------------------------------------

class BudgetExceeded(RuntimeError):
    pass


def _jev_spend_usd() -> tuple[int, float]:
    """Input tokens, and dollars at TypeSafe's list price, for every live Jev call.

    The cap is enforced on the list price even where a provider reports a
    lower charge (the gateway's free plan reports $0): that is what the same
    run would cost the moment the plan changes.
    """
    from obs.telemetry import snapshot
    from rag.jev_client import USD_PER_MTOK_INPUT

    tokens = sum(e["tokens_in"] for e in snapshot()
                 if str(e["op"]).startswith("jev:") and not e["cached"])
    return tokens, tokens * USD_PER_MTOK_INPUT / 1e6


def _jev_reported_usd() -> float | None:
    """What the provider itself reported charging, or None if it reported nothing."""
    from obs.telemetry import snapshot

    reported = [e["cost_usd"] for e in snapshot()
                if str(e["op"]).startswith("jev:") and not e["cached"] and "cost_usd" in e]
    return round(sum(reported), 6) if reported else None


def _jev_tokens_all() -> int:
    """Input plus output tokens: the gateway's daily allowance counts both."""
    from obs.telemetry import snapshot

    return sum(e["tokens_in"] + e["tokens_out"] for e in snapshot()
               if str(e["op"]).startswith("jev:") and not e["cached"])


_TOKEN_CAP = 600_000  # set by main(): well under the gateway's 1M tokens a day


def _check_budget(budget_usd: float) -> None:
    _, spent = _jev_spend_usd()
    if spent > budget_usd:
        raise BudgetExceeded(f"Jev spend ${spent:.4f} passed the ${budget_usd:.2f} cap; stopping")
    if _jev_tokens_all() > _TOKEN_CAP:
        raise BudgetExceeded(f"Jev tokens passed the {_TOKEN_CAP:,} cap; stopping")


def _jev_events() -> list[dict]:
    from obs.telemetry import snapshot

    return [e for e in snapshot() if str(e["op"]).startswith("jev:")]


def run_triage(rows: list[dict[str, Any]], *, budget_usd: float, order_check: bool,
               repeats: int) -> dict[str, Any]:
    """The shipped path (CRG_JEV=triage) plus Jev alone, on every row."""
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
        # Jev failing falls back to stage 1, which is the product working as
        # designed, but it is not a Jev decision, so it is scored apart.
        row["shipped_unavailable"] = (decision.jev is not None
                                      and decision.jev.decision == "unavailable")
        parsed = parse_query(row["question"])  # the front door, as a user would hit it
        row["front_door_hazard"] = parsed.hazard.value if parsed.hazard else None
        row["front_door_source"] = parsed.triage.hazard_source if parsed.triage else None
        row["front_door_unavailable"] = bool(
            parsed.triage and parsed.triage.jev_status.startswith("unavailable"))
        alone = triage(row["question"])  # Jev without stage 1 (cached where already asked)
        row["jev_alone"] = {
            "decision": alone.decision, "out_of_scope": alone.out_of_scope,
            "hazard": alone.hazard, "hazard_p": alone.hazard_probability,
            "topic": alone.topic, "refuse_p": alone.refuse_probability,
            "noul": alone.about_assessed_hazard, "status": alone.status,
        }
        row["jev_alone_unavailable"] = alone.decision == "unavailable"
        row["jev_alone_refused"] = alone.decision == "refuse"
        row["jev_alone_hazard"] = alone.hazard
        _check_budget(budget_usd)

    asked = [r for r in rows if r["silent_bucket"] and not r["jev_alone_unavailable"]]
    extra: dict[str, Any] = {}
    if order_check:
        flips, skipped = [], []
        for row in asked:
            flipped = triage(row["question"], topic_order=tuple(reversed(TOPIC_OPTIONS)),
                             use_cache=False)
            row["reversed_decision"] = flipped.decision
            if flipped.decision == "unavailable":
                skipped.append(row["id"])
            elif flipped.decision != row["jev_alone"]["decision"]:
                flips.append(row["id"])
            _check_budget(budget_usd)
        extra["order_check"] = {"n": len(asked) - len(skipped), "decision_flips": flips,
                                "skipped_unavailable": skipped}
    if repeats > 1:
        unstable, skipped = [], []
        for row in asked:
            seen = {row["jev_alone"]["decision"]}
            for _ in range(repeats - 1):
                seen.add(triage(row["question"], use_cache=False).decision)
                _check_budget(budget_usd)
            if "unavailable" in seen:
                skipped.append(row["id"])
            elif len(seen) > 1:
                unstable.append(row["id"])
        extra["repeats"] = {"runs": repeats, "n": len(asked) - len(skipped),
                            "unstable": unstable, "skipped_unavailable": skipped}
    return extra


def run_verify(pairs: list[dict[str, Any]], *, budget_usd: float) -> None:
    """Ask Jev the cookbook relation question for each pair; write the answer onto it."""
    from rag.jev_client import JevError, get_client, verify_policy
    from rag.jev_verify import PASSAGE_CHARS, RELATION_QUESTION, strength

    client = get_client()
    for pair in pairs:
        state = {"claim": pair["claim"], "passage": pair["chunk"].text[:PASSAGE_CHARS]}
        try:
            result = client.ask(state, RELATION_QUESTION, op="verify", policy=verify_policy())
        except JevError as exc:
            pair["error"] = exc.reason
            continue
        relation = result.choice("relation")
        pair["relation"] = relation.choice
        pair["supports"] = round(relation.p("supports"), 4)
        pair["contradicts"] = round(relation.p("contradicts"), 4)
        pair["strength"] = strength(pair["supports"], pair["contradicts"])
        _check_budget(budget_usd)


def verify_scores(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    by_kind: dict[str, Any] = {}
    for kind in ("cid_true", "quote", "cid_flipped", "premise", "cross"):
        group = [p for p in pairs if p["kind"] == kind and "relation" in p]
        if not group:
            continue
        by_kind[kind] = {
            "n": len(group),
            "expected": group[0]["expected"],
            "relation_matches_expected": sum(p["relation"] == p["expected"] for p in group),
            "mean_supports": round(mean(p["supports"] for p in group), 4),
            "mean_contradicts": round(mean(p["contradicts"] for p in group), 4),
            "strength": {s: sum(p["strength"] == s for p in group)
                         for s in ("strong", "moderate", "weak", "contradicted")},
        }
    positives = [p["supports"] for p in pairs
                 if p["kind"] in ("cid_true", "quote") and "supports" in p]
    negatives = [p["supports"] for p in pairs if p["kind"] == "cross" and "supports" in p]
    return {"by_kind": by_kind, "auc_supports_vs_cross": auc(positives, negatives),
            "errors": [p["id"] for p in pairs if "error" in p]}


# --- the report ------------------------------------------------------------------------

def _fmt_ids(ids: list[str]) -> str:
    return ", ".join(ids) if ids else "none"


def _pct(score: dict[str, Any]) -> str:
    if score.get("accuracy") is None:
        return "not measured"
    return f"{score['correct']}/{score['n']} ({score['accuracy']:.0%})"


def write_report(artifact: dict[str, Any], path: Path) -> None:
    """The dated markdown report, rendered from the artifact so no number is retyped."""
    a = artifact
    lines = [
        f"# Jev dev-set eval, {a['date']}",
        "",
        f"Run {a['run_utc']}. Dev set only (`evals/gold_set.json`, {a['dev_questions']} "
        f"questions, frozen hash in `{a['dev_sha256_file']}`: {a['dev_sha256_ok']}). "
        "The held-out set was not read. No Gemini call was made.",
        "",
        f"- Jev status: **{a['jev']['status']}**, through provider "
        f"`{a['jev'].get('provider', 'typesafe')}`",
        f"- Model requested: `{a['jev']['model_requested']}`; models that answered: "
        f"{', '.join(a['jev']['models_answered']) or 'none'}",
        f"- Live Jev calls: {a['jev']['live_calls']}, failures: {a['jev']['failures']}, "
        f"retries: {a['jev']['retries']}, input tokens: {a['jev']['input_tokens']}",
        f"- Spend: ${a['jev']['spend_usd']:.6f} at TypeSafe's list price (cap "
        f"${a['budget_usd']:.2f}); reported by the provider: "
        + (f"${a['jev']['reported_spend_usd']:.6f}"
           if a["jev"].get("reported_spend_usd") is not None else "nothing reported"),
        "- Latency per live call: " + (
            f"p50 {a['jev']['latency_ms_p50']} ms, p95 {a['jev']['latency_ms_p95']} ms "
            f"(first try only: p50 {a['jev'].get('first_try_latency_ms_p50')} ms, p95 "
            f"{a['jev'].get('first_try_latency_ms_p95')} ms; "
            f"{a['jev'].get('calls_over_1000_ms', 0)} calls over 1,000 ms)"
            if a["jev"]["latency_ms_p50"] is not None else "not measured"),
        f"- Requests paced {a['jev'].get('pacing_s_between_requests', 0):g} s apart; "
        "pacing waits are not counted as latency",
        f"- Decision thresholds (fixed before the run): refuse at summed refusal "
        f"probability >= {a['thresholds']['refuse_p']} with the companion noul <= "
        f"{a['thresholds']['noul_no']}; hint at >= {a['thresholds']['hint_p']} with the noul "
        f">= {a['thresholds']['noul_yes']}",
        "",
        "## Scope: refuse at the guard or not",
        "",
        "| set | n | stage 1 today | stage 1 + Jev (shipped with CRG_JEV=triage) | Jev alone |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("dev", "probe"):
        s = a["scope"][name]
        lines.append(f"| {name} | {s['stage1']['n']} | {_pct(s['stage1'])} | "
                     f"{_pct(s['shipped'])} | {_pct(s['jev_alone'])} |")
    lines += ["", "False refusals and false accepts, by item:", ""]
    for name in ("dev", "probe"):
        for arm in ("stage1", "shipped", "jev_alone"):
            s = a["scope"][name][arm]
            if s.get("accuracy") is None:
                continue
            lines.append(f"- {name}, {arm}: false refusals {_fmt_ids(s['false_refusals'])}; "
                         f"false accepts {_fmt_ids(s['false_accepts'])}")
            if s.get("excluded_unavailable"):
                lines.append(f"- {name}, {arm}: left out because Jev was unavailable for them: "
                             f"{_fmt_ids(s['excluded_unavailable'])}")
    lines += [
        "",
        "A false accept is a question that should have been refused at scope reaching the "
        "answering model, which refuses it itself today. It is a missed instant refusal, "
        "not a false answer.",
        "",
        "## Hazard named by the front door",
        "",
        "| set | items with a hazard | lexical parser today | front door with Jev | Jev alone |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in ("dev", "probe"):
        h = a["hazard"][name]
        cells = []
        for arm in ("lexical", "front_door", "jev_alone"):
            score = h[arm]
            cells.append("not measured" if score["accuracy"] is None
                         else f"{score['right']}/{score['n']} ({score['accuracy']:.0%})")
        lines.append(f"| {name} | {h['lexical']['n']} | " + " | ".join(cells) + " |")
    lines += ["", "Missed or wrong, by item:", ""]
    for name in ("dev", "probe"):
        for arm in ("lexical", "front_door", "jev_alone"):
            score = a["hazard"][name][arm]
            if score["accuracy"] is None:
                continue
            lines.append(f"- {name}, {arm}: missed {_fmt_ids(score['missed'])}; "
                         f"wrong {_fmt_ids(score['wrong'])}")
            if score.get("excluded_unavailable"):
                lines.append(f"- {name}, {arm}: left out because Jev was unavailable for them: "
                             f"{_fmt_ids(score['excluded_unavailable'])}")
    if a.get("order_check"):
        oc = a["order_check"]
        lines += ["", "## Robustness", "",
                  f"- Topic options reversed: {len(oc['decision_flips'])} of {oc['n']} "
                  f"decisions changed ({_fmt_ids(oc['decision_flips'])}); left out, Jev "
                  f"unavailable: {_fmt_ids(oc['skipped_unavailable'])}."]
    if a.get("repeats"):
        rp = a["repeats"]
        lines.append(f"- {rp['runs']} uncached runs per question: {len(rp['unstable'])} of "
                     f"{rp['n']} decisions were not the same every time "
                     f"({_fmt_ids(rp['unstable'])}); left out, Jev unavailable: "
                     f"{_fmt_ids(rp['skipped_unavailable'])}.")
    if a.get("verify"):
        v = a["verify"]
        lines += ["", "## Citation support check", ""]
        if v["by_kind"]:
            lines += ["| pair kind | n | expected | Jev agrees | mean P(supports) | "
                      "mean P(contradicts) | strong / moderate / weak / contradicted |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for kind, s in v["by_kind"].items():
                st = s["strength"]
                lines.append(f"| {kind} | {s['n']} | {s['expected']} | "
                             f"{s['relation_matches_expected']}/{s['n']} | "
                             f"{s['mean_supports']} | {s['mean_contradicts']} | "
                             f"{st['strong']} / {st['moderate']} / {st['weak']} / "
                             f"{st['contradicted']} |")
            lines += ["", f"- AUC of P(supports), supporting pairs against cross pairs: "
                          f"{v['auc_supports_vs_cross']}"]
        else:
            built: dict[str, int] = {}
            for pair in v["pairs"]:
                built[pair["kind"]] = built.get(pair["kind"], 0) + 1
            lines.append(f"Not measured: {len(v['pairs'])} pairs were built "
                         f"({', '.join(f'{k} {n}' for k, n in built.items())}) and none was sent.")
        lines.append(f"- Pairs skipped while building: {v['skipped']}; failed calls: "
                     f"{_fmt_ids(v['errors'])}")
    lines += ["", "## Silent-bucket items, as Jev read them", "",
              "The questions stage 1 cannot see, the only ones the shipped path sends to Jev.", "",
              "| id | slice | should refuse | Jev decision | refuse p | noul | hazard (p) |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in a["rows"]:
        if not row["silent_bucket"]:
            continue
        j = row.get("jev_alone")
        if j is None:
            lines.append(f"| {row['id']} | {row['slice']} | {row['should_refuse']} | "
                         "not run | | | |")
            continue
        lines.append(f"| {row['id']} | {row['slice']} | {row['should_refuse']} | "
                     f"{j['decision']} | {j['refuse_p']} | {j['noul']} | "
                     f"{j['hazard']} ({j['hazard_p']}) |")
    lines += ["", "## Caveats", ""] + [f"- {c}" for c in a["caveats"]] + [""]
    path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jev dev-set eval: triage and citation support.")
    parser.add_argument("--task", choices=("triage", "verify", "all"), default="triage")
    parser.add_argument("--dry-run", action="store_true",
                        help="no Jev call: baseline numbers and request sizes only")
    parser.add_argument("--budget-usd", type=float, default=2.0,
                        help="hard stop on measured Jev spend (default 2.00)")
    parser.add_argument("--order-check", action="store_true",
                        help="re-ask the silent bucket with the topic options reversed")
    parser.add_argument("--repeats", type=int, default=1,
                        help="uncached runs per silent-bucket question (stability)")
    parser.add_argument("--max-rpm", type=float, default=8.0,
                        help="client-side cap on Jev requests a minute (default 8: the "
                             "gateway's free plan allows 10)")
    parser.add_argument("--max-tokens", type=int, default=600_000,
                        help="hard stop on input plus output tokens (default 600,000: the "
                             "gateway's free plan allows 1M a day)")
    return parser.parse_args(argv)


# Every environment variable a run changes, restored when it ends, so calling
# main() in-process (the tests do) leaves no pacing, budget or flag behind.
_SCOPED_ENV = ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
               "GOOGLE_API_KEY", "CRG_SCOPE_STAGE2", "CRG_JEV", "CRG_JEV_MIN_INTERVAL_S",
               "CRG_JEV_TRIAGE_BUDGET_S", "CRG_JEV_VERIFY_BUDGET_S")


def main(argv: list[str] | None = None) -> int:
    saved = {name: os.environ.get(name) for name in _SCOPED_ENV}
    try:
        return _run(_parse_args(argv))
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _run(args: argparse.Namespace) -> int:
    global _TOKEN_CAP
    from obs.log import configure

    configure()
    for name in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY",
                 "GOOGLE_API_KEY"):
        os.environ.pop(name, None)  # nothing here may reach Gemini or Vertex
    os.environ["CRG_SCOPE_STAGE2"] = "off"  # today's shipped guard is the baseline
    _TOKEN_CAP = args.max_tokens
    # Pace every request to the plan, and give a rate-limited call room to back
    # off: a 429 with no Retry-After waits one paced slot, which the product's
    # few-second budgets would not allow. Each attempt keeps the provider's own
    # read timeout, so a slow reply is still a slow reply.
    if args.max_rpm > 0:
        os.environ["CRG_JEV_MIN_INTERVAL_S"] = f"{60.0 / args.max_rpm:g}"
    os.environ.setdefault("CRG_JEV_TRIAGE_BUDGET_S", "40")
    os.environ.setdefault("CRG_JEV_VERIFY_BUDGET_S", "40")

    from rag.jev_client import JevError, get_client, get_provider, jev_mode
    from rag.jev_triage import HINT_P, NOUL_NO, NOUL_YES, REFUSE_P, build_questions

    gold = load_gold_set()
    dev_sha = FROZEN_HASH_PATH.read_text(encoding="utf-8").strip()
    rows = dev_rows() + probe_rows()
    for row in rows:
        lexical_baseline(row)
        row["stage1_hazard"] = row["lexical_hazard"]
    started = time.perf_counter()

    jev_status = "not run (dry run)" if args.dry_run else "ok"
    models_listed: list[str] = []
    extra: dict[str, Any] = {}
    pairs: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    provider_name, model_requested, models_total = "not resolved", "not resolved", 0
    try:
        provider = get_provider()
        provider_name, model_requested = provider.name, provider.default_model
    except JevError as exc:
        jev_status = f"unavailable: {exc.reason}"
    if not args.dry_run and jev_status == "ok":
        try:
            client = get_client()
            model_requested = client.model
            listed = client.list_models()
            models_total = len(listed)
            # a gateway lists hundreds of models; only Jev's belong in the record
            models_listed = [str(m["name"]) for m in listed if "jev" in str(m["name"]).lower()]
        except JevError as exc:
            jev_status = f"unavailable: {exc.reason}"
    if args.task in ("verify", "all"):
        try:
            from rag.corpus import CorpusError, load_corpus_chunks

            pairs, skipped = verify_pairs(list(load_corpus_chunks()))
        except CorpusError as exc:
            skipped = {"corpus_missing": 1}
            print(f"verify pairs skipped: {exc}", file=sys.stderr)

    try:
        if jev_status == "ok" and args.task in ("triage", "all"):
            extra = run_triage(rows, budget_usd=args.budget_usd, order_check=args.order_check,
                               repeats=max(1, args.repeats))
        if jev_status == "ok" and pairs:
            run_verify(pairs, budget_usd=args.budget_usd)
    except BudgetExceeded as exc:
        jev_status = f"stopped: {exc}"
    finally:
        os.environ.pop("CRG_JEV", None)

    events = _jev_events()
    live = [e for e in events if not e["cached"]]
    tokens, spend = _jev_spend_usd()
    if jev_status == "ok" and live and all(not e["ok"] for e in live):
        jev_status = f"unavailable: {live[0].get('detail', 'every call failed')}"
    # Each task is "measured" only if it ran and Jev answered it at least once;
    # rows Jev could not answer are left out of its arms below, and listed.
    triage_measured = (jev_status == "ok" and args.task in ("triage", "all")
                       and any(e["ok"] and e["op"] == "jev:triage" for e in live))
    verify_measured = jev_status == "ok" and any(e["ok"] and e["op"] == "jev:verify"
                                                 for e in live)
    measured = triage_measured or verify_measured

    def scored(subset: list[dict[str, Any]], scorer, key: str, gone: str) -> dict[str, Any]:
        kept = [r for r in subset if not r.get(gone)]
        return {**scorer(kept, key),
                "excluded_unavailable": [r["id"] for r in subset if r.get(gone)]}

    def arms(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"stage1": scope_scores(subset, "stage1_refused")}
        if triage_measured:
            out["shipped"] = scored(subset, scope_scores, "shipped_refused", "shipped_unavailable")
            out["jev_alone"] = scored(subset, scope_scores, "jev_alone_refused",
                                      "jev_alone_unavailable")
        else:
            out["shipped"] = out["jev_alone"] = {"n": len(subset), "accuracy": None}
        return out

    def hazards(subset: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"lexical": hazard_scores(subset, "lexical_hazard")}
        for arm, key, gone in (("front_door", "front_door_hazard", "front_door_unavailable"),
                               ("jev_alone", "jev_alone_hazard", "jev_alone_unavailable")):
            out[arm] = (scored(subset, hazard_scores, key, gone) if triage_measured
                        else {"n": out["lexical"]["n"], "accuracy": None})
        return out

    dev = [r for r in rows if r["set"] == "dev"]
    probes = [r for r in rows if r["set"] == "probe"]
    from rag.jev_client import encode_request

    # the exact bytes the client puts on the wire for a 100-character question
    one_request = len(encode_request(model_requested, {"question": "x" * 100},
                                     build_questions()))
    ok_calls = [e for e in live if e["ok"]]
    first_try = [e["latency_ms"] for e in ok_calls if e["retries"] == 0]
    today = datetime.now(timezone.utc)
    artifact: dict[str, Any] = {
        "date": f"{today:%Y-%m-%d}",
        "run_utc": today.isoformat(timespec="seconds"),
        "eval_set": "dev",
        "dev_questions": len(gold.questions),
        "dev_sha256_file": "evals/gold_set.sha256",
        "dev_sha256_ok": gold.content_hash() == dev_sha,
        "probe_questions": len(probes),
        "budget_usd": args.budget_usd,
        "wall_s": round(time.perf_counter() - started, 1),
        "jev": {
            "status": jev_status,
            "triage_measured": triage_measured,
            "verify_measured": verify_measured,
            "flag_during_run": "triage" if triage_measured else jev_mode(),
            "provider": provider_name,
            "model_requested": model_requested,
            "models_listed": models_listed,
            "models_listed_total": models_total,
            "models_answered": sorted({e["model"] for e in live if e["ok"]}),
            "live_calls": len(live),
            "cached_calls": len(events) - len(live),
            "failures": sum(1 for e in live if not e["ok"]),
            "retries": sum(e["retries"] for e in live),
            "input_tokens": tokens,
            "input_plus_output_tokens": _jev_tokens_all(),
            "spend_usd": round(spend, 6),  # at the list price, the figure the cap uses
            "reported_spend_usd": _jev_reported_usd(),  # what the provider says it charged
            "pacing_s_between_requests": float(os.environ.get("CRG_JEV_MIN_INTERVAL_S", 0) or 0),
            "latency_ms_p50": percentile([e["latency_ms"] for e in ok_calls], 0.50),
            "latency_ms_p95": percentile([e["latency_ms"] for e in ok_calls], 0.95),
            "first_try_latency_ms_p50": percentile(first_try, 0.50),
            "first_try_latency_ms_p95": percentile(first_try, 0.95),
            "calls_over_1000_ms": sum(1 for e in ok_calls if e["latency_ms"] > 1000),
            "triage_request_bytes_for_a_100_char_question": one_request,
            "failure_details": sorted({e.get("detail", "") for e in live if not e["ok"]}),
        },
        "thresholds": {"refuse_p": REFUSE_P, "hint_p": HINT_P, "noul_yes": NOUL_YES,
                       "noul_no": NOUL_NO},
        "silent_bucket": {"dev": [r["id"] for r in dev if r["silent_bucket"]],
                          "probe": [r["id"] for r in probes if r["silent_bucket"]]},
        "scope": {"dev": arms(dev), "probe": arms(probes)},
        "hazard": {"dev": hazards(dev), "probe": hazards(probes)},
        **extra,
        "rows": rows,
        "caveats": [
            "Thresholds and question wording were fixed before this run, from TypeSafe's "
            "documented starting points; the dev set steers development by protocol, so any "
            "later retune on these numbers makes them optimistic.",
            "The dev set is IPCC-knowledge questions, not front-door questions with a place "
            "and a horizon; the 14 paraphrase probes are closer to front-door wording but "
            "were authored, not sampled from users.",
            "Counts are small: 2 dev items should be refused inside the silent bucket, and "
            "7 probes. A difference of one item is not evidence of a difference.",
            "No prompt-injection probes were sent to Jev: TypeSafe's terms bar testing "
            "their systems, and the dev set has no labelled injection items.",
        ],
    }
    if pairs:
        artifact["verify"] = {**verify_scores(pairs), "skipped": skipped,
                              "pairs": [{k: v for k, v in p.items() if k != "chunk"}
                                        | {"chunk_id": p["chunk"].chunk_id} for p in pairs]}
    elif skipped:
        artifact["verify"] = None
        artifact["verify_skipped"] = skipped

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"jev-dev-{artifact['date']}.json"
    md_path = RESULTS_DIR / f"jev-dev-{artifact['date']}.md"
    json_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    if artifact.get("verify") is None:
        artifact.pop("verify", None)
    write_report(artifact, md_path)
    print(f"jev status: {jev_status} | live calls {len(live)} | spend ${spend:.6f}")
    print(f"artifacts: {json_path} and {md_path}")
    return 0 if (measured or args.dry_run) else 2


if __name__ == "__main__":
    raise SystemExit(main())
