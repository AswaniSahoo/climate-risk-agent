"""Calibrate the semantic scope guard's thresholds — and show the work.

Two thresholds decide everything in `rag/scope_semantic.py`: SIM_FLOOR (how
close the nearest anchor must be before any verdict is allowed) and MARGIN (how
far it must beat the other class). Numbers chosen by feel are not evidence, so
they are chosen HERE, against three labelled probes, and the table is written to
evals/results/scope-stage2-calibration.json.

The probes:
- `dev_no_signal`  the 13 DEV-set questions the lexical stage is blind to, with
  their gold expected behaviour. A false refusal here is a real regression.
- `paraphrase_in`  in-scope questions written to dodge the lexical vocabulary
  ("storms getting stronger", "the hottest days"). Stage 2 exists for these.
- `paraphrase_out` out-of-scope questions that likewise name no known term.

The HELD-OUT test set is not read here. Run:
  uv run python -m evals.scope_stage2_calibration
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rag.scope import has_supported_signal, out_of_scope_hazard
from rag.scope_semantic import load_anchors

# (question, expected) — expected is "in" (must not be refused) or "out".
PARAPHRASE_IN: list[tuple[str, str | None]] = [
    ("Are storms getting stronger?", "wind"),
    ("Is it getting windier along the coast?", "wind"),
    ("Am I more likely to lose roof tiles in a gale these days?", "wind"),
    ("Are the hottest days of the year getting hotter in Delhi?", "heatwave"),
    ("How much worse will summer scorchers get by 2050?", "heatwave"),
    ("Are downpours becoming more severe?", "extreme precipitation"),
    ("Will cloudbursts hit Mumbai more often?", "extreme precipitation"),
]

PARAPHRASE_OUT: list[tuple[str, str]] = [
    ("Is the smoke season getting longer where I live?", "wildfire smoke"),
    ("Will the sea come further inland by 2100?", "coastal flooding / sea level"),
    ("Is the air getting harder to breathe in Delhi?", "air quality"),
    ("How much of our GDP will we lose to warming?", "climate policy and economics"),
    ("What has India promised at the climate talks?", "climate policy and economics"),
    ("Should I worry about tremors around here?", "earthquake / seismic hazard"),
    ("What is a good recipe for dal tadka?", "unrelated to climate risk"),
]

GRID_FLOORS = [0.50, 0.55, 0.60, 0.65, 0.70]
GRID_MARGINS = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10]


def _no_signal_dev_questions() -> list[dict]:
    from evals.gold_set import load_gold_set

    rows = []
    for q in load_gold_set().questions:
        if out_of_scope_hazard(q.question) is None and not has_supported_signal(q.question):
            rows.append({"id": q.id, "question": q.question, "slice": q.slice.value,
                         "expected": q.expected_behavior.value})
    return rows


def _unit(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1.0, norms)


def main() -> None:
    from obs.log import configure

    configure()
    from rag.embed import DiskVectorCache, cached_embed_texts
    from rag.scope_semantic import CACHE_DIR

    anchors = load_anchors()
    cache = DiskVectorCache(CACHE_DIR)
    anchor_matrix = _unit(
        cached_embed_texts([a.text for a in anchors], task_type="RETRIEVAL_DOCUMENT", cache=cache)
    )

    dev = _no_signal_dev_questions()
    probes = (
        [{"probe": "dev_no_signal", **row} for row in dev]
        + [{"probe": "paraphrase_in", "id": f"PIN-{i:02d}", "question": q,
            "expected": "answer", "hazard": h} for i, (q, h) in enumerate(PARAPHRASE_IN, 1)]
        + [{"probe": "paraphrase_out", "id": f"POUT-{i:02d}", "question": q,
            "expected": "refuse", "topic": t} for i, (q, t) in enumerate(PARAPHRASE_OUT, 1)]
    )
    query_matrix = _unit(
        cached_embed_texts([p["question"] for p in probes], task_type="RETRIEVAL_QUERY",
                           cache=cache)
    )

    for probe, vector in zip(probes, query_matrix):
        sims = anchor_matrix @ vector
        scored = list(zip((float(s) for s in sims), anchors))
        in_sim, in_anchor = max(
            (p for p in scored if p[1].label == "in_scope"), key=lambda p: p[0]
        )
        out_sim, out_anchor = max(
            (p for p in scored if p[1].label == "out_of_scope"), key=lambda p: p[0]
        )
        probe.update({
            "best_in_sim": round(in_sim, 4),
            "best_in_hazard": in_anchor.hazard,
            "best_in_text": in_anchor.text,
            "best_out_sim": round(out_sim, 4),
            "best_out_topic": out_anchor.topic,
            "best_out_text": out_anchor.text,
        })

    # Grid search. A "cost" is a refusal on something that must be answered;
    # a "win" is a refusal on something that must be refused or a correct hint.
    grid = []
    for floor in GRID_FLOORS:
        for margin in GRID_MARGINS:
            false_refuse, true_refuse, good_hint, bad_hint, deferred = 0, 0, 0, 0, 0
            for p in probes:
                refuse = p["best_out_sim"] >= floor and p["best_out_sim"] - p["best_in_sim"] >= margin
                hint = (not refuse and p["best_in_sim"] >= floor
                        and p["best_in_sim"] - p["best_out_sim"] >= margin
                        and p["best_in_hazard"])
                if refuse:
                    if p["expected"] == "refuse":
                        true_refuse += 1
                    else:
                        false_refuse += 1
                elif hint:
                    if p.get("hazard") == p["best_in_hazard"]:
                        good_hint += 1
                    elif p["expected"] == "refuse":
                        bad_hint += 1
                else:
                    deferred += 1
            grid.append({"floor": floor, "margin": margin, "false_refuse": false_refuse,
                         "true_refuse": true_refuse, "correct_hint": good_hint,
                         "hint_on_out_of_scope": bad_hint, "deferred": deferred})

    print(f"{'floor':>6} {'margin':>7} {'false_refuse':>13} {'true_refuse':>12} "
          f"{'correct_hint':>13} {'bad_hint':>9} {'deferred':>9}")
    for row in grid:
        print(f"{row['floor']:6.2f} {row['margin']:7.2f} {row['false_refuse']:13d} "
              f"{row['true_refuse']:12d} {row['correct_hint']:13d} "
              f"{row['hint_on_out_of_scope']:9d} {row['deferred']:9d}")

    print("\n-- per-probe nearest anchors --")
    for p in probes:
        print(f"{p['id']:9s} {p['probe']:15s} exp={p['expected']:6s} "
              f"in={p['best_in_hazard'] or 'background':22s} {p['best_in_sim']:.3f} | "
              f"out={p['best_out_topic']:30s} {p['best_out_sim']:.3f}")

    out_dir = Path("evals/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scope-stage2-calibration.json"
    out_path.write_text(
        json.dumps({"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "n_anchors": len(anchors), "probes": probes, "grid": grid},
                   indent=2),
        encoding="utf-8",
    )
    print(f"\nartifact written: {out_path}")


if __name__ == "__main__":
    main()
