"""Precompute the IPCC grounding for (hazard, AR6 area) pairs into a committed seed file.

WHY. The research step asks `_ipcc_question(hazard, _ipcc_area(state))`, and the
area is the AR6 region, never the city (agent/graph.py). So the retrieval, the
cited answer and the Ch.12 projection are the same for every city in a region.
In production there are no Upstash credentials (env var names read from the
live service on 2026-11-11) and min-instances is 0, so the grounding cache is a
per-instance disk that dies at every scale to zero; shipping the entries in the
image is what lets a cold visitor skip the retrieval and the model call.

WHAT. For each pair it runs the graph's OWN research() and project() nodes on
one point inside the region, with the committed seed turned off and a scratch
disk as the live tier, then copies the two entries those nodes wrote
(namespaces `grounding` and `projection`) into tools/cache_seed/ipcc.json. What
ships is therefore exactly what a live run would have cached: same keys (corpus,
ranking code, prompt and model versions), same rules (no abstention, no
BM25-only ranking).

COST. At most MAX_CALLS_PER_PAIR model calls a pair: the research question
embedding, one answer, the Ch.12 question embedding (fewer when a query vector
is already in data/cache/embeddings). The run stops before a pair could take it
past --max-calls. The full set is 3 hazards x 46 regions = 138 pairs.

Run (repo root, with the corpus and the embedding cache in data/, and model auth):
  uv run python -m scripts.precompute_grounding --plan
  uv run python -m scripts.precompute_grounding --presets          # the six UI presets: 6 pairs
  uv run python -m scripts.precompute_grounding --areas WCE,SAS --hazards heatwave
  uv run python -m scripts.precompute_grounding                    # all 138 pairs
  uv run python -m scripts.precompute_grounding --check --presets
Any change to the corpus, the ranking code, rag/answer.py, rag/cid.py or the
model moves every key: run it again after such a change.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from agent.contracts import Hazard
from tools.ar6_regions import AR6Region, _land_regions, region_for
from tools.cache_backend import SEED_DIR, write_seed_file

_log = logging.getLogger("scripts.precompute_grounding")

SEED_FILE = SEED_DIR / "ipcc.json"
SCHEMA_VERSION = 1
MAX_CALLS_PER_PAIR = 3
CITIES_FILE = Path(__file__).resolve().parent / "precompute_cities.json"


@dataclass(frozen=True)
class Pair:
    hazard: Hazard
    region: AR6Region
    latitude: float
    longitude: float

    @property
    def label(self) -> str:
        return f"{self.hazard.value}/{self.region.acronym}"


def _hazards() -> list[Hazard]:
    from agent.graph import _ANSWERABLE

    return [h for h in Hazard if h in _ANSWERABLE]


def region_points() -> list[tuple[AR6Region, float, float]]:
    """One point inside each AR6 land region, checked to map back to that region."""
    points = []
    for acronym, _name, polygon in _land_regions():
        for candidate in (polygon.representative_point(), polygon.centroid):
            region = region_for(candidate.y, candidate.x)
            if region is not None and region.acronym == acronym:
                points.append((region, candidate.y, candidate.x))
                break
        else:
            _log.warning("no point found that maps back to %s; region skipped", acronym)
    return points


def all_pairs() -> list[Pair]:
    return [Pair(h, region, lat, lon) for region, lat, lon in region_points() for h in _hazards()]


def preset_pairs(cities_file: Path = CITIES_FILE) -> list[Pair]:
    """The (hazard, area) pairs the UI preset buttons ask for, at the preset points."""
    payload = json.loads(Path(cities_file).read_text(encoding="utf-8"))
    pairs: dict[tuple[str, str], Pair] = {}
    for city in payload["cities"]:
        if not city.get("preset"):
            continue
        region = region_for(city["latitude"], city["longitude"])
        if region is None:
            continue
        for hazard in _hazards():
            pairs.setdefault((hazard.value, region.acronym),
                             Pair(hazard, region, city["latitude"], city["longitude"]))
    return list(pairs.values())


def _keys(pair: Pair) -> tuple[str, str]:
    """(grounding key, projection key), namespaced exactly as the live nodes store them."""
    from agent import graph

    grounding = graph._grounding_cache()
    question = graph._ipcc_question(pair.hazard, pair.region.label)
    g_raw = grounding.key(hazard=pair.hazard.value, area=pair.region.label, question=question,
                          top_k=graph._IPCC_TOP_K)
    projection = graph._projection_cache()
    p_raw = projection.key(pair.region, pair.hazard)
    return grounding.json_cache.key(g_raw), projection.json_cache.key(p_raw)


def read_rows(path: Path = SEED_FILE) -> list[dict]:
    if not Path(path).exists():
        return []
    return list(json.loads(Path(path).read_text(encoding="utf-8"))["entries"])


def save_rows(rows: Sequence[dict], path: Path = SEED_FILE) -> None:
    from rag.answer_cache import corpus_version, prompt_version, retrieval_version
    from rag.gemini_client import GENERATE_MODEL

    header = {
        "schema_version": SCHEMA_VERSION,
        "namespaces": ["grounding", "projection"],
        "provenance": {
            "what": "the research and project nodes' results per (hazard, AR6 area), computed "
                    "by running those nodes (agent/graph.py) once per pair",
            "corpus": "IPCC AR6 WGI SPM, Chapter 11 and Chapter 12 (rag/corpus.py)",
            "generated_by": "scripts/precompute_grounding.py",
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "version": {"corpus": corpus_version(), "retrieval": retrieval_version(),
                    "prompt": prompt_version(), "model": GENERATE_MODEL},
    }
    write_seed_file(path, header=header, entries=list(rows))


def _model_calls() -> int:
    from obs.telemetry import snapshot

    return sum(1 for e in snapshot() if e["op"] in ("embed", "generate"))


@dataclass
class Summary:
    written: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    not_cached: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    model_calls: int = 0


def run(pairs: Sequence[Pair], *, seed_path: Path = SEED_FILE, max_calls: int = 30) -> Summary:
    """Ground every pair not yet shipped, spending at most `max_calls` model calls."""
    from agent import graph
    from agent.graph import AgentState

    rows = {row["key"]: row for row in read_rows(seed_path)}
    summary = Summary()
    start = _model_calls()
    for pair in pairs:
        g_key, p_key = _keys(pair)
        if g_key in rows and p_key in rows:
            summary.skipped.append(pair.label)
            continue
        if summary.model_calls + MAX_CALLS_PER_PAIR > max_calls:
            summary.deferred.append(pair.label)
            continue
        state: AgentState = {"location": pair.region.label, "latitude": pair.latitude,
                             "longitude": pair.longitude, "hazard": pair.hazard, "horizon_days": 7}
        graph.research(state)
        graph.project(state)
        summary.model_calls = _model_calls() - start
        common = {"hazard": pair.hazard.value, "region": pair.region.acronym}
        grounded = graph._grounding_cache().json_cache.backend.get(g_key)
        projected = graph._projection_cache().json_cache.backend.get(p_key)
        if projected is not None:
            rows[p_key] = {"key": p_key, "namespace": "projection", **common,
                           "value": json.loads(projected)}
        if grounded is not None:
            rows[g_key] = {"key": g_key, "namespace": "grounding", **common,
                           "value": json.loads(grounded)}
            summary.written.append(pair.label)
        else:
            summary.not_cached.append(
                f"{pair.label}: no grounding entry (abstained, BM25-only or failed)")
        save_rows(list(rows.values()), seed_path)  # after every pair: an interrupted run resumes
    return summary


def check(pairs: Sequence[Pair], seed_path: Path = SEED_FILE) -> dict:
    """How many of `pairs` the file answers for the CURRENT versions, and how many rows are stale."""
    keys = {row["key"] for row in read_rows(seed_path)}
    wanted = {_keys(pair)[0]: pair for pair in pairs}
    current_anywhere = {k for pair in [*all_pairs(), *pairs] for k in _keys(pair)}
    return {
        "pairs": len(wanted),
        "current": sum(1 for k in wanted if k in keys),
        "missing": [pair.label for k, pair in wanted.items() if k not in keys],
        "stale": len(keys - current_anywhere),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Precompute the IPCC grounding per (hazard, area).")
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument("--presets", action="store_true", help="only the six UI presets' pairs")
    scope.add_argument("--areas", default=None, help="comma-separated AR6 acronyms, e.g. WCE,SAS")
    parser.add_argument("--hazards", default=None, help="comma-separated hazards (default: all)")
    parser.add_argument("--max-calls", type=int, default=30, help="model calls this run may make")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="list the pairs and the worst-case cost")
    mode.add_argument("--check", action="store_true", help="exit 1 if a listed pair is missing")
    args = parser.parse_args(argv)

    # Before anything touches the cache: compute live (seed off) into a scratch
    # disk tier, so what gets copied into the seed is what THIS run computed.
    os.environ["CRG_CACHE_SEED_DIR"] = ""
    os.environ.setdefault("CRG_CACHE_DIR", tempfile.mkdtemp(prefix="crg-precompute-grounding-"))
    from obs.log import configure

    configure()

    pairs = preset_pairs() if args.presets else all_pairs()
    if args.areas:
        wanted = {a.strip().upper() for a in args.areas.split(",") if a.strip()}
        pairs = [p for p in pairs if p.region.acronym in wanted]
    if args.hazards:
        hazards = {Hazard(h.strip()) for h in args.hazards.split(",") if h.strip()}
        pairs = [p for p in pairs if p.hazard in hazards]

    if args.plan:
        for pair in pairs:
            print(f"  {pair.label:<22} {pair.region.label} at ({pair.latitude:.2f}, {pair.longitude:.2f})")
        print(f"{len(pairs)} pairs, at most {MAX_CALLS_PER_PAIR * len(pairs)} model calls")
        return 0
    if args.check:
        result = check(pairs)
        print(json.dumps(result, indent=1))
        return 1 if result["missing"] else 0

    summary = run(pairs, max_calls=args.max_calls)
    print(f"written {len(summary.written)} | already shipped {len(summary.skipped)} | "
          f"not cached {len(summary.not_cached)} | deferred {len(summary.deferred)} | "
          f"model calls {summary.model_calls}")
    for line in summary.not_cached:
        print(f"  NOT CACHED {line}")
    if summary.deferred:
        print(f"  deferred by --max-calls: {', '.join(summary.deferred)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
