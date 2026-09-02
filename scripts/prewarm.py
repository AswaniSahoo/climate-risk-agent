"""Fill the shared cache before anyone asks, so no user pays for a cold fit.

WHY. A cold hazard statistic costs ~42 s of Open-Meteo Archive fetch plus 12.9 s
of GEV fit + trend test + bootstrap (measured). Cache v2 makes that cost
survivable across replicas and redeploys, but SOMEONE still pays it first — and
on a portfolio app that someone is whoever opens the demo. This script pays it
instead, from a laptop or a release step, for the cities most likely to be
typed. Run it once after a deploy that changed tools/hazard_stats.py or
tools/gev_trend.py (the code fingerprint retires the old entries, by design).

It is deliberately observable, per the house rule that no long-running script
waits silently: a tqdm bar with the current city, per-item timing, a LOUD
warning on any failure, and a summary table of hits / misses / errors /
seconds. Whether an item was a hit or a miss is read from the telemetry ring,
not guessed from the clock.

Run:  uv run python -m scripts.prewarm --cities scripts/prewarm_cities.json \
          --hazards heatwave,extreme_precip,wind
      uv run python -m scripts.prewarm --dry-run          # plan only, no work
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field
from tqdm import tqdm

from agent.contracts import Hazard
from tools.climatology import ClimatologyError, climatology_hazard_stat

_log = logging.getLogger("scripts.prewarm")

DEFAULT_CITIES = Path("scripts/prewarm_cities.json")


class City(BaseModel):
    """One prewarm target. Coordinates are approximate city centres (2 dp)."""

    name: str
    country: str = ""
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)


@dataclass
class Result:
    """What one (city, hazard) fit cost and where the answer came from."""

    city: str
    hazard: Hazard
    seconds: float
    status: str  # "miss" (fitted now) | "hit" (already cached) | "error"
    detail: str = ""


def load_cities(path: Path) -> list[City]:
    """Parse the city file. A malformed entry fails HERE, not 20 fits later."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [City.model_validate(entry) for entry in payload["cities"]]


def parse_hazards(text: str) -> list[Hazard]:
    """"heatwave,wind" -> [Hazard.HEATWAVE, Hazard.WIND]; unknown names raise."""
    names = [part.strip() for part in text.split(",") if part.strip()]
    try:
        return [Hazard(name) for name in names]
    except ValueError as exc:
        supported = ", ".join(h.value for h in Hazard)
        raise SystemExit(f"unknown hazard in {text!r} — supported: {supported}") from exc


def _cache_status_since(marker: int) -> str:
    """Did the fit come from cache? Read it off the telemetry the cache emits."""
    from obs.telemetry import snapshot

    events = [e for e in snapshot()[marker:] if e["op"] == "cache:hazard_fit"]
    if not events:
        return "miss"  # no cache event at all: the work was certainly done
    return "hit" if events[0]["cached"] else "miss"


def prewarm(
    cities: Sequence[City],
    hazards: Sequence[Hazard],
    *,
    fit: Callable[..., object] = climatology_hazard_stat,
) -> list[Result]:
    """Fit every (city, hazard) pair, filling the cache as a side effect.

    `fit` is injected so tests can exercise the loop, the timing and the
    failure path without touching the network.
    """
    from obs.telemetry import snapshot

    results: list[Result] = []
    pairs = [(city, hazard) for city in cities for hazard in hazards]
    progress = tqdm(pairs, desc="Prewarming", unit="fit")
    for city, hazard in progress:
        progress.set_postfix_str(f"{city.name} / {hazard.value}", refresh=True)
        marker = len(snapshot())
        started = time.perf_counter()
        try:
            fit(city.latitude, city.longitude, hazard)
        except (ClimatologyError, ValueError) as exc:
            elapsed = time.perf_counter() - started
            # Loud, and the run continues: one dead city must not cost the other 24.
            _log.warning("prewarm FAILED for %s / %s: %s", city.name, hazard.value, exc)
            results.append(Result(city.name, hazard, elapsed, "error", str(exc)))
            continue
        elapsed = time.perf_counter() - started
        results.append(Result(city.name, hazard, elapsed, _cache_status_since(marker)))
    progress.close()
    return results


def summarize(results: Sequence[Result]) -> dict[str, float]:
    """Counts and seconds — the numbers the summary table prints."""
    return {
        "items": len(results),
        "hits": sum(1 for r in results if r.status == "hit"),
        "misses": sum(1 for r in results if r.status == "miss"),
        "errors": sum(1 for r in results if r.status == "error"),
        "seconds": round(sum(r.seconds for r in results), 1),
    }


def _print_table(results: Sequence[Result]) -> None:
    totals = summarize(results)
    slowest = sorted(results, key=lambda r: r.seconds, reverse=True)[:5]
    print(f"\n{'city':<16}{'hazard':<16}{'status':<8}{'seconds':>8}")
    print("-" * 48)
    for r in results:
        print(f"{r.city:<16}{r.hazard.value:<16}{r.status:<8}{r.seconds:>8.1f}")
    print("-" * 48)
    print(
        f"{totals['items']} items | {totals['hits']} hits | {totals['misses']} misses "
        f"| {totals['errors']} errors | {totals['seconds']} s total"
    )
    if slowest and slowest[0].seconds > 0:
        worst = ", ".join(f"{r.city}/{r.hazard.value} {r.seconds:.1f}s" for r in slowest)
        print(f"slowest: {worst}")
    if totals["errors"]:
        print(f"WARNING: {totals['errors']} item(s) failed — see the warnings above")


def main(argv: Sequence[str] | None = None) -> int:
    from obs.log import configure

    parser = argparse.ArgumentParser(description="Warm the hazard-statistic cache.")
    parser.add_argument("--cities", type=Path, default=DEFAULT_CITIES,
                        help=f"JSON city list (default: {DEFAULT_CITIES})")
    parser.add_argument("--hazards", default=",".join(h.value for h in Hazard),
                        help="comma-separated hazards (default: all)")
    parser.add_argument("--limit", type=int, default=None,
                        help="only the first N cities (a quick smoke run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and exit without fitting anything")
    args = parser.parse_args(argv)

    configure()  # entrypoint owns logging config
    cities = load_cities(args.cities)[: args.limit]
    hazards = parse_hazards(args.hazards)
    print(f"{len(cities)} cities x {len(hazards)} hazards = {len(cities) * len(hazards)} fits")

    if args.dry_run:
        for city in cities:
            print(f"  would warm {city.name:<16} ({city.latitude}, {city.longitude})")
        return 0

    results = prewarm(cities, hazards)
    _print_table(results)
    return 1 if summarize(results)["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
