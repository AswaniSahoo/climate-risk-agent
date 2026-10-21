"""Recompute the ERA5 warming trend the app reports, and write it to an artifact.

WHY THIS EXISTS
---------------
The README quotes a warming trend for Berlin and Delhi. A number in a README
with no file behind it cannot be checked, so this script recomputes it through
the exact call the app makes and commits the output under evals/results/.

HOW
---
For each place: `tools.geocode.geocode` (the app's own geocoder), then
`tools.climatology.climatology_hazard_stat(lat, lon, Hazard.HEATWAVE)` with the
app's defaults (ERA5 through the Open-Meteo archive, 1960-2022, annual maxima
of daily max 2 m temperature). The trend is the drifting-location GEV slope and
the likelihood-ratio p-value from `tools/gev_trend.py`, exactly as the report's
climatology driver shows them.

`--point "LABEL=LAT,LON"` measures a fixed coordinate instead, which is how
the UI's preset location buttons reach the same function (Delhi there is 28.61, 77.21,
a different ERA5 grid cell from the geocoded 28.65, 77.23), so both entry
points can be checked side by side.

The fit cache is bypassed by default: the shared backend is pointed at a fresh
temporary directory and the Redis tier is switched off for this process, so the
numbers come from a live ERA5 fetch and a fresh fit, not from a stored entry.

Run:  uv run python -m scripts.measure_warming_trend
      uv run python -m scripts.measure_warming_trend --places Berlin Delhi \\
          --point "Berlin (UI preset)=52.52,13.40" \\
          --point "Delhi (UI preset)=28.61,77.21" \\
          --out evals/results/warming-trend-2026-10-21.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.contracts import Hazard

DEFAULT_PLACES = ("Berlin", "Delhi")
_HAZARD = Hazard.HEATWAVE


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def trend_row(
    query: str,
    latitude: float,
    longitude: float,
    *,
    location: str,
    resolved_by: str,
    hazard_stat: Callable,
) -> dict[str, Any]:
    """One point -> the trend numbers the app would report for it."""
    stat = hazard_stat(latitude, longitude, _HAZARD)
    trend = stat.trend
    return {
        "query": query,
        "resolved_by": resolved_by,
        "location": location,
        "latitude": latitude,
        "longitude": longitude,
        "variable": stat.variable,
        "unit": stat.unit,
        "record_start_year": stat.record_start_year,
        "record_end_year": stat.record_end_year,
        "n_years": stat.n_years,
        "slope_per_decade": trend.slope_per_decade if trend is not None else None,
        "p_value": trend.p_value if trend is not None else None,
        "significant": trend.significant if trend is not None else None,
        "evaluated_at_year": trend.evaluated_at_year if trend is not None else None,
    }


def parse_point(spec: str) -> tuple[str, float, float]:
    """'LABEL=LAT,LON' -> (label, lat, lon)."""
    label, _, coords = spec.rpartition("=")
    lat, lon = (float(v) for v in coords.split(","))
    return label.strip(), lat, lon


def build_artifact(
    places: Sequence[str],
    points: Sequence[tuple[str, float, float]],
    *,
    geocode: Callable,
    hazard_stat: Callable,
    code_fingerprint: str,
    fit_cache: str,
) -> dict[str, Any]:
    rows = []
    for place in places:
        located = geocode(place)
        rows.append(trend_row(
            place, located.latitude, located.longitude,
            location=", ".join(p for p in (located.name, located.country) if p),
            resolved_by="tools.geocode.geocode",
            hazard_stat=hazard_stat,
        ))
    for label, lat, lon in points:
        rows.append(trend_row(
            label, lat, lon, location=label, resolved_by="fixed coordinates",
            hazard_stat=hazard_stat,
        ))
    return {
        "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "what": (
            "Warming trend in the annual maxima of daily max 2 m temperature: "
            "slope of the drifting-location GEV, p-value of the likelihood-ratio "
            "test against the stationary fit (tools/gev_trend.py)."
        ),
        "source": "ERA5 via the Open-Meteo archive (models=era5)",
        "hazard": _HAZARD.value,
        "code_version": {
            "git_commit": _git_commit(),
            "climatology_fingerprint": code_fingerprint,
            "entry_point": "tools.climatology.climatology_hazard_stat",
        },
        "fit_cache": fit_cache,
        "results": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--places", nargs="*", default=list(DEFAULT_PLACES))
    parser.add_argument(
        "--point", action="append", default=[], metavar="LABEL=LAT,LON",
        help="also measure a fixed coordinate (repeatable)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--use-cache", action="store_true",
        help="read the configured fit cache instead of fitting fresh",
    )
    args = parser.parse_args(argv)

    fit_cache = "configured backend"
    if not args.use_cache:
        # Must run before the first cache read: the backend is resolved once.
        os.environ["CRG_CACHE_DIR"] = tempfile.mkdtemp(prefix="crg-trend-")
        os.environ.pop("UPSTASH_REDIS_REST_URL", None)
        os.environ.pop("UPSTASH_REDIS_REST_TOKEN", None)
        fit_cache = "bypassed (fresh temporary disk cache, no Redis)"

    from tools.climatology import _code_fingerprint, climatology_hazard_stat
    from tools.geocode import geocode

    artifact = build_artifact(
        args.places,
        [parse_point(spec) for spec in args.point],
        geocode=geocode,
        hazard_stat=climatology_hazard_stat,
        code_fingerprint=_code_fingerprint(),
        fit_cache=fit_cache,
    )
    text = json.dumps(artifact, indent=2) + "\n"
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
