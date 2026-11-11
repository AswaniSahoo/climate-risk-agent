"""Precompute the ERA5 hazard fits the live path computes, into a committed seed file.

WHY. A cold first report pays the climatology step in full: the 1960-2022
Open-Meteo Archive request plus the GEV fit, trend test and bootstrap. The
service runs at min-instances 0 with no Upstash credentials (env var names
read from the live service on 2026-11-11), so the shared cache is a per
instance disk that dies at every scale to zero and the next visitor pays again.
The ERA5 record is static (tools/climatology.py), so the fits for a list of
likely cities can be computed once and shipped in the image.

WHAT IT WRITES. tools/cache_seed/climatology.json, one entry per (city,
hazard), under the exact key tools/climatology.py builds: the same two-decimal
rounding, years, return periods, bootstrap counts and code fingerprint. The
shared cache layer (tools/cache_backend.py) reads these entries before Redis or
disk, so the live path finds a listed city first and everything else falls
through to the live fetch unchanged. An entry fitted here also keeps the annual
maxima it was fitted on, so a later change to the fitting code is re-fitted
offline (--refit) instead of paying the archive again.

WHAT IT COSTS. Open-Meteo, read at the source on 2026-11-11:
- https://open-meteo.com/en/terms: "Less than 10'000 API calls per day, 5'000
  per hour and 600 per minute." and "You may only use the free API services
  for non-commercial purposes."
- https://open-meteo.com/en/pricing: "Requests for data covering more than 10
  weather variables or extending over a period of more than 2 weeks for a
  single location are considered multiple API calls." and "fractional counts
  are used. For example, a request for 2 weeks of data with 15 weather
  variables will be calculated as 1.5 API calls".
The live record is 1960-01-01 to 2022-12-31, 23,011 days, so ONE request counts
as about 1,644 calls: the whole free day covers six. This script asks for all
three hazard variables of a city in one request (three is under the
ten-variable step, so the count does not grow), refuses any request the
--budget-calls budget cannot cover, spaces requests by --min-interval-s, and
stops at the first HTTP 429. A full list therefore takes many days on the free
tier; --plan prints the count before anything is sent.

Run (repo root):
  uv run python -m scripts.precompute_climatology --plan
  uv run python -m scripts.precompute_climatology                 # paced, default budget
  uv run python -m scripts.precompute_climatology --check         # entries current for this code?
  uv run python -m scripts.precompute_climatology --refit         # offline, after a fit-code change
  uv run python -m scripts.precompute_climatology --resolve       # geocode new list entries

To extend the list: add {"name", "country", "query"} to
scripts/precompute_cities.json (query = what a person would type), run
--resolve to pin the coordinates the geocoder returns for it, then a paced run.
Set CRG_CACHE_DIR to a warm cache to reuse fits a live run already paid for.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel, Field

from agent.contracts import Hazard
from tools import archive_budget
from tools import climatology as clim
from tools.cache_backend import (
    CACHE_SCHEMA_VERSION,
    SEED_DIR,
    CacheBackend,
    JsonCache,
    _backend_from_env,
    write_seed_file,
)
from tools.hazard_stats import HazardStat

_log = logging.getLogger("scripts.precompute_climatology")

CITIES_FILE = Path(__file__).resolve().parent / "precompute_cities.json"
SEED_FILE = SEED_DIR / "climatology.json"
SCHEMA_VERSION = 1

# The live path's own defaults, read off its signature so the two cannot drift.
_LIVE_DEFAULTS = inspect.signature(clim.climatology_hazard_stat).parameters
START_YEAR: int = _LIVE_DEFAULTS["start_year"].default
END_YEAR: int = _LIVE_DEFAULTS["end_year"].default
RETURN_PERIODS: tuple[int, ...] = tuple(_LIVE_DEFAULTS["return_periods"].default)

# Free tier, quoted in the module docstring.
FREE_CALLS_PER_DAY = 10_000
FREE_CALLS_PER_HOUR = 5_000
# Two full-record requests: a third of the free day, leaving the rest for
# anything else this machine asks Open-Meteo for.
DEFAULT_BUDGET_CALLS = 3_300
# One request per 40 minutes keeps any rolling hour near a third of its limit.
DEFAULT_MIN_INTERVAL_S = 2_400

PROVENANCE = {
    "what": "ERA5 hazard fits computed with the live code path (tools/climatology.py), "
            "served before the archive request for the listed cities",
    "api": clim.ARCHIVE_URL,
    "api_docs": "https://open-meteo.com/en/docs/historical-weather-api",
    "model": "era5",
    "licence": "CC BY 4.0 (https://open-meteo.com/en/licence)",
    "attribution": "Weather data by Open-Meteo.com (https://open-meteo.com/). ERA5: Hersbach, H. "
                   "et al. (2023), ERA5 hourly data on single levels from 1940 to present, ECMWF, "
                   "https://doi.org/10.24381/cds.adbb2d47. Generated using Copernicus Climate "
                   "Change Service information.",
    "changes": "daily series reduced to annual maxima and fitted with a GEV "
               "(tools/hazard_stats.py, tools/gev_trend.py)",
    "generated_by": "scripts/precompute_climatology.py",
}


class City(BaseModel):
    """One list entry. `preset` marks a UI button; `query` is a typed place name
    whose geocoded coordinates are pinned here, so the typed path hits too."""

    name: str
    country: str = ""
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    query: str | None = None
    preset: bool = False


def load_cities(path: Path = CITIES_FILE) -> list[City]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [City.model_validate(entry) for entry in payload["cities"]]


def request_days(start_year: int, end_year: int) -> int:
    return (date(end_year, 12, 31) - date(start_year, 1, 1)).days + 1


weighted_calls = archive_budget.weighted_calls  # one formula for the script and the live budget


def _coords(city: City) -> tuple[float, float]:
    if city.latitude is None or city.longitude is None:
        raise ValueError(f"{city.name} has no coordinates yet: run --resolve")
    return city.latitude, city.longitude


def fit_key(city: City, hazard: Hazard) -> str:
    """The live fit-cache key (tools/climatology.py), unnamespaced."""
    latitude, longitude = _coords(city)
    return clim._fit_cache_key(latitude, longitude, hazard, START_YEAR, END_YEAR, RETURN_PERIODS)


def entry_key(city: City, hazard: Hazard) -> str:
    """The key the shared cache layer looks up: `crg:<schema>:hazard_fit:<fit key>`."""
    return JsonCache("hazard_fit").key(fit_key(city, hazard))


def series_code() -> str:
    """Identity of the code that turns the archive response into annual maxima.

    A stored series may only be refitted while this is unchanged; otherwise the
    maxima themselves would be stale and the archive has to be asked again.
    """
    variables = {h.value: v.daily_var for h, v in clim._HAZARD_VARS.items()}
    text = inspect.getsource(clim.annual_maxima) + json.dumps(variables, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def version_header() -> dict:
    boot = clim.bootstrap_settings()
    return {
        "cache_schema": CACHE_SCHEMA_VERSION,
        "code_fingerprint": clim._code_fingerprint(),
        "n_boot": boot["n_boot"],
        "trend_n_boot": boot["trend_n_boot"],
        "start_year": START_YEAR,
        "end_year": END_YEAR,
        "return_periods": list(RETURN_PERIODS),
        "series_code": series_code(),
    }


def archive_params(city: City, hazards: Sequence[Hazard]) -> dict[str, str | int | float]:
    """The live request (tools/climatology.py), with every asked variable in one `daily` list."""
    latitude, longitude = _coords(city)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": f"{START_YEAR}-01-01",
        "end_date": f"{END_YEAR}-12-31",
        "daily": ",".join(clim._HAZARD_VARS[h].daily_var for h in hazards),
        "models": "era5",
        "timezone": "auto",
    }


def _row(city: City, hazard: Hazard, stat: HazardStat, series: dict | None) -> dict:
    latitude, longitude = _coords(city)
    row: dict = {"key": entry_key(city, hazard), "city": city.name, "country": city.country,
                 "latitude": latitude, "longitude": longitude, "hazard": hazard.value}
    if series is not None:
        row["series"] = series
    row["value"] = json.loads(stat.model_dump_json())
    return row


def fit_rows(city: City, hazards: Sequence[Hazard], payload: dict) -> list[dict]:
    """The live fit (annual_maxima then build_hazard_stat) for each asked hazard."""
    latitude, longitude = _coords(city)
    daily = payload["daily"]
    tz = payload.get("timezone", "auto")
    rows = []
    for hazard in hazards:
        years, maxima = clim.annual_maxima(daily["time"], daily[clim._HAZARD_VARS[hazard].daily_var])
        stat = clim.build_hazard_stat(
            years, maxima, hazard=hazard, latitude=latitude, longitude=longitude,
            timezone=tz, return_periods=RETURN_PERIODS,
        )
        rows.append(_row(city, hazard, stat, {"timezone": tz, "years": years, "maxima": maxima,
                                              "series_code": series_code()}))
    return rows


def read_rows(path: Path = SEED_FILE) -> list[dict]:
    if not Path(path).exists():
        return []
    return list(json.loads(Path(path).read_text(encoding="utf-8"))["entries"])


def save_rows(rows: Sequence[dict], path: Path = SEED_FILE) -> None:
    header = {
        "schema_version": SCHEMA_VERSION,
        "namespace": "hazard_fit",
        "provenance": {**PROVENANCE,
                       "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")},
        "version": version_header(),
        "city_list": "scripts/precompute_cities.json",
    }
    write_seed_file(path, header=header, entries=list(rows))


def _reuse(live: CacheBackend | None, city: City, hazard: Hazard) -> HazardStat | None:
    if live is None:
        return None
    return JsonCache("hazard_fit", backend=live).get_model(fit_key(city, hazard), HazardStat)


@dataclass
class Summary:
    requests: int = 0
    weighted_calls: float = 0.0
    fitted: int = 0
    reused: int = 0
    present: int = 0
    deferred: list[str] = field(default_factory=list)
    stopped_by: str | None = None


def run(
    cities: Sequence[City],
    hazards: Sequence[Hazard],
    *,
    budget_calls: float,
    min_interval_s: float,
    live: CacheBackend | None,
    seed_path: Path = SEED_FILE,
    fetch: Callable[..., httpx.Response] = httpx.get,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> Summary:
    """Fill the seed file for `cities` x `hazards`, paying no more than `budget_calls`."""
    rows = {row["key"]: row for row in read_rows(seed_path)}
    summary = Summary()
    days = request_days(START_YEAR, END_YEAR)
    last_request: float | None = None
    for city in cities:
        missing: list[Hazard] = []
        changed = False
        for hazard in hazards:
            key = entry_key(city, hazard)
            if key in rows:
                summary.present += 1
                continue
            stat = _reuse(live, city, hazard)
            if stat is not None:
                rows[key] = _row(city, hazard, stat, None)
                summary.reused += 1
                changed = True
                continue
            missing.append(hazard)
        if missing:
            cost = weighted_calls(len(missing), days)
            if summary.weighted_calls + cost > budget_calls:
                summary.deferred.append(city.name)
            else:
                if last_request is not None:
                    wait = min_interval_s - (clock() - last_request)
                    if wait > 0:
                        print(f"  pacing: {wait:.0f} s before {city.name}", flush=True)
                        sleep(wait)
                print(f"  fetching {city.name} ({', '.join(h.value for h in missing)}): "
                      f"~{cost:.0f} weighted calls", flush=True)
                response = fetch(clim.ARCHIVE_URL, params=archive_params(city, missing), timeout=120)
                last_request = clock()
                summary.requests += 1
                summary.weighted_calls += cost
                if response.status_code != 200:
                    summary.stopped_by = f"HTTP {response.status_code}: {response.text[:200]}"
                    _log.warning("archive refused %s (%s); stopping, no retry",
                                 city.name, summary.stopped_by)
                    if changed:
                        save_rows(list(rows.values()), seed_path)
                    break
                for row in fit_rows(city, missing, response.json()):
                    rows[row["key"]] = row
                    summary.fitted += 1
                changed = True
        if changed:
            save_rows(list(rows.values()), seed_path)  # after every city: an interrupted run resumes
    return summary


def refit(seed_path: Path = SEED_FILE) -> int:
    """Re-fit every entry that carries its series, under the current fit code. No network.

    Entries are re-keyed, so the fit-code change that retired them is the one
    the new entries are valid for. Returns how many were refitted.
    """
    current = series_code()
    out: dict[str, dict] = {}
    refitted = 0
    for row in read_rows(seed_path):
        series = row.get("series")
        if series is None or series.get("series_code") != current:
            out[row["key"]] = row  # nothing to refit from: kept as is, `--check` says if stale
            continue
        city = City(name=row["city"], country=row.get("country", ""),
                    latitude=row["latitude"], longitude=row["longitude"])
        hazard = Hazard(row["hazard"])
        stat = clim.build_hazard_stat(
            series["years"], series["maxima"], hazard=hazard, latitude=row["latitude"],
            longitude=row["longitude"], timezone=series["timezone"], return_periods=RETURN_PERIODS,
        )
        new = _row(city, hazard, stat, series)
        out[new["key"]] = new
        refitted += 1
    save_rows(list(out.values()), seed_path)
    return refitted


def check(cities: Sequence[City], hazards: Sequence[Hazard], seed_path: Path = SEED_FILE) -> dict:
    """How many listed (city, hazard) pairs the file answers for the CURRENT code."""
    keys = {row["key"] for row in read_rows(seed_path)}
    wanted = {entry_key(c, h): (c, h) for c in cities for h in hazards}
    current = [pair for key, pair in wanted.items() if key in keys]
    presets_missing = [f"{c.name}/{h.value}" for key, (c, h) in wanted.items()
                       if c.preset and key not in keys]
    return {"listed": len(wanted), "current": len(current), "stale": len(keys - set(wanted)),
            "presets_missing": presets_missing}


def _row_key(row: dict) -> str:
    city = City(name=row["city"], country=row.get("country", ""),
                latitude=row["latitude"], longitude=row["longitude"])
    return entry_key(city, Hazard(row["hazard"]))


def stale_rows(seed_path: Path = SEED_FILE) -> list[str]:
    """Rows whose key the current code no longer builds, so they can never answer.

    The key is rebuilt from each row's own coordinates and hazard with the
    live key function, so a change to the fitting code (its source
    fingerprint), the bootstrap counts, the years or the return periods shows
    up here. Needs no network.
    """
    return [
        f"hazard_fit {row['city']}/{row['hazard']} at ({row['latitude']}, {row['longitude']})"
        for row in read_rows(seed_path) if row["key"] != _row_key(row)
    ]


def prune_stale(seed_path: Path = SEED_FILE) -> int:
    """Drop the stale rows. Run --refit first: a row with a stored series is
    re-keyed there instead, and one without can only come back by a paced fetch."""
    rows = read_rows(seed_path)
    kept = [row for row in rows if row["key"] == _row_key(row)]
    if len(kept) != len(rows):
        save_rows(kept, seed_path)
    return len(rows) - len(kept)


def resolve(path: Path = CITIES_FILE, pause_s: float = 0.5) -> int:
    """Geocode list entries that have a query but no coordinates; write them back.

    The resolved name and country replace the typed ones, so a wrong pick by
    the geocoder (it ranks by population) is visible in the list, not hidden.
    """
    from tools.geocode import GeocodeError, geocode

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    resolved = 0
    for entry in payload["cities"]:
        if entry.get("query") and entry.get("latitude") is None:
            try:
                found = geocode(entry["query"])
            except GeocodeError as exc:
                _log.warning("could not resolve %r (%s); left without coordinates", entry["query"], exc)
                continue
            entry["latitude"], entry["longitude"] = found.latitude, found.longitude
            entry["name"], entry["country"] = found.name, found.country
            resolved += 1
            time.sleep(pause_s)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return resolved


def _plan(cities: Sequence[City], hazards: Sequence[Hazard], live: CacheBackend | None) -> None:
    keys = {row["key"] for row in read_rows()}
    days = request_days(START_YEAR, END_YEAR)
    need = [c for c in cities
            if any(entry_key(c, h) not in keys and _reuse(live, c, h) is None for h in hazards)]
    per_request = weighted_calls(len(hazards), days)
    total = per_request * len(need)
    print(f"{len(cities)} cities x {len(hazards)} hazards; {len(need)} cities still need a request")
    print(f"one {START_YEAR}-{END_YEAR} request = {days} days = ~{per_request:.0f} weighted calls")
    print(f"to finish: ~{total:,.0f} weighted calls = {total / FREE_CALLS_PER_DAY:.1f} free days "
          f"(free tier: {FREE_CALLS_PER_DAY:,}/day, {FREE_CALLS_PER_HOUR:,}/hour)")
    print(f"default budget {DEFAULT_BUDGET_CALLS:,} per run covers "
          f"{int(DEFAULT_BUDGET_CALLS // per_request)} requests")


def main(argv: Sequence[str] | None = None) -> int:
    from obs.log import configure

    parser = argparse.ArgumentParser(description="Precompute ERA5 hazard fits into the seed file.")
    parser.add_argument("--cities", type=Path, default=CITIES_FILE)
    parser.add_argument("--hazards", default=",".join(h.value for h in Hazard))
    parser.add_argument("--limit", type=int, default=None, help="only the first N cities")
    parser.add_argument("--budget-calls", type=float, default=DEFAULT_BUDGET_CALLS,
                        help="weighted Open-Meteo calls this run may spend")
    parser.add_argument("--min-interval-s", type=float, default=DEFAULT_MIN_INTERVAL_S)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="print the cost and exit")
    mode.add_argument("--check", action="store_true",
                      help="exit 1 if a preset lacks a current entry or a shipped row is stale")
    mode.add_argument("--refit", action="store_true", help="refit stored series offline")
    mode.add_argument("--prune-stale", action="store_true",
                      help="drop shipped rows the current code no longer builds a key for")
    mode.add_argument("--resolve", action="store_true", help="geocode list entries without coordinates")
    args = parser.parse_args(argv)
    configure()

    if args.resolve:
        print(f"resolved {resolve(args.cities)} list entries")
        return 0
    if args.refit:
        print(f"refitted {refit()} entries from their stored series")
        return 0
    if args.prune_stale:
        print(f"pruned {prune_stale()} stale entries")
        return 0
    cities = load_cities(args.cities)[: args.limit]
    hazards = [Hazard(name.strip()) for name in args.hazards.split(",") if name.strip()]
    if args.check:
        result = {**check(cities, hazards), "stale_rows": stale_rows()}
        print(json.dumps(result, indent=1))
        return 1 if result["presets_missing"] or result["stale_rows"] else 0
    # Reuse what a live run already paid for: the live tier only, never the seed.
    live = _backend_from_env(os.environ)
    if args.plan:
        _plan(cities, hazards, live)
        return 0
    summary = run(cities, hazards, budget_calls=args.budget_calls,
                  min_interval_s=args.min_interval_s, live=live)
    print(f"requests {summary.requests} (~{summary.weighted_calls:.0f} weighted calls) | "
          f"fitted {summary.fitted} | reused {summary.reused} | already present {summary.present} | "
          f"deferred {len(summary.deferred)} cities")
    if summary.stopped_by:
        print(f"STOPPED: {summary.stopped_by}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
