"""Measure how Open-Meteo forecast error grows with lead time.

WHY THIS EXISTS
---------------
The agent already reports *how extreme* a forecast peak is against 60+ years of
ERA5 (the GEV moat). What it cannot yet say is *how much to trust the forecast
that produced that peak*. A 40 C day predicted 1 day out and the same 40 C day
predicted 14 days out are not the same claim, and today the report treats them
identically. This script is the measurement half of the fix: it measures, from
real archived forecasts, how far off the forecast typically is at each lead
time, so a later change can weight the report's confidence by the horizon that
was actually asked for.

HOW THE MEASUREMENT WORKS
-------------------------
Open-Meteo's Previous Model Runs API archives, for every valid time, what the
model *had predicted* for that time N days earlier. Quoting the docs
(https://open-meteo.com/en/docs/previous-runs-api, read 2026-09-02):

    "Data from past model runs is aligned to fixed lead-time offsets of 1-7
    days. Requesting temperature_2m_previous_day1 returns the value predicted
    24 hours before valid time; _previous_day2 returns 48 hours before, and so
    on up to day 7."

    "_previous_day0 is the current model run (equivalent to the live Forecast
    API)."

So for one location we can pull eight aligned series for the same valid times:
the day-0 run (our reference) and the day-1..day-7 forecasts. Subtracting gives
the error at each lead time directly.

WHAT "TRUTH" MEANS HERE  (read this before quoting the numbers)
--------------------------------------------------------------
The reference series is the **day-0 model run**, not a station observation and
not ERA5. It is the model's own most recent analysis of that hour. So every
number below is a *run-to-run consistency* error: how much the forecast moved
between issuing it N days out and the model's final word on the same hour. That
is a LOWER BOUND on true forecast error, because it contains no model bias that
was present at both lead times. The docs themselves point at reanalysis as the
alternative reference ("Comparing these offsets against observations or
reanalysis..."); pairing against ERA5 would need the separate Historical
Weather API on a different grid and is deliberately out of scope here. The
committed table records this under provenance.truth_series so nobody reads the
numbers as verification-grade forecast error.

API SHAPE (verified live, 2026-09-02)
-------------------------------------
* Endpoint: https://previous-runs-api.open-meteo.com/v1/forecast
* start_date / end_date work (same as the Forecast API).
* `timezone=auto` returns whole local days, matching how tools/forecast.py
  frames a "day" for the agent.
* Only HOURLY variables take the _previous_dayN suffix. Daily aggregates are
  rejected outright: `daily=temperature_2m_max_previous_day1` returns HTTP 400
  "Cannot initialize ForecastVariableDaily from invalid String value". We
  therefore aggregate hourly -> daily ourselves, matching the agent's fields.
* `wind_gusts_10m_previous_dayN` IS served with real values even though the
  docs' variable table does not list gusts. Verified by fetching it.
* `temperature_2m_previous_day0` is accepted but comes back keyed as plain
  `temperature_2m`, so day-0 is requested by the bare variable name.
* Units follow the Forecast API defaults: Celsius, mm, km/h.

WHY THE MODEL IS PINNED  (measured, not assumed)
------------------------------------------------
The obvious choice is `best_match`, the default, because that is what
tools/forecast.py serves. Measuring it produced nonsense, and the nonsense is
worth recording. At Delhi over Mar-Apr 2024, best_match gave:

    wind_speed_10m   day-1 bias -5.26 km/h, day-6 bias -5.22, day-7 bias -0.11
    temperature_2m   day-1..day-6 bias about 0.0, day-7 bias +5.33 C

A day-1 forecast that is 5 km/h off while a day-7 forecast is spot on is not
forecast skill. best_match is a per-location, per-variable seamless blend, so
the archived offsets it returns are stitched from *different underlying models*
and are not comparable to each other or to day-0. `gfs_seamless` shows the same
artefact inside CONUS (New York wind_speed bias jumps +0.82 at day 1 to +3.54
at day 2) because it blends HRRR/NBM there.

Pinning one global model removes the artefact entirely. `gfs_global` (NCEP GFS
Global 0.11/0.25 deg) was chosen because, verified live:
  * it is a single model, so the eight series are mutually comparable — MAE
    rises monotonically with lead time at every site tested and biases sit near
    zero, which is what a real skill curve looks like;
  * it archives all four hazard variables at previous-run offsets. ECMWF IFS
    0.25 deg does NOT archive wind gusts at those offsets (day-1/4/7 come back
    all-null) and has a coverage hole in early January 2024;
  * GFS has the longest archive on this API (2 m temperature from March 2021).

The cost of pinning: the agent itself calls best_match, so this table describes
GFS-global skill, which is an approximation of what the agent's own forecasts
deliver. That trade is recorded in provenance.model and in the README.

THE DAY-5 STEP IS OUTPUT RESOLUTION, NOT PREDICTABILITY
-------------------------------------------------------
GFS emits hourly fields out to forecast hour 120 and 3-hourly beyond it, so
lead days 5, 6 and 7 arrive as a 3-hourly series spread back over hours. It is
directly visible: at Mumbai in July 2024 the fraction of hours that exactly
repeat the previous hour jumps from 0.19 at lead day 4 to 0.58 at lead day 5,
and the month's precipitation total falls from 794 mm at day 4 to 397 mm at
day 5 while the day-0 reference reads 1223 mm. Temperature is unaffected
(repeat fraction stays near 0.17 at every lead, monthly totals all near 21,000).

This is a defect in the *delivered product*, not a measurement bug: the agent's
own day-6 precipitation_sum is computed from the same coarse fields, so the
error is real error a user would receive. But it is not atmospheric
predictability decaying, and a step change at day 5 must not be read as such.
Every row therefore carries `hourly_repeat_fraction`, measured, so the step is
visible in the committed table rather than buried here. Treat precipitation at
lead days 5-7 as resolution-contaminated: its bias is dominated by the coarser
accumulation, not by the forecast being wrong about whether it will rain.

RATE LIMITS (https://open-meteo.com/en/pricing, read 2026-09-02)
---------------------------------------------------------------
Free tier: 600 calls/min, 5,000 calls/hour, 10,000 calls/day, 300,000/month.
Cost is weighted, not per-HTTP-request: "Requests for data covering more than
10 weather variables or extending over a period of more than 2 weeks for a
single location are considered multiple API calls... a request for 2 weeks of
data with 15 weather variables will be calculated as 1.5 API calls". We chunk
by calendar month and pace request launches; `--dry-run` prints the estimated
weighted cost before anything is fetched.

Usage
-----
    python scripts/measure_forecast_skill.py --dry-run
    python scripts/measure_forecast_skill.py --limit 3
    python scripts/measure_forecast_skill.py            # full 25-city run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx
import numpy as np
from tqdm import tqdm

log = logging.getLogger("measure_forecast_skill")

REPO_ROOT = Path(__file__).resolve().parent.parent
CITIES_PATH = REPO_ROOT / "scripts" / "prewarm_cities.json"
RAW_DIR = REPO_ROOT / "data" / "skill"
TABLE_PATH = REPO_ROOT / "tools" / "forecast_skill_table.json"

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# See "WHY THE MODEL IS PINNED" above. Not negotiable without re-measuring: the
# default best_match blends models across lead offsets and fabricates biases.
DEFAULT_MODEL = "gfs_global"

# The API archives lead-time offsets 1..7 only. The agent's forecast tool allows
# horizons up to 16 days; tools/forecast_skill.py clamps beyond 7 and flags it.
LEAD_DAYS = tuple(range(1, 8))

DEFAULT_START = date(2024, 1, 1)
DEFAULT_END = date(2025, 12, 31)


@dataclass(frozen=True)
class Hazard:
    """One hazard variable: how to ask for it, and how to fold hours into a day.

    `key` deliberately mirrors the field name tools/forecast.py already serves
    (precipitation_sum, temperature_2m_max, wind_speed_10m_max) so the skill
    table can be looked up by the same name the report already uses.
    """

    key: str
    api_variable: str
    aggregation: str  # "max" | "sum"
    unit: str


HAZARDS: tuple[Hazard, ...] = (
    Hazard("temperature_2m_max", "temperature_2m", "max", "°C"),
    Hazard("precipitation_sum", "precipitation", "sum", "mm"),
    Hazard("wind_speed_10m_max", "wind_speed_10m", "max", "km/h"),
    Hazard("wind_gusts_10m_max", "wind_gusts_10m", "max", "km/h"),
)

# A day needs (nearly) all its hours to aggregate honestly. DST transitions give
# 23 or 25 hours under timezone=auto, so 23 is the floor rather than 24.
MIN_HOURS_PER_DAY = 23

# Below this many extreme days a hit rate is noise, not a measurement.
MIN_EXTREME_DAYS = 5

EXTREME_PERCENTILE = 95.0


# --------------------------------------------------------------------------- #
# Metrics: pure, array-in / dict-out, so they are testable without a network.
# --------------------------------------------------------------------------- #


def extreme_threshold(truth: np.ndarray) -> float:
    """The location's own 95th-percentile day. NaN if there is nothing to fit."""
    finite = truth[np.isfinite(truth)]
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, EXTREME_PERCENTILE))


def lead_metrics(
    truth: np.ndarray, forecast: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Error stats for one lead day at one location.

    Pairs the two series, drops any day missing on either side, and returns both
    the finished metrics and the raw sums, because pooling across cities has to
    re-weight by sample count rather than average averages.

    `threshold` is the truth series' 95th percentile. The extreme-hit rate is
    the fraction of days that truth put above it which the forecast also put
    above it: the probability of detection for that location's worst 5% of days.
    """
    truth = np.asarray(truth, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    if truth.shape != forecast.shape:
        raise ValueError(f"shape mismatch: truth {truth.shape} vs forecast {forecast.shape}")

    paired = np.isfinite(truth) & np.isfinite(forecast)
    t = truth[paired]
    f = forecast[paired]
    n = int(t.size)

    result: dict[str, Any] = {
        "n_days": n,
        "sum_abs_error": 0.0,
        "sum_error": 0.0,
        "sum_sq_error": 0.0,
        "mae": None,
        "bias": None,
        "rmse": None,
        "n_extreme_days": 0,
        "n_extreme_hits": 0,
        "extreme_hit_rate": None,
    }
    if n == 0:
        return result

    err = f - t
    result["sum_abs_error"] = float(np.sum(np.abs(err)))
    result["sum_error"] = float(np.sum(err))
    result["sum_sq_error"] = float(np.sum(err * err))
    result["mae"] = result["sum_abs_error"] / n
    result["bias"] = result["sum_error"] / n
    result["rmse"] = math.sqrt(result["sum_sq_error"] / n)

    # A threshold of 0 (or NaN) means the percentile is degenerate — an arid
    # location where 95% of days record no rain. "Above zero" is not an extreme,
    # so refuse to report a hit rate rather than report a meaningless one.
    if math.isfinite(threshold) and threshold > 0:
        is_extreme = t > threshold
        n_extreme = int(np.count_nonzero(is_extreme))
        # Counts are recorded unconditionally so that pooling across cities is
        # correct; only the *reported rate* is gated on having enough events.
        # (Gating the counts instead silently pooled zero hits against a
        # non-zero event total and reported a hit rate of 0.000 everywhere.)
        result["n_extreme_days"] = n_extreme
        result["n_extreme_hits"] = int(np.count_nonzero(f[is_extreme] > threshold))
        if n_extreme >= MIN_EXTREME_DAYS:
            result["extreme_hit_rate"] = result["n_extreme_hits"] / n_extreme

    return result


def pool_metrics(per_city: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine one lead day's per-city metrics into a single row.

    Weighted by sample count, not a mean of means: a city with half the archive
    coverage should not carry equal weight. RMSE pools through the mean squared
    error, which is why lead_metrics hands back the raw sums.
    """
    n = sum(int(c["n_days"]) for c in per_city)
    n_extreme = sum(int(c["n_extreme_days"]) for c in per_city)
    hits = sum(int(c["n_extreme_hits"]) for c in per_city)
    scored = [c for c in per_city if c["n_days"] > 0]

    row: dict[str, Any] = {
        "mae": None,
        "bias": None,
        "rmse": None,
        "extreme_hit_rate": None,
        "n_days": n,
        "n_extreme_days": n_extreme,
        "n_cities": len(scored),
    }
    if n > 0:
        row["mae"] = sum(float(c["sum_abs_error"]) for c in per_city) / n
        row["bias"] = sum(float(c["sum_error"]) for c in per_city) / n
        row["rmse"] = math.sqrt(sum(float(c["sum_sq_error"]) for c in per_city) / n)
    if n_extreme >= MIN_EXTREME_DAYS:
        row["extreme_hit_rate"] = hits / n_extreme

    # Coarse-source detector, pooled over cities. See "THE DAY-5 STEP" in the
    # module docstring: a jump here between lead 4 and lead 5 is GFS switching
    # from hourly to 3-hourly output, not predictability collapsing.
    pairs = sum(int(c.get("repeat_pairs", 0)) for c in per_city)
    row["hourly_repeat_fraction"] = (
        sum(int(c.get("repeat_equal", 0)) for c in per_city) / pairs if pairs else None
    )
    return row


def aggregate_daily(
    times: list[str], values: list[float | None], how: str
) -> dict[str, float]:
    """Fold an hourly series into local days, dropping days with missing hours.

    A partial day would bias a sum low and a max low, so an incomplete day is
    simply absent from the result. Dropping is safe because the caller pairs
    truth and forecast by date afterwards.
    """
    buckets: dict[str, list[float]] = defaultdict(list)
    holes: set[str] = set()
    for stamp, value in zip(times, values):
        day = stamp[:10]
        if value is None:
            holes.add(day)
            continue
        buckets[day].append(float(value))

    out: dict[str, float] = {}
    for day, hours in buckets.items():
        if day in holes or len(hours) < MIN_HOURS_PER_DAY:
            continue
        out[day] = max(hours) if how == "max" else float(sum(hours))
    return out


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def month_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split a date range into whole calendar months, clipped to the range.

    Whole months matter: a day split across two requests would aggregate from
    half its hours in each and be dropped by aggregate_daily.
    """
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        if cursor.month == 12:
            next_month = date(cursor.year + 1, 1, 1)
        else:
            next_month = date(cursor.year, cursor.month + 1, 1)
        chunks.append((cursor, min(next_month - timedelta(days=1), end)))
        cursor = next_month
    return chunks


def hourly_variables() -> list[str]:
    """Day-0 (bare name) plus day-1..day-7 for every hazard, in a stable order."""
    names: list[str] = []
    for hazard in HAZARDS:
        names.append(hazard.api_variable)
        names.extend(f"{hazard.api_variable}_previous_day{d}" for d in LEAD_DAYS)
    return names


def estimated_api_calls(n_requests: int, n_variables: int, days_per_request: float) -> float:
    """Open-Meteo's own weighting: >10 variables or >14 days costs a fraction more."""
    return n_requests * max(1.0, n_variables / 10.0) * max(1.0, days_per_request / 14.0)


class Pacer:
    """Floor on the gap between request launches, shared across worker threads.

    Pacing on *requests per minute* is the wrong unit and it fails in practice:
    the free tier's 600/min ceiling counts Open-Meteo's WEIGHTED calls, and one
    month of 32 variables costs about 7 of them. A first full run launched at
    ~4 requests/second and drew a wall of HTTP 429s. The interval is therefore
    derived from the per-request cost and a target calls-per-minute budget.

    `penalise` exists because a 429 means the whole pool is over budget, not
    just the one worker that saw it: pushing the shared gate forward slows every
    thread at once instead of letting the others keep hammering.
    """

    def __init__(self, calls_per_request: float, calls_per_minute: float) -> None:
        self.min_interval = calls_per_request * 60.0 / max(1.0, calls_per_minute)
        self._lock = threading.Lock()
        self._next_at = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self.min_interval
        if wait_for > 0:
            time.sleep(wait_for)

    def penalise(self, seconds: float) -> None:
        """Hold every worker back after a rate-limit rejection."""
        with self._lock:
            self._next_at = max(self._next_at, time.monotonic()) + seconds


def wait_for_quota(client: httpx.Client, poll_s: float = 60.0, max_wait_s: float = 4200.0) -> bool:
    """Block until the API answers a cheap request, or give up loudly.

    Open-Meteo's hourly ceiling is 5,000 weighted calls and a rejection says so
    in the body: "Hourly API request limit exceeded. Please try again in the
    next hour." Once that trips, every request fails until the window rolls
    over, so a run started inside it burns retries and finishes with nothing.
    This polls a 1-variable, 1-day request (cost: 1 call) and reports progress
    on every attempt rather than sitting silent.
    """
    probe = {
        "latitude": 52.52, "longitude": 13.41, "hourly": "temperature_2m",
        "start_date": "2024-03-01", "end_date": "2024-03-01",
    }
    started = time.monotonic()
    attempt = 0
    while time.monotonic() - started < max_wait_s:
        attempt += 1
        try:
            response = client.get(PREVIOUS_RUNS_URL, params=probe, timeout=60.0)
            if response.status_code == 200:
                log.info("quota available after %.0fs (%d probes)", time.monotonic() - started, attempt)
                return True
            reason = response.json().get("reason", response.text[:120]) if response.content else ""
            log.warning(
                "QUOTA BLOCKED (probe %d, %.0fs elapsed): HTTP %d %s — retrying in %.0fs",
                attempt, time.monotonic() - started, response.status_code, reason, poll_s,
            )
        except httpx.HTTPError as exc:
            log.warning("QUOTA PROBE failed: %s — retrying in %.0fs", exc, poll_s)
        time.sleep(poll_s)
    log.warning("GAVE UP waiting for quota after %.0fs", time.monotonic() - started)
    return False


def fetch_chunk(
    client: httpx.Client,
    pacer: Pacer,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    model: str = DEFAULT_MODEL,
    max_attempts: int = 5,
) -> dict[str, list[Any]]:
    """One (city, month) request, retrying 429 and 5xx with backoff.

    429 carries a Retry-After often enough to be worth honouring; when it is
    absent we fall back to exponential backoff. A 4xx that is not 429 is our own
    bad request and must fail loudly rather than spin.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(hourly_variables()),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "auto",
        "models": model,
    }
    last_error = ""
    for attempt in range(max_attempts):
        pacer.wait()
        try:
            response = client.get(PREVIOUS_RUNS_URL, params=params, timeout=180.0)
            if response.status_code == 200:
                return response.json()["hourly"]
            if response.status_code == 429 or response.status_code >= 500:
                retry_after = response.headers.get("Retry-After")
                # A 429 means the minutely budget is spent, so back off on that
                # timescale, not on a few seconds.
                delay = (
                    float(retry_after)
                    if retry_after and retry_after.isdigit()
                    else min(60.0, 5.0 * 2**attempt)
                )
                last_error = f"HTTP {response.status_code}"
                log.warning(
                    "RATE-LIMIT/SERVER %s for %.2f,%.2f %s..%s — retry %d/%d in %.0fs",
                    last_error, latitude, longitude, start, end, attempt + 1, max_attempts, delay,
                )
                pacer.penalise(delay)
                time.sleep(delay)
                continue
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
        except (httpx.TransportError, json.JSONDecodeError, KeyError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            delay = 2.0 * 2**attempt
            log.warning(
                "TRANSPORT %s for %.2f,%.2f %s..%s — retry %d/%d in %.0fs",
                last_error, latitude, longitude, start, end, attempt + 1, max_attempts, delay,
            )
            if attempt == max_attempts - 1:
                break
            time.sleep(delay)
    raise RuntimeError(f"gave up after {max_attempts} attempts ({last_error})")


# --------------------------------------------------------------------------- #
# Per-city assembly
# --------------------------------------------------------------------------- #


def repeat_counts(values: list[float | None]) -> tuple[int, int]:
    """Consecutive hours holding an identical value, over comparable pairs.

    The detector for a coarser source: GFS is hourly to forecast hour 120 and
    3-hourly after it, and a 3-hourly field spread over hours repeats in runs.
    A high fraction at lead days 5-7 and a low one at 1-4 is the fingerprint.
    """
    equal = pairs = 0
    for a, b in zip(values, values[1:]):
        if a is None or b is None:
            continue
        pairs += 1
        equal += a == b
    return equal, pairs


@dataclass
class CitySeries:
    """Daily series for one city: series[variable_name][YYYY-MM-DD] -> value."""

    series: dict[str, dict[str, float]] = field(
        default_factory=lambda: defaultdict(dict)
    )
    repeat_equal: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    repeat_pairs: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def absorb(self, hourly: dict[str, list[Any]]) -> None:
        times = hourly["time"]
        for hazard in HAZARDS:
            for name in (hazard.api_variable, *(f"{hazard.api_variable}_previous_day{d}" for d in LEAD_DAYS)):
                values = hourly.get(name)
                if values is None:
                    log.warning("MISSING VARIABLE in response: %s", name)
                    continue
                self.series[name].update(aggregate_daily(times, values, hazard.aggregation))
                equal, pairs = repeat_counts(values)
                self.repeat_equal[name] += equal
                self.repeat_pairs[name] += pairs


def city_metrics(city: CitySeries) -> dict[str, Any]:
    """Per hazard, per lead day: paired error stats against the day-0 series."""
    out: dict[str, Any] = {}
    for hazard in HAZARDS:
        truth_map = city.series.get(hazard.api_variable, {})
        days = sorted(truth_map)
        truth = np.array([truth_map[d] for d in days], dtype=float)
        threshold = extreme_threshold(truth)
        leads: dict[str, Any] = {}
        for lead in LEAD_DAYS:
            name = f"{hazard.api_variable}_previous_day{lead}"
            fmap = city.series.get(name, {})
            forecast = np.array([fmap.get(d, np.nan) for d in days], dtype=float)
            row = lead_metrics(truth, forecast, threshold)
            equal, pairs = city.repeat_equal.get(name, 0), city.repeat_pairs.get(name, 0)
            row["repeat_equal"] = equal
            row["repeat_pairs"] = pairs
            row["hourly_repeat_fraction"] = (equal / pairs) if pairs else None
            leads[str(lead)] = row
        out[hazard.key] = {
            "unit": hazard.unit,
            "api_variable": hazard.api_variable,
            "daily_aggregation": hazard.aggregation,
            "extreme_threshold": None if not math.isfinite(threshold) else threshold,
            "n_truth_days": int(np.count_nonzero(np.isfinite(truth))),
            "lead_days": leads,
        }
    return out


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def script_version_hash() -> str:
    """First 12 hex of this file's SHA-256 — the table says which code made it."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()[:12]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def load_cities(limit: int | None = None, stride: int = 1) -> list[dict[str, Any]]:
    """The prewarm set, optionally thinned.

    `stride` exists because the full 25-city, 2-year run costs about 4,200
    weighted API calls and the free tier allows 5,000 per hour, which leaves no
    room for a retry or a second attempt (measured: a first run overran the
    hourly limit and every subsequent request came back
    "Hourly API request limit exceeded"). A stride thins the set without
    hand-picking it: prewarm_cities.json is ordered by region, so taking every
    Nth entry keeps the climates spread instead of concentrating on one
    continent, and the rule is stated in provenance for anyone checking.
    """
    cities = json.loads(CITIES_PATH.read_text(encoding="utf-8"))["cities"]
    cities = cities[::stride]
    return cities[:limit] if limit else cities


def iter_tasks(
    cities: list[dict[str, Any]], chunks: list[tuple[date, date]]
) -> Iterator[tuple[dict[str, Any], tuple[date, date]]]:
    for city in cities:
        for chunk in chunks:
            yield city, chunk


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, default=None, help="only the first N cities")
    p.add_argument("--stride", type=int, default=1, help="take every Nth city (keeps the regional spread)")
    p.add_argument("--dry-run", action="store_true", help="print the request plan and cost, fetch nothing")
    p.add_argument("--start", type=date.fromisoformat, default=DEFAULT_START)
    p.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END)
    p.add_argument("--model", default=DEFAULT_MODEL, help="Open-Meteo models= value; must be a single model, not a seamless blend")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument(
        "--calls-per-minute", type=float, default=250.0,
        help="weighted-call budget to pace against; free tier ceiling is 600/min",
    )
    p.add_argument("--max-attempts", type=int, default=6)
    p.add_argument(
        "--wait-for-quota", action="store_true",
        help="poll until the hourly rate-limit window rolls over before starting",
    )
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--table", type=Path, default=TABLE_PATH)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    cities = load_cities(args.limit, args.stride)
    chunks = month_chunks(args.start, args.end)
    variables = hourly_variables()
    n_requests = len(cities) * len(chunks)
    span_days = (args.end - args.start).days + 1
    cost = estimated_api_calls(n_requests, len(variables), span_days / len(chunks))
    per_request_cost = cost / n_requests
    pacer = Pacer(per_request_cost, args.calls_per_minute)

    print(f"cities            : {len(cities)}")
    print(f"date range        : {args.start} .. {args.end}  ({span_days} days)")
    print(f"month chunks/city : {len(chunks)}")
    print(f"hourly variables  : {len(variables)} ({len(HAZARDS)} hazards x day-0 + day-1..7)")
    print(f"HTTP requests     : {n_requests}")
    print(f"est. API calls    : {cost:.0f}  (free tier: 10,000/day, 600/min)")
    print(f"cost per request  : {per_request_cost:.1f} weighted calls")
    print(
        f"pacing            : {pacer.min_interval:.2f}s between launches "
        f"(budget {args.calls_per_minute:.0f} calls/min), est. wall time "
        f"{n_requests * pacer.min_interval / 60:.0f} min"
    )
    print(f"endpoint          : {PREVIOUS_RUNS_URL}")
    print(f"model             : {args.model}")
    if args.dry_run:
        print("\n--dry-run: no requests issued. Sample URL for the first task:")
        first, (cs, ce) = next(iter_tasks(cities, chunks))
        print(
            f"  {PREVIOUS_RUNS_URL}?latitude={first['latitude']}&longitude={first['longitude']}"
            f"&hourly={','.join(variables[:3])},...&start_date={cs}&end_date={ce}"
            f"&timezone=auto&models={args.model}"
        )
        return 0

    if cost > 9000:
        log.warning("ESTIMATED COST %.0f calls is close to the 10,000/day free-tier ceiling", cost)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    per_city: dict[str, CitySeries] = {c["name"]: CitySeries() for c in cities}
    failures: list[str] = []

    with httpx.Client(headers={"User-Agent": "climate-risk-agent/skill-measurement"}) as client:
        if args.wait_for_quota and not wait_for_quota(client):
            log.warning("ABORTING: rate-limit window never cleared, nothing was measured")
            return 1
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(
                    fetch_chunk, client, pacer, c["latitude"], c["longitude"], s, e,
                    args.model, args.max_attempts,
                ): (c, s, e)
                for c, (s, e) in iter_tasks(cities, chunks)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="fetching", unit="req"):
                city, s, e = futures[future]
                try:
                    per_city[city["name"]].absorb(future.result())
                except Exception as exc:  # noqa: BLE001 — every failure must be visible
                    label = f"{city['name']} {s}..{e}"
                    failures.append(label)
                    log.warning("FAILED %s: %s", label, exc)

    if failures:
        log.warning("%d of %d requests failed. Affected: %s", len(failures), n_requests, ", ".join(failures[:10]))

    # Per-city raw results, gitignored (data/ is in .gitignore).
    results: dict[str, dict[str, Any]] = {}
    for city in cities:
        metrics = city_metrics(per_city[city["name"]])
        results[city["name"]] = metrics
        payload = {"city": city, "hazards": metrics}
        (args.raw_dir / f"{slugify(city['name'])}.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    used = [
        c["name"] for c in cities
        if any(results[c["name"]][h.key]["n_truth_days"] > 0 for h in HAZARDS)
    ]
    if len(used) < len(cities):
        log.warning("%d cities returned no usable data at all", len(cities) - len(used))

    table: dict[str, Any] = {
        "schema_version": 1,
        "provenance": {
            "api": PREVIOUS_RUNS_URL,
            "api_docs": "https://open-meteo.com/en/docs/previous-runs-api",
            "model": args.model,
            "model_note": (
                "Pinned to a single model on purpose. The API default best_match, and any "
                "*_seamless blend, stitch different underlying models across lead offsets and "
                "produce lead-independent biases that are model differences, not forecast error "
                "(measured: Delhi wind_speed_10m best_match bias -5.26 km/h at day 1 vs -0.11 at "
                "day 7). The agent's own tools/forecast.py calls best_match, so this table is a "
                "GFS-global approximation of the skill the agent actually delivers."
            ),
            "truth_series": (
                "day-0 model run (the bare variable name, equivalent to the live Forecast API). "
                "NOT station observations and NOT ERA5, so these are run-to-run consistency errors "
                "and a lower bound on true forecast error."
            ),
            "date_range": {"start": args.start.isoformat(), "end": args.end.isoformat()},
            "timezone": "auto (local days, matching tools/forecast.py)",
            "city_count": len(used),
            "cities": used,
            "city_selection": (
                f"every {args.stride}th entry of scripts/prewarm_cities.json (25 cities, ordered by "
                f"region) -> {len(cities)} requested, {len(used)} with usable data. Thinned because "
                f"the full 25-city 2-year run costs ~4,200 weighted API calls against a 5,000/hour "
                f"free-tier ceiling, leaving no room to retry."
                if args.stride > 1
                else "all of scripts/prewarm_cities.json"
            ),
            "requests_attempted": n_requests,
            "requests_failed": len(failures),
            "hourly_repeat_fraction_note": (
                "Fraction of consecutive hours holding an identical value, measured per lead day. "
                "GFS emits hourly fields to forecast hour 120 and 3-hourly beyond it, so this jumps "
                "at lead day 5. The step in MAE/bias at day 5 is therefore partly output resolution, "
                "not atmospheric predictability. Precipitation is worst affected: daily sums at lead "
                "days 5-7 are systematically low because the coarser accumulation is spread back over "
                "hours. Read those rows as delivered-product error, not as a skill cliff."
            ),
            "extreme_percentile": EXTREME_PERCENTILE,
            "min_extreme_days": MIN_EXTREME_DAYS,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "script": "scripts/measure_forecast_skill.py",
            "script_sha256_12": script_version_hash(),
        },
        "hazards": {},
    }
    for hazard in HAZARDS:
        rows = {
            str(lead): pool_metrics([results[c][hazard.key]["lead_days"][str(lead)] for c in results])
            for lead in LEAD_DAYS
        }
        table["hazards"][hazard.key] = {
            "unit": hazard.unit,
            "api_variable": hazard.api_variable,
            "daily_aggregation": hazard.aggregation,
            "lead_days": rows,
        }

    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.table}")
    print(f"wrote {len(cities)} per-city files to {args.raw_dir}")

    for key, block in table["hazards"].items():
        print(f"\n{key} ({block['unit']})")
        print("  lead   MAE    bias    RMSE   hit-rate      n   repeat")
        for lead in LEAD_DAYS:
            r = block["lead_days"][str(lead)]
            hr = "  n/a " if r["extreme_hit_rate"] is None else f"{r['extreme_hit_rate']:6.3f}"
            rp = " n/a " if r["hourly_repeat_fraction"] is None else f"{r['hourly_repeat_fraction']:5.3f}"
            if r["mae"] is None:
                print(f"  d{lead}     no data")
                continue
            print(
                f"  d{lead}  {r['mae']:6.3f} {r['bias']:+7.3f} {r['rmse']:6.3f}   {hr}  "
                f"{r['n_days']:6d}   {rp}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
