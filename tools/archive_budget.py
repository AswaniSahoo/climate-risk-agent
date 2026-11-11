"""A budget on Open-Meteo Archive requests, and a breaker for its rate limit.

WHY. Open-Meteo counts a long request as many calls. Read at the source on
2026-11-11: the terms (open-meteo.com/en/terms) allow the free API "Less than
10'000 API calls per day, 5'000 per hour and 600 per minute", and the pricing
page (open-meteo.com/en/pricing) says "Requests for data covering more than 10
weather variables or extending over a period of more than 2 weeks for a single
location are considered multiple API calls", counted in fractions. The live
ERA5 fetch in tools/climatology.py asks 1960-01-01 to 2022-12-31, 23,011 days,
so one new place costs about 1,644 of those calls. Production runs on Cloud
Run, whose egress addresses other tenants share, and the terms add "Rate
limits are applied by IP addresses and application".

WHAT. Before every live archive request, `reserve(weight)` adds the request's
weight to this hour's and today's (UTC) counts on the shared counter
(tools/cache_backend.py `counter_from_env`: the Cloud Storage counter when
CRG_CACHE_GCS_BUCKET is set, this process otherwise) and refuses when either
would pass its limit, giving the reservation back. A refusal is one sentence
for the reader. After any HTTP 429, `trip_breaker()` stops every archive
request until the next full hour (UTC), for this process at once and, through
the shared cache, for the next instance too.

No per-minute budget: one full-record request already weighs more than the
600-a-minute figure, so that limit cannot be met by pacing; the breaker is the
answer if the service enforces it.

Settings (weighted calls; "off" disables a limit, anything unreadable keeps
the default): CRG_ARCHIVE_DAILY_CALLS (default 8,000) and
CRG_ARCHIVE_HOURLY_CALLS (default 4,000), under the free 10,000 and 5,000.
"""
from __future__ import annotations

import logging
import math
import os
import threading
import time
from collections.abc import Mapping
from datetime import date, datetime, timezone

from pydantic import BaseModel

from tools.cache_backend import (
    CACHE_SCHEMA_VERSION,
    CacheUnavailable,
    Counter,
    JsonCache,
    counter_from_env,
)

_log = logging.getLogger(__name__)

FREE_CALLS_PER_DAY = 10_000
FREE_CALLS_PER_HOUR = 5_000
FREE_CALLS_PER_MINUTE = 600
DEFAULT_DAILY_CALLS = 8_000
DEFAULT_HOURLY_CALLS = 4_000

_HOUR_TTL_S = 2 * 3600  # a key outlives its window, then the store drops it
_DAY_TTL_S = 2 * 24 * 3600

_now = time.time  # module attribute so tests can move the clock

_SKIPPED = "ERA5 history skipped for this place: "
_DAY_REASON = (_SKIPPED + "today's free data allowance for new places is used up. "
               "It resets at 00:00 UTC.")
_HOUR_REASON = (_SKIPPED + "this hour's free data allowance for new places is used up. "
                "It resets at {at} UTC.")
_BREAKER_REASON = (_SKIPPED + "the free data service is limiting this server's requests. "
                   "It resets at {at} UTC.")


def weighted_calls(n_variables: int, days: int) -> float:
    """Open-Meteo's count for one request: more than 10 variables or more than
    14 days for one location make it several calls, in fractions."""
    return max(1.0, n_variables / 10.0) * max(1.0, days / 14.0)


def request_days(start_date: str, end_date: str) -> int:
    return (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days + 1


def request_weight(params: Mapping[str, object]) -> int:
    """The weighted calls an archive request with these query params costs, rounded up."""
    variables = [v for v in str(params.get("daily", "")).split(",") if v.strip()]
    days = request_days(str(params["start_date"]), str(params["end_date"]))
    return math.ceil(weighted_calls(max(1, len(variables)), days))


# --- settings ----------------------------------------------------------------

_warned: set[tuple[str, str]] = set()


def _limit(env: Mapping[str, str], name: str, default: int) -> int | None:
    raw = (env.get(name) or "").strip()
    if raw.lower() == "off":
        return None
    if not raw:
        return default
    try:
        value = int(raw)
        if value < 0:
            raise ValueError
    except ValueError:
        if (name, raw) not in _warned:
            _warned.add((name, raw))
            _log.warning("%s=%r is not a whole number >= 0 or 'off'; using the default %d",
                         name, raw, default)
        return default
    return value


def daily_limit(env: Mapping[str, str] = os.environ) -> int | None:
    return _limit(env, "CRG_ARCHIVE_DAILY_CALLS", DEFAULT_DAILY_CALLS)


def hourly_limit(env: Mapping[str, str] = os.environ) -> int | None:
    return _limit(env, "CRG_ARCHIVE_HOURLY_CALLS", DEFAULT_HOURLY_CALLS)


# --- state -------------------------------------------------------------------

_lock = threading.Lock()
_counter: Counter | None = None
_breaker_until = 0.0  # this process's copy of the breaker


def reset(counter: Counter | None = None) -> None:
    """Start over on `counter`, or on one built from the environment (tests)."""
    global _counter, _breaker_until
    with _lock:
        _counter = counter
        _breaker_until = 0.0


def _get_counter() -> Counter:
    global _counter
    with _lock:
        if _counter is None:
            _counter = counter_from_env()
        return _counter


class _Breaker(BaseModel):
    until: float


def _breaker_cache() -> JsonCache:
    return JsonCache("archive_breaker")


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _next_hour(epoch: float) -> float:
    return (math.floor(epoch / 3600) + 1) * 3600.0


def _hour_label(epoch: float) -> str:
    return _utc(epoch).strftime("%Y-%m-%dT%H")


def _clock_text(epoch: float) -> str:
    return _utc(epoch).strftime("%H:%M")


def _key(window: str, label: str) -> str:
    return f"crg:{CACHE_SCHEMA_VERSION}:archive_budget:{window}:{label}"


def _breaker_active(now: float) -> float | None:
    if _breaker_until > now:
        return _breaker_until
    hit = _breaker_cache().get_model(_hour_label(now), _Breaker)
    return hit.until if hit is not None and hit.until > now else None


# --- the two calls tools/climatology.py makes --------------------------------

def reserve(weight: int) -> str | None:
    """Hold `weight` weighted calls against this hour and today, or refuse.

    Returns None when the request may go ahead, else the sentence for the
    reader. A refused reservation is given back, so it never counts.
    """
    now = _now()
    tripped = _breaker_active(now)
    if tripped is not None:
        return _BREAKER_REASON.format(at=_clock_text(tripped))
    hourly, daily = hourly_limit(), daily_limit()
    counter = _get_counter()
    taken: list[tuple[str, int]] = []
    try:
        if hourly is not None:
            key = _key("hour", _hour_label(now))
            taken.append((key, _HOUR_TTL_S))
            if counter.add(key, weight, _HOUR_TTL_S) > hourly:
                _give_back(counter, taken, weight)
                return _HOUR_REASON.format(at=_clock_text(_next_hour(now)))
        if daily is not None:
            key = _key("day", _utc(now).strftime("%Y-%m-%d"))
            taken.append((key, _DAY_TTL_S))
            if counter.add(key, weight, _DAY_TTL_S) > daily:
                _give_back(counter, taken, weight)
                return _DAY_REASON
    except CacheUnavailable as exc:
        # Only a bare shared counter raises (counter_from_env wraps it in a
        # fallback that does not); the request then goes ahead uncounted.
        _log.warning("archive budget counter unavailable (%s); this request is not counted", exc)
    return None


def _give_back(counter: Counter, taken: list[tuple[str, int]], weight: int) -> None:
    for key, ttl_s in taken:
        try:
            counter.add(key, -weight, ttl_s)
        except CacheUnavailable as exc:
            _log.warning("could not give back an archive reservation (%s); the count stays "
                         "higher than what was sent", exc)


def trip_breaker() -> None:
    """Open-Meteo answered 429: no archive request from anyone until the next hour."""
    global _breaker_until
    now = _now()
    until = _next_hour(now)
    with _lock:
        _breaker_until = max(_breaker_until, until)
    _breaker_cache().set_model(_hour_label(now), _Breaker(until=until),
                               ttl_s=max(1, math.ceil(until - now)))
    _log.warning("Open-Meteo Archive answered 429: no archive request until %s UTC",
                 _clock_text(until))
