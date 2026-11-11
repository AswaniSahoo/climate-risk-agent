"""Tests for the Open-Meteo Archive budget and its rate-limit breaker.

Open-Meteo counts a long request as many calls (tools/archive_budget.py quotes
the rule), so one 1960 to 2022 fetch weighs about 1,644 against a free limit of
10,000 a day and 5,000 an hour, and production shares Cloud Run's egress
addresses with strangers. The budget refuses before the request; the breaker
stops every request until the next hour after a 429. Both turn into
ArchiveBudgetExceeded, a ClimatologyError, so every caller already degrades.
"""
from datetime import datetime, timezone

import pytest

import tools.archive_budget as budget
import tools.climatology as clim
from agent.contracts import Hazard
from tools.cache_backend import DiskCache, JsonCache, LocalCounter

LIVE_PARAMS = {"latitude": 52.52, "longitude": 13.4, "start_date": "1960-01-01",
               "end_date": "2022-12-31", "daily": "temperature_2m_max", "models": "era5",
               "timezone": "auto"}


class Clock:
    def __init__(self, iso: str) -> None:
        self.now = datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp()

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    clock = Clock("2026-11-11T14:20:00")
    monkeypatch.setattr(budget, "_now", clock)
    budget.reset(LocalCounter(clock=clock))
    return clock


def test_the_live_request_weighs_what_open_meteo_counts():
    assert budget.request_weight(LIVE_PARAMS) == 1644  # 23,011 days / 14, rounded up
    two_vars = {**LIVE_PARAMS, "daily": "temperature_2m_max,precipitation_sum"}
    assert budget.request_weight(two_vars) == 1644  # under the 10-variable step
    short = {**LIVE_PARAMS, "start_date": "2022-12-25", "end_date": "2022-12-31"}
    assert budget.request_weight(short) == 1


def test_requests_inside_the_budget_go_ahead(clock):
    assert budget.reserve(1644) is None
    assert budget.reserve(1644) is None


def test_the_hourly_budget_refuses_and_says_when_it_resets(clock):
    assert budget.reserve(1644) is None
    assert budget.reserve(1644) is None  # 3,288 of 4,000

    refusal = budget.reserve(1644)

    assert refusal == ("ERA5 history skipped for this place: this hour's free data allowance "
                       "for new places is used up. It resets at 15:00 UTC.")


def test_the_daily_budget_refuses_with_the_midnight_reset(clock, monkeypatch):
    monkeypatch.setenv("CRG_ARCHIVE_HOURLY_CALLS", "off")

    for _ in range(4):
        assert budget.reserve(1644) is None  # 6,576 of 8,000
    refusal = budget.reserve(1644)

    assert refusal == ("ERA5 history skipped for this place: today's free data allowance for "
                       "new places is used up. It resets at 00:00 UTC.")


def test_a_refused_request_gives_its_reservation_back(clock):
    budget.reserve(1644)
    budget.reserve(1644)
    budget.reserve(1644)  # refused: must not stay counted

    clock.now += 3600  # next hour: the hour's count is fresh, the day keeps 3,288
    assert budget.reserve(1644) is None
    assert budget.reserve(1644) is None
    assert budget.reserve(1644) is not None  # this hour holds 3,288 again; 4,932 would pass 4,000


def test_the_limits_come_from_the_environment(clock, monkeypatch):
    monkeypatch.setenv("CRG_ARCHIVE_HOURLY_CALLS", "1000")

    assert budget.reserve(1644) is not None
    monkeypatch.setenv("CRG_ARCHIVE_HOURLY_CALLS", "off")
    monkeypatch.setenv("CRG_ARCHIVE_DAILY_CALLS", "off")
    for _ in range(10):
        assert budget.reserve(1644) is None


def test_an_unreadable_limit_keeps_the_default(clock, monkeypatch, caplog):
    monkeypatch.setenv("CRG_ARCHIVE_HOURLY_CALLS", "lots")

    assert budget.hourly_limit() == budget.DEFAULT_HOURLY_CALLS
    assert "CRG_ARCHIVE_HOURLY_CALLS" in caplog.text


def test_the_breaker_stops_every_request_until_the_next_hour(clock):
    budget.trip_breaker()

    refusal = budget.reserve(1)
    assert refusal == ("ERA5 history skipped for this place: the free data service is limiting "
                       "this server's requests. It resets at 15:00 UTC.")
    clock.now = Clock("2026-11-11T15:00:01")()
    assert budget.reserve(1) is None


def test_the_breaker_is_shared_through_the_cache(clock, tmp_path, monkeypatch):
    shared = JsonCache("archive_breaker", backend=DiskCache(tmp_path))
    monkeypatch.setattr(budget, "_breaker_cache", lambda: shared)
    budget.trip_breaker()
    budget.reset(LocalCounter(clock=clock))  # another process: no in-memory breaker

    assert budget.reserve(1) is not None


# --- in front of every live archive request ---------------------------------

@pytest.fixture
def fit_cache(tmp_path, monkeypatch):
    clim.climatology_hazard_stat.cache_clear()
    cache = JsonCache("hazard_fit", backend=DiskCache(tmp_path / "fits"))
    monkeypatch.setattr(clim, "_fit_cache", lambda: cache)
    yield cache
    clim.climatology_hazard_stat.cache_clear()


def test_a_refused_budget_sends_no_archive_request(clock, fit_cache, httpx_mock, monkeypatch):
    monkeypatch.setenv("CRG_ARCHIVE_DAILY_CALLS", "100")

    with pytest.raises(clim.ArchiveBudgetExceeded) as refused:
        clim.climatology_hazard_stat(52.52, 13.40, Hazard.HEATWAVE)

    assert "ERA5 history skipped for this place" in str(refused.value)
    assert isinstance(refused.value, clim.ClimatologyError)  # every caller already degrades
    assert httpx_mock.get_requests() == []


def test_a_429_trips_the_breaker_for_every_later_place(clock, fit_cache, httpx_mock):
    httpx_mock.add_response(status_code=429, text="Hourly API request limit exceeded.")

    with pytest.raises(clim.ClimatologyError):
        clim.climatology_hazard_stat(52.52, 13.40, Hazard.HEATWAVE)
    with pytest.raises(clim.ArchiveBudgetExceeded):
        clim.climatology_hazard_stat(48.85, 2.35, Hazard.HEATWAVE)

    assert len(httpx_mock.get_requests()) == 1  # Paris never reached the service


def test_a_cached_fit_costs_no_budget(clock, fit_cache, httpx_mock, monkeypatch):
    httpx_mock.add_response(json={"timezone": "Europe/Berlin", "daily": {
        "time": ["2000-07-01", "2001-07-01", "2002-07-01"],
        "temperature_2m_max": [30.0, 31.0, 32.0]}})
    clim.climatology_hazard_stat(52.52, 13.40, Hazard.HEATWAVE, start_year=2000, end_year=2002)
    clim.climatology_hazard_stat.cache_clear()
    monkeypatch.setenv("CRG_ARCHIVE_DAILY_CALLS", "0")  # nothing left: only a hit can answer

    stat = clim.climatology_hazard_stat(52.52, 13.40, Hazard.HEATWAVE,
                                        start_year=2000, end_year=2002)

    assert stat.n_years == 3
