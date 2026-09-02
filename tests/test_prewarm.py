"""Tests for scripts/prewarm.py — the cache warmer.

The fit function is injected, so nothing here touches the network or spends the
~55 s a real cold fit costs. What is under test is the loop's contract: every
pair is attempted, a failure is loud and non-fatal, and the summary reports
hits/misses/errors/seconds truthfully.
"""
import json

import pytest

from agent.contracts import Hazard
from scripts.prewarm import (
    DEFAULT_CITIES,
    City,
    load_cities,
    main,
    parse_hazards,
    prewarm,
    summarize,
)

_CITIES = [
    City(name="Rourkela", country="IN", latitude=22.26, longitude=84.85),
    City(name="Berlin", country="DE", latitude=52.52, longitude=13.4),
]
_HAZARDS = [Hazard.HEATWAVE, Hazard.WIND]


def test_prewarm_fits_every_city_hazard_pair():
    calls = []

    def fake_fit(lat, lon, hazard):
        calls.append((lat, lon, hazard))
        return object()

    results = prewarm(_CITIES, _HAZARDS, fit=fake_fit)

    assert len(calls) == 4  # 2 cities x 2 hazards
    assert [r.status for r in results] == ["miss"] * 4  # NullCache under pytest
    assert all(r.seconds >= 0.0 for r in results)


def test_a_failed_city_is_loud_and_does_not_stop_the_run(caplog):
    from tools.climatology import ClimatologyError

    def flaky_fit(lat, lon, hazard):
        if hazard is Hazard.WIND:
            raise ClimatologyError("Open-Meteo Archive request failed: 503")
        return object()

    results = prewarm(_CITIES, _HAZARDS, fit=flaky_fit)

    assert summarize(results) == {
        "items": 4, "hits": 0, "misses": 2, "errors": 2,
        "seconds": summarize(results)["seconds"],
    }
    assert caplog.text.count("prewarm FAILED") == 2  # loud, once per failure
    assert "503" in caplog.text  # the cause, not just the fact


def test_a_cache_hit_is_read_from_telemetry_not_guessed_from_the_clock():
    from tools.cache_backend import JsonCache
    from tools.hazard_stats import HazardStat

    class _Memory:
        name = "redis"
        store: dict[str, str] = {}

        def get(self, key):
            return self.store.get(key)

        def set(self, key, value, ttl_s=None):
            self.store[key] = value

    cache = JsonCache("hazard_fit", backend=_Memory())
    seen = {}

    def fit_through_cache(lat, lon, hazard):
        key = f"{lat}|{lon}|{hazard.value}"
        cached = cache.get_model(key, HazardStat)
        if cached is not None:
            return cached
        stat = seen[key]
        cache.set_model(key, stat, ttl_s=60)
        return stat

    stat = _minimal_stat()
    seen.update({f"22.26|84.85|{h.value}": stat for h in _HAZARDS})
    seen.update({f"52.52|13.4|{h.value}": stat for h in _HAZARDS})

    first = prewarm(_CITIES, _HAZARDS, fit=fit_through_cache)
    second = prewarm(_CITIES, _HAZARDS, fit=fit_through_cache)

    assert [r.status for r in first] == ["miss"] * 4
    assert [r.status for r in second] == ["hit"] * 4


def _minimal_stat():
    from tools.hazard_stats import HazardStat, Representativeness, ReturnLevel

    return HazardStat(
        variable="temperature_2m_max",
        statistic_definition="annual maximum of ERA5 daily-maximum 2 m air temperature",
        unit="°C",
        source="Open-Meteo Archive (ERA5 reanalysis)",
        model="era5",
        native_resolution_deg=0.25,
        captures_diurnal_peak=True,
        timezone="Asia/Kolkata",
        latitude=22.26,
        longitude=84.85,
        n_years=3,
        record_start_year=2000,
        record_end_year=2002,
        record_max=46.0,
        return_levels=[ReturnLevel(return_period_years=10, level=46.0)],
        trend=None,
        is_bias_corrected=False,
        representativeness=Representativeness.POINT_INTERPOLATED_REANALYSIS,
        interpretation="test fixture",
    )


def test_parse_hazards_rejects_an_unknown_name():
    assert parse_hazards("heatwave, wind") == [Hazard.HEATWAVE, Hazard.WIND]

    with pytest.raises(SystemExit) as exc:
        parse_hazards("heatwave,earthquake")
    assert "earthquake" in str(exc.value)  # names the bad input, lists the good ones


def test_dry_run_does_no_work(capsys, monkeypatch):
    import scripts.prewarm as prewarm_mod

    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not fit anything")

    monkeypatch.setattr(prewarm_mod, "prewarm", explode)

    assert main(["--dry-run", "--limit", "2"]) == 0

    out = capsys.readouterr().out
    assert "would warm" in out and "Rourkela" in out


def test_shipped_city_file_is_valid_and_global():
    cities = load_cities(DEFAULT_CITIES)

    assert len(cities) == 25
    named = {c.name for c in cities}
    for expected in ("Rourkela", "Delhi", "Mumbai", "Berlin", "London", "New York",
                     "Tokyo", "Sydney", "Sao Paulo", "Nairobi", "Cairo"):
        assert expected in named
    # 2 dp keeps every entry on the cache key's rounding grid, so a prewarmed
    # city is a hit for the coordinates the UI sends.
    for city in cities:
        assert round(city.latitude, 2) == city.latitude
        assert round(city.longitude, 2) == city.longitude
    # spread: both hemispheres, and east + west of Greenwich
    assert min(c.latitude for c in cities) < 0 < max(c.latitude for c in cities)
    assert min(c.longitude for c in cities) < 0 < max(c.longitude for c in cities)


def test_city_file_documents_that_coordinates_are_approximate():
    payload = json.loads(DEFAULT_CITIES.read_text(encoding="utf-8"))

    assert "approximate" in payload["_note"].lower()
