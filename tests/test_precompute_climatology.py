"""Tests for scripts/precompute_climatology.py: precomputed ERA5 fits for listed cities.

The contract under test: a precomputed entry is the live path's own result,
stored under the live path's own key, and a city that is not in the file
falls back to the live fetch exactly as before. All HTTP is mocked.
"""
import ast
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import tools.climatology as clim
from agent.contracts import Hazard
from scripts import precompute_climatology as pre
from tools.cache_backend import DiskCache, JsonCache, _with_seed
from tools.hazard_stats import HazardStat

REPO_ROOT = Path(__file__).resolve().parents[1]
START, END = 1990, 2014  # 25 years: enough for the trend test, small enough to stay fast


@pytest.fixture(autouse=True)
def _small_and_isolated(monkeypatch):
    """Short record, few bootstrap refits, no lru_cache carry-over between tests."""
    monkeypatch.setattr(pre, "START_YEAR", START)
    monkeypatch.setattr(pre, "END_YEAR", END)
    monkeypatch.setattr(clim, "_N_BOOT", 20)
    monkeypatch.setattr(clim, "_TREND_N_BOOT", 20)
    clim.climatology_hazard_stat.cache_clear()
    yield
    clim.climatology_hazard_stat.cache_clear()


def _series(seed: int) -> dict[str, list]:
    """Two days a year, deterministic, distinct per variable."""
    times, heat, rain, gust = [], [], [], []
    for i, year in enumerate(range(START, END + 1)):
        for day in ("06-15", "07-15"):
            times.append(f"{year}-{day}")
            heat.append(round(30.0 + 0.05 * i + ((i * 7 + seed) % 5) * 0.3, 1))
            rain.append(round(40.0 + ((i * 3 + seed) % 11) * 2.5, 1))
            gust.append(round(60.0 + ((i * 5 + seed) % 7) * 3.0, 1))
    return {"time": times, "temperature_2m_max": heat, "precipitation_sum": rain,
            "wind_gusts_10m_max": gust}


def _archive(httpx_mock, series_by_lat: dict[float, dict]):
    """Answer any archive request with exactly the `daily` variables it asked for."""

    def respond(request: httpx.Request) -> httpx.Response:
        query = parse_qs(urlparse(str(request.url)).query)
        series = series_by_lat[float(query["latitude"][0])]
        asked = query["daily"][0].split(",")
        daily = {"time": series["time"], **{v: series[v] for v in asked}}
        return httpx.Response(200, json={"timezone": "Europe/Berlin", "daily": daily})

    httpx_mock.add_callback(respond, is_reusable=True, is_optional=True)


BERLIN = pre.City(name="Berlin", country="DE", latitude=52.52, longitude=13.40, preset=True)
PARIS = pre.City(name="Paris", country="FR", latitude=48.85341, longitude=2.3488, query="Paris")


def _live_fit(city, hazard, tmp_path):
    cache = JsonCache("hazard_fit", backend=DiskCache(tmp_path / "live"))
    original = clim._fit_cache
    clim._fit_cache = lambda: cache  # the live path, with its own empty disk tier
    try:
        return clim.climatology_hazard_stat(
            city.latitude, city.longitude, hazard,
            start_year=START, end_year=END, return_periods=pre.RETURN_PERIODS,
        )
    finally:
        clim._fit_cache = original


def test_a_precomputed_fit_equals_the_live_fit_and_sits_under_the_live_key(httpx_mock, tmp_path):
    _archive(httpx_mock, {52.52: _series(1)})
    seed = tmp_path / "seed" / "climatology.json"

    summary = pre.run([BERLIN], list(Hazard), budget_calls=10_000, min_interval_s=0,
                      live=None, seed_path=seed, sleep=lambda s: None)

    assert summary.requests == 1  # one request for all three hazards of the city
    rows = {row["hazard"]: row for row in json.loads(seed.read_text())["entries"]}
    for hazard in Hazard:
        live = _live_fit(BERLIN, hazard, tmp_path / hazard.value)
        assert HazardStat.model_validate(rows[hazard.value]["value"]) == live
        raw_key = clim._fit_cache_key(BERLIN.latitude, BERLIN.longitude, hazard,
                                      START, END, pre.RETURN_PERIODS)
        assert rows[hazard.value]["key"] == JsonCache("hazard_fit").key(raw_key)


def test_the_live_path_serves_a_listed_city_from_the_seed_without_a_request(httpx_mock, tmp_path,
                                                                         monkeypatch):
    _archive(httpx_mock, {52.52: _series(1)})
    seed_dir = tmp_path / "seed"
    pre.run([BERLIN], [Hazard.HEATWAVE], budget_calls=10_000, min_interval_s=0, live=None,
            seed_path=seed_dir / "climatology.json", sleep=lambda s: None)
    live = _live_fit(BERLIN, Hazard.HEATWAVE, tmp_path / "reference")
    asked_before = len(httpx_mock.get_requests())

    seeded = _with_seed(DiskCache(tmp_path / "empty"), {"CRG_CACHE_SEED_DIR": str(seed_dir)})
    monkeypatch.setattr(clim, "_fit_cache", lambda: JsonCache("hazard_fit", backend=seeded))
    clim.climatology_hazard_stat.cache_clear()
    served = clim.climatology_hazard_stat(BERLIN.latitude, BERLIN.longitude, Hazard.HEATWAVE,
                                          start_year=START, end_year=END,
                                          return_periods=pre.RETURN_PERIODS)

    assert served == live
    assert len(httpx_mock.get_requests()) == asked_before  # no archive request at all


def test_a_city_missing_from_the_seed_falls_back_to_the_live_fetch(httpx_mock, tmp_path, monkeypatch):
    _archive(httpx_mock, {52.52: _series(1), 48.85341: _series(2)})
    seed_dir = tmp_path / "seed"
    pre.run([BERLIN], [Hazard.HEATWAVE], budget_calls=10_000, min_interval_s=0, live=None,
            seed_path=seed_dir / "climatology.json", sleep=lambda s: None)
    reference = _live_fit(PARIS, Hazard.HEATWAVE, tmp_path / "reference")
    asked_before = len(httpx_mock.get_requests())

    seeded = _with_seed(DiskCache(tmp_path / "empty"), {"CRG_CACHE_SEED_DIR": str(seed_dir)})
    monkeypatch.setattr(clim, "_fit_cache", lambda: JsonCache("hazard_fit", backend=seeded))
    clim.climatology_hazard_stat.cache_clear()
    served = clim.climatology_hazard_stat(PARIS.latitude, PARIS.longitude, Hazard.HEATWAVE,
                                          start_year=START, end_year=END,
                                          return_periods=pre.RETURN_PERIODS)

    assert served == reference
    assert len(httpx_mock.get_requests()) == asked_before + 1  # the live archive request


def test_the_combined_request_is_the_live_request_with_every_variable(httpx_mock, tmp_path):
    _archive(httpx_mock, {52.52: _series(1)})
    _live_fit(BERLIN, Hazard.WIND, tmp_path / "live")
    pre.run([BERLIN], list(Hazard), budget_calls=10_000, min_interval_s=0, live=None,
            seed_path=tmp_path / "seed.json", sleep=lambda s: None)

    live_q, pre_q = (parse_qs(urlparse(str(r.url)).query) for r in httpx_mock.get_requests())
    assert pre_q.pop("daily") == ["temperature_2m_max,precipitation_sum,wind_gusts_10m_max"]
    assert live_q.pop("daily") == ["wind_gusts_10m_max"]
    assert pre_q == live_q  # same point, dates, model and timezone handling


def test_open_meteo_weighting_prices_the_live_record_at_about_1644_calls():
    days = pre.request_days(1960, 2022)

    assert days == 23_011
    assert pre.weighted_calls(1, days) == pytest.approx(1643.6, abs=0.1)
    assert pre.weighted_calls(3, days) == pre.weighted_calls(1, days)  # under the 10-variable step
    assert pre.weighted_calls(15, 14) == pytest.approx(1.5)  # the pricing page's own example


def test_a_run_never_sends_a_request_the_budget_cannot_cover(httpx_mock, tmp_path):
    summary = pre.run([BERLIN], list(Hazard), budget_calls=100, min_interval_s=0, live=None,
                      seed_path=tmp_path / "seed.json", sleep=lambda s: None)

    assert summary.requests == 0
    assert summary.deferred == ["Berlin"]
    assert httpx_mock.get_requests() == []


def test_requests_are_spaced_by_the_minimum_interval(httpx_mock, tmp_path):
    _archive(httpx_mock, {52.52: _series(1), 48.85341: _series(2)})
    slept: list[float] = []

    pre.run([BERLIN, PARIS], [Hazard.HEATWAVE], budget_calls=10_000, min_interval_s=600,
            live=None, seed_path=tmp_path / "seed.json", sleep=slept.append)

    assert len(slept) == 1 and slept[0] > 590  # nothing before the first, a full gap before the second


def test_a_rate_limit_answer_stops_the_run(httpx_mock, tmp_path):
    httpx_mock.add_response(status_code=429, text="Hourly API request limit exceeded.")

    summary = pre.run([BERLIN, PARIS], [Hazard.HEATWAVE], budget_calls=10_000, min_interval_s=0,
                      live=None, seed_path=tmp_path / "seed.json", sleep=lambda s: None)

    assert summary.requests == 1  # no retry, and Paris is never asked
    assert summary.stopped_by == "HTTP 429: Hourly API request limit exceeded."
    assert not (tmp_path / "seed.json").exists() or json.loads(
        (tmp_path / "seed.json").read_text())["entries"] == []


def test_a_fit_already_in_the_live_cache_is_reused_without_a_request(httpx_mock, tmp_path):
    _archive(httpx_mock, {52.52: _series(1)})
    live_backend = DiskCache(tmp_path / "live")
    stat = _live_fit(BERLIN, Hazard.HEATWAVE, tmp_path / "ref")
    raw_key = clim._fit_cache_key(BERLIN.latitude, BERLIN.longitude, Hazard.HEATWAVE,
                                  START, END, pre.RETURN_PERIODS)
    JsonCache("hazard_fit", backend=live_backend).set_model(raw_key, stat)
    asked_before = len(httpx_mock.get_requests())

    summary = pre.run([BERLIN], [Hazard.HEATWAVE], budget_calls=0, min_interval_s=0,
                      live=live_backend, seed_path=tmp_path / "seed.json", sleep=lambda s: None)

    assert (summary.requests, summary.reused) == (0, 1)
    assert len(httpx_mock.get_requests()) == asked_before
    row = json.loads((tmp_path / "seed.json").read_text())["entries"][0]
    assert HazardStat.model_validate(row["value"]) == stat
    assert "series" not in row  # a reused fit carries no series, so it cannot be refitted offline


def test_refit_recomputes_from_the_stored_series_without_a_request(httpx_mock, tmp_path, monkeypatch):
    _archive(httpx_mock, {52.52: _series(1)})
    seed = tmp_path / "seed.json"
    pre.run([BERLIN], [Hazard.HEATWAVE], budget_calls=10_000, min_interval_s=0, live=None,
            seed_path=seed, sleep=lambda s: None)
    old_key = json.loads(seed.read_text())["entries"][0]["key"]
    asked_before = len(httpx_mock.get_requests())

    monkeypatch.setattr(clim, "_N_BOOT", 25)  # a new fit setting: every key moves
    monkeypatch.setattr(clim, "_TREND_N_BOOT", 25)
    refitted = pre.refit(seed_path=seed)

    row = json.loads(seed.read_text())["entries"][0]
    assert refitted == 1 and row["key"] != old_key
    assert len(httpx_mock.get_requests()) == asked_before
    assert HazardStat.model_validate(row["value"]) == _live_fit(BERLIN, Hazard.HEATWAVE, tmp_path / "r")


def test_the_script_fits_with_the_live_path_defaults():
    import inspect

    live = inspect.signature(clim.climatology_hazard_stat).parameters
    assert (pre.RETURN_PERIODS, live["start_year"].default, live["end_year"].default) == (
        live["return_periods"].default, 1960, 2022)


def test_the_city_list_carries_every_ui_preset_at_its_exact_coordinates():
    """The sidebar labels its preset buttons "cached". The list must cover each one,
    at the coordinates the button sends, or the label is not true on a cold start."""
    tree = ast.parse((REPO_ROOT / "ui" / "app.py").read_text(encoding="utf-8"))
    presets = next(
        ast.literal_eval(node.value) for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "LOCATIONS"
    )
    listed = {(c.latitude, c.longitude) for c in pre.load_cities() if c.preset}

    assert set(presets.values()) == listed


def test_every_listed_city_has_coordinates():
    cities = pre.load_cities()

    assert all(c.latitude is not None and c.longitude is not None for c in cities)
    assert len({(c.latitude, c.longitude) for c in cities}) == len(cities)  # no duplicates


def test_prune_stale_drops_only_the_rows_the_code_no_longer_builds(httpx_mock, tmp_path,
                                                                   monkeypatch):
    _archive(httpx_mock, {52.52: _series(1), 48.85341: _series(2)})
    seed = tmp_path / "seed.json"
    pre.run([BERLIN, PARIS], [Hazard.HEATWAVE], budget_calls=10_000, min_interval_s=0,
            live=None, seed_path=seed, sleep=lambda s: None)
    payload = json.loads(seed.read_text())
    payload["entries"][0]["key"] = "crg:v1:hazard_fit:retired"
    seed.write_text(json.dumps(payload))

    assert len(pre.stale_rows(seed)) == 1
    assert pre.prune_stale(seed) == 1
    assert pre.stale_rows(seed) == [] and len(pre.read_rows(seed)) == 1
