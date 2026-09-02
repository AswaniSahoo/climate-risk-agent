"""Tests for the forecast-skill measurement and its committed table.

Three things need to hold, and none of them may touch the network:

1. The metric maths is right. Checked against a tiny synthetic series whose
   MAE, bias, RMSE and hit rate can be worked out by hand, so a regression in
   the arithmetic shows up as a wrong number rather than a plausible one.
2. The loader clamps horizons past the measured archive. The API publishes
   lead offsets 1-7; tools/forecast.py accepts up to 16 days. Everything past
   day 7 must come back flagged, never silently interpolated.
3. `--dry-run` fetches nothing. pytest-httpx fails the test on any unmatched
   request, so an accidental network call in the planning path is caught here.
"""
from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from scripts.measure_forecast_skill import (
    aggregate_daily,
    estimated_api_calls,
    extreme_threshold,
    hourly_variables,
    lead_metrics,
    month_chunks,
    pool_metrics,
    repeat_counts,
)
from scripts.measure_forecast_skill import main as measure_main
from tools.forecast_skill import (
    MAX_MEASURED_LEAD_DAY,
    SkillEntry,
    SkillTableError,
    available_hazards,
    forecast_skill,
    load_skill_table,
    skill_for,
)

# --------------------------------------------------------------------------- #
# 1. Metrics on a hand-checkable series
# --------------------------------------------------------------------------- #


def test_lead_metrics_on_hand_computed_series():
    # errors: +1, -1, +2, 0, -2  ->  |e| = 1,1,2,0,2 (mean 1.2)
    #                                 e  = +1,-1,+2,0,-2 (mean 0.0)
    #                                 e^2 = 1,1,4,0,4 (mean 2.0)
    truth = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    forecast = np.array([11.0, 19.0, 32.0, 40.0, 48.0])

    m = lead_metrics(truth, forecast, threshold=100.0)  # threshold above everything

    assert m["n_days"] == 5
    assert m["mae"] == pytest.approx(1.2)
    assert m["bias"] == pytest.approx(0.0)
    assert m["rmse"] == pytest.approx(math.sqrt(2.0))
    # No day clears the threshold, so there is nothing to score.
    assert m["n_extreme_days"] == 0
    assert m["extreme_hit_rate"] is None


def test_lead_metrics_extreme_hit_rate_counts_only_days_truth_called_extreme():
    truth = np.array([1.0, 2.0, 3.0, 50.0, 60.0, 70.0])
    # threshold 10: truth days 3,4,5 are extreme. Forecast clears it on two.
    forecast = np.array([1.0, 99.0, 3.0, 55.0, 65.0, 5.0])

    m = lead_metrics(truth, forecast, threshold=10.0)

    assert m["n_extreme_days"] == 3
    assert m["n_extreme_hits"] == 2
    # day 2's false alarm (99 > 10 while truth was 2) must NOT count as a hit
    assert m["extreme_hit_rate"] is None  # below MIN_EXTREME_DAYS, gated
    assert m["n_extreme_hits"] == 2  # ...but the count is still recorded for pooling


def test_lead_metrics_drops_days_missing_on_either_side():
    truth = np.array([10.0, np.nan, 30.0, 40.0])
    forecast = np.array([12.0, 20.0, np.nan, 44.0])

    m = lead_metrics(truth, forecast, threshold=1000.0)

    assert m["n_days"] == 2  # only days 0 and 3 are paired
    assert m["mae"] == pytest.approx(3.0)  # |2| and |4|


def test_lead_metrics_empty_pairing_is_not_an_error():
    m = lead_metrics(np.array([np.nan, np.nan]), np.array([1.0, 2.0]), threshold=1.0)
    assert m["n_days"] == 0
    assert m["mae"] is None and m["rmse"] is None


def test_lead_metrics_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="shape mismatch"):
        lead_metrics(np.array([1.0]), np.array([1.0, 2.0]), threshold=1.0)


def test_degenerate_threshold_suppresses_hit_rate():
    """An arid location where 95% of days are dry gives q95 == 0.

    "Above zero" is not an extreme, so the rate must be refused rather than
    reported as a perfect score.
    """
    # 40 dry days and one wet one: the 95th percentile lands inside the run of
    # zeros. (With only 20 samples numpy interpolates and returns 0.25, which is
    # why the degenerate case needs a realistic dry-day count to reproduce.)
    truth = np.zeros(41)
    truth[-1] = 5.0
    assert extreme_threshold(truth) == 0.0

    m = lead_metrics(truth, truth.copy(), threshold=0.0)
    assert m["n_extreme_days"] == 0
    assert m["extreme_hit_rate"] is None


def test_extreme_threshold_ignores_missing_days():
    assert extreme_threshold(np.array([np.nan, 0.0, 100.0])) == pytest.approx(95.0)
    assert math.isnan(extreme_threshold(np.array([np.nan, np.nan])))


# --------------------------------------------------------------------------- #
# 2. Pooling across cities
# --------------------------------------------------------------------------- #


def test_pool_metrics_weights_by_sample_count_not_by_city():
    """A 100-day city and a 1-day city must not carry equal weight."""
    big = lead_metrics(np.zeros(100), np.ones(100), threshold=1e9)  # error +1 everywhere
    small = lead_metrics(np.zeros(1), np.full(1, 11.0), threshold=1e9)  # error +11 once

    pooled = pool_metrics([big, small])

    assert pooled["n_days"] == 101
    assert pooled["n_cities"] == 2
    # Weighted: (100*1 + 1*11) / 101 = 1.099..., not the mean-of-means 6.0
    assert pooled["mae"] == pytest.approx(111 / 101)
    assert pooled["mae"] < 2.0


def test_pool_metrics_hit_rate_survives_cities_below_the_reporting_floor():
    """Regression: gating the raw counts pooled 0 hits against a real total.

    Each city here has 3 extreme days, under MIN_EXTREME_DAYS, so neither
    reports its own rate. Pooled they are 6 events and the rate must appear.
    """
    truth = np.array([0.0, 0.0, 0.0, 0.0, 50.0, 60.0, 70.0])
    hit_all = np.array([0.0, 0.0, 0.0, 0.0, 50.0, 60.0, 70.0])
    miss_all = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0])

    a = lead_metrics(truth, hit_all, threshold=10.0)
    b = lead_metrics(truth, miss_all, threshold=10.0)
    assert a["extreme_hit_rate"] is None and b["extreme_hit_rate"] is None

    pooled = pool_metrics([a, b])
    assert pooled["n_extreme_days"] == 6
    assert pooled["extreme_hit_rate"] == pytest.approx(0.5)


def test_pool_metrics_with_no_data_returns_nulls_not_zeros():
    empty = lead_metrics(np.array([np.nan]), np.array([np.nan]), threshold=1.0)
    pooled = pool_metrics([empty])
    assert pooled["mae"] is None
    assert pooled["n_days"] == 0
    assert pooled["n_cities"] == 0


# --------------------------------------------------------------------------- #
# 3. Hourly -> daily folding
# --------------------------------------------------------------------------- #


def _day(stamp_prefix: str, n: int = 24) -> list[str]:
    return [f"{stamp_prefix}T{h:02d}:00" for h in range(n)]


def test_aggregate_daily_max_and_sum():
    times = _day("2024-03-01")
    values: list[float | None] = [float(h) for h in range(24)]

    assert aggregate_daily(times, values, "max") == {"2024-03-01": 23.0}
    assert aggregate_daily(times, values, "sum") == {"2024-03-01": pytest.approx(276.0)}


def test_aggregate_daily_drops_days_with_a_missing_hour():
    """A partial day would bias a sum low and a max low, so it is dropped."""
    times = _day("2024-03-01")
    values: list[float | None] = [1.0] * 24
    values[5] = None

    assert aggregate_daily(times, values, "sum") == {}


def test_aggregate_daily_drops_short_days_but_tolerates_dst():
    # 22 hours: too short, dropped. 23 hours: a DST spring-forward day, kept.
    assert aggregate_daily(_day("2024-03-01", 22), [1.0] * 22, "sum") == {}
    assert aggregate_daily(_day("2024-03-01", 23), [1.0] * 23, "sum") == {
        "2024-03-01": pytest.approx(23.0)
    }


def test_repeat_counts_detects_a_three_hourly_source():
    """The coarse-source fingerprint: values held across runs of 3 hours."""
    hourly: list[float | None] = [1.0, 2.0, 3.0, 4.0]
    assert repeat_counts(hourly) == (0, 3)

    three_hourly: list[float | None] = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    equal, pairs = repeat_counts(three_hourly)
    assert (equal, pairs) == (4, 5)
    assert equal / pairs > 0.5

    assert repeat_counts([1.0, None, 1.0]) == (0, 0)


# --------------------------------------------------------------------------- #
# 4. Request planning
# --------------------------------------------------------------------------- #


def test_month_chunks_are_whole_calendar_months_clipped_to_the_range():
    chunks = month_chunks(date(2024, 1, 15), date(2024, 3, 10))
    assert [(str(s), str(e)) for s, e in chunks] == [
        ("2024-01-15", "2024-01-31"),
        ("2024-02-01", "2024-02-29"),  # leap year
        ("2024-03-01", "2024-03-10"),
    ]


def test_hourly_variables_covers_day0_plus_seven_leads_per_hazard():
    names = hourly_variables()
    assert len(names) == 32  # 4 hazards x (day-0 + day-1..7)
    # day-0 is the bare variable name: the API returns _previous_day0 under it
    assert "temperature_2m" in names
    assert "temperature_2m_previous_day7" in names
    assert "temperature_2m_previous_day0" not in names
    assert "wind_gusts_10m_previous_day1" in names


def test_estimated_api_calls_matches_open_meteos_published_weighting():
    """Docs: "2 weeks of data with 15 weather variables ... 1.5 API calls"."""
    assert estimated_api_calls(1, 15, 14) == pytest.approx(1.5)
    assert estimated_api_calls(1, 15, 28) == pytest.approx(3.0)
    assert estimated_api_calls(1, 5, 7) == pytest.approx(1.0)  # never below one call


# --------------------------------------------------------------------------- #
# 5. The script's dry run must not touch the network
# --------------------------------------------------------------------------- #


def test_dry_run_plans_without_fetching(httpx_mock, capsys):
    # httpx_mock registers no responses; pytest-httpx raises on any request, so
    # this fails loudly if the planning path ever reaches out.
    exit_code = measure_main(["--dry-run", "--limit", "2"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "no requests issued" in out
    assert "cities            : 2" in out
    assert "models=gfs_global" in out
    assert httpx_mock.get_requests() == []


# --------------------------------------------------------------------------- #
# 6. The committed table and its loader
# --------------------------------------------------------------------------- #


def test_committed_table_is_present_and_shaped_right():
    table = load_skill_table()

    assert table.schema_version == 1
    assert set(available_hazards()) >= {
        "temperature_2m_max",
        "precipitation_sum",
        "wind_speed_10m_max",
        "wind_gusts_10m_max",
    }
    for key in ("api", "model", "truth_series", "date_range", "city_count", "generated_at",
                "script_sha256_12"):
        assert key in table.provenance, f"provenance is missing {key}"
    assert table.provenance["city_count"] > 0
    for hazard, block in table.hazards.items():
        assert set(block["lead_days"]) == {str(d) for d in range(1, 8)}, hazard


def test_skill_for_returns_the_measured_row_within_the_archive():
    entry = skill_for("temperature_2m_max", horizon_days=3)

    assert isinstance(entry, SkillEntry)
    assert entry.lead_day == 3
    assert entry.requested_horizon_days == 3
    assert entry.extrapolated is False
    assert entry.mae > 0
    assert entry.unit == "°C"


@pytest.mark.parametrize("horizon", [8, 10, 16])
def test_skill_for_clamps_beyond_the_archive_and_flags_it(horizon):
    """The API stops at lead day 7; tools/forecast.py allows 16."""
    entry = skill_for("temperature_2m_max", horizon_days=horizon)
    day7 = skill_for("temperature_2m_max", horizon_days=MAX_MEASURED_LEAD_DAY)

    assert entry.lead_day == MAX_MEASURED_LEAD_DAY
    assert entry.requested_horizon_days == horizon
    assert entry.extrapolated is True
    assert entry.mae == day7.mae  # clamped, not extrapolated upward


def test_temperature_skill_degrades_with_lead_time():
    """The whole point of the table: a day-7 forecast is worse than a day-1 one.

    Checked on temperature, which is the clean case. Precipitation MAE
    saturates and wobbles once the forecast loses the timing of a rain event,
    so it is deliberately not asserted monotone here.
    """
    maes = [skill_for("temperature_2m_max", d).mae for d in range(1, 8)]

    assert maes[-1] > maes[0] * 1.2, f"day-7 should be clearly worse than day-1: {maes}"
    # Non-decreasing, with a little room for sampling noise between adjacent days.
    for earlier, later in zip(maes, maes[1:]):
        assert later >= earlier * 0.98, f"MAE improved with lead time: {maes}"


def test_extreme_hit_rate_falls_with_lead_time():
    """Detection of the worst 5% of days gets worse further out, or it is not skill."""
    rates = [skill_for("temperature_2m_max", d).extreme_hit_rate for d in range(1, 8)]
    assert all(r is not None for r in rates), f"hit rate should be measured at every lead: {rates}"
    assert all(0.0 <= r <= 1.0 for r in rates)  # type: ignore[operator]
    assert rates[-1] < rates[0], f"hit rate should fall with lead time: {rates}"


def test_skill_for_rejects_a_nonsense_horizon():
    with pytest.raises(ValueError, match="must be >= 1"):
        skill_for("temperature_2m_max", horizon_days=0)


def test_skill_for_rejects_an_unknown_hazard():
    with pytest.raises(SkillTableError, match="not in the skill table"):
        skill_for("sea_level_rise", horizon_days=1)


def test_missing_table_says_how_to_build_it(tmp_path):
    with pytest.raises(SkillTableError, match="measure_forecast_skill"):
        skill_for("temperature_2m_max", 1, path=tmp_path / "absent.json")


def test_malformed_table_is_rejected_rather_than_half_read(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text('{"schema_version": 1}', encoding="utf-8")
    with pytest.raises(SkillTableError, match="unexpected shape"):
        skill_for("temperature_2m_max", 1, path=bad)

    worse = tmp_path / "worse.json"
    worse.write_text("not json at all", encoding="utf-8")
    with pytest.raises(SkillTableError, match="not valid JSON"):
        skill_for("temperature_2m_max", 1, path=worse)


def test_loader_reads_a_synthetic_table_without_the_committed_file(tmp_path):
    """The loader must work off any path, not just the committed one."""
    synthetic = {
        "schema_version": 1,
        "provenance": {"api": "x"},
        "hazards": {
            "temperature_2m_max": {
                "unit": "°C",
                "lead_days": {
                    str(d): {
                        "mae": float(d), "bias": 0.1, "rmse": float(d) * 2,
                        "extreme_hit_rate": None, "hourly_repeat_fraction": 0.2,
                        "n_days": 10, "n_extreme_days": 0, "n_cities": 1,
                    }
                    for d in range(1, 8)
                },
            }
        },
    }
    path = tmp_path / "table.json"
    path.write_text(json.dumps(synthetic), encoding="utf-8")

    assert skill_for("temperature_2m_max", 4, path=path).mae == 4.0
    assert skill_for("temperature_2m_max", 99, path=path).mae == 7.0
    assert skill_for("temperature_2m_max", 99, path=path).extrapolated is True


# --------------------------------------------------------------------------- #
# 7. The confidence weight built from the table
# --------------------------------------------------------------------------- #


def test_confidence_weight_is_one_at_day_one_and_falls_with_lead_time():
    """Day 1 is the baseline, so nothing scores higher than it did before weighting."""
    weights = [forecast_skill("temperature_2m_max", d).confidence_weight for d in range(1, 8)]

    assert weights[0] == 1.0
    assert weights == sorted(weights, reverse=True), weights
    # 47% detection at day 7 against 85% at day 1
    assert weights[-1] == pytest.approx(0.4704545 / 0.8466667, rel=1e-3)


def test_precipitation_weight_takes_the_running_minimum_not_the_raw_row():
    """The measured precipitation hit rate ticks UP from day 6 (1.7%) to day 7 (3.5%).

    On ~458 extreme days that is sampling noise, not a forecast that improved
    with lead time, so the weight must not tick up with it: confidence has to be
    monotone in the horizon.
    """
    day6, day7 = (skill_for("precipitation_sum", d).extreme_hit_rate for d in (6, 7))
    assert day7 > day6, "table changed: this test guards the wrong-way step"

    w6 = forecast_skill("precipitation_sum", 6).confidence_weight
    w7 = forecast_skill("precipitation_sum", 7).confidence_weight
    assert w7 == w6


def test_weight_is_full_when_the_table_measured_no_hit_rate(tmp_path):
    """A gap in OUR measurement is never a reason to penalise the forecast."""
    synthetic = {
        "schema_version": 1,
        "provenance": {"model": "gfs_global", "city_count": 13},
        "hazards": {
            "temperature_2m_max": {
                "unit": "°C",
                "lead_days": {
                    str(d): {
                        "mae": float(d), "bias": 0.1, "rmse": float(d) * 2,
                        "extreme_hit_rate": None, "hourly_repeat_fraction": 0.2,
                        "n_days": 10, "n_extreme_days": 0, "n_cities": 1,
                    }
                    for d in range(1, 8)
                },
            }
        },
    }
    path = tmp_path / "no_rates.json"
    path.write_text(json.dumps(synthetic), encoding="utf-8")

    skill = forecast_skill("temperature_2m_max", 7, path=path)

    assert skill.confidence_weight == 1.0
    assert "too few extreme days" in skill.detail()


def test_raw_per_city_output_is_gitignored():
    """data/skill/ holds the bulky per-city results and must not be committed."""
    gitignore = (Path(__file__).resolve().parent.parent / ".gitignore").read_text(encoding="utf-8")
    assert any(line.strip() in {"data/", "data"} for line in gitignore.splitlines())
