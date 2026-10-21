"""Tests for scripts/measure_warming_trend.py (offline: geocode and the fit are stubbed).

The artifact exists so a README trend number has a file behind it, so what is
tested is that every row carries the point it was measured at and the exact
slope and p-value the app's HazardStat reported, unrounded.
"""
import pytest

from agent.contracts import Hazard
from scripts.measure_warming_trend import build_artifact, parse_point
from tools.geocode import GeoLocation
from tools.hazard_stats import TrendInfo


class _Stat:
    variable = "temperature_2m_max"
    unit = "°C"
    record_start_year = 1960
    record_end_year = 2022
    n_years = 63

    def __init__(self, trend):
        self.trend = trend


def test_rows_carry_the_point_and_the_unrounded_trend_the_app_reports():
    asked = []

    def hazard_stat(lat, lon, hazard):
        asked.append((lat, lon, hazard))
        if lat > 50:
            return _Stat(TrendInfo(slope_per_decade=0.7526, p_value=2.86e-7,
                                   significant=True, evaluated_at_year=2022))
        return _Stat(TrendInfo(slope_per_decade=0.0593, p_value=0.459, significant=False))

    artifact = build_artifact(
        ["Berlin"],
        [("Delhi (UI preset)", 28.61, 77.21)],
        geocode=lambda place: GeoLocation(name=place, latitude=52.52437,
                                          longitude=13.41053, country="Germany"),
        hazard_stat=hazard_stat,
        code_fingerprint="abc123",
        fit_cache="bypassed",
    )

    berlin, delhi = artifact["results"]
    assert asked == [(52.52437, 13.41053, Hazard.HEATWAVE), (28.61, 77.21, Hazard.HEATWAVE)]
    assert berlin["location"] == "Berlin, Germany"
    assert berlin["resolved_by"] == "tools.geocode.geocode"
    assert berlin["slope_per_decade"] == 0.7526 and berlin["p_value"] == 2.86e-7
    assert berlin["significant"] is True and berlin["evaluated_at_year"] == 2022
    assert delhi["resolved_by"] == "fixed coordinates"
    assert delhi["significant"] is False and delhi["evaluated_at_year"] is None
    assert artifact["code_version"]["climatology_fingerprint"] == "abc123"
    assert artifact["fit_cache"] == "bypassed"


def test_a_stat_with_no_trend_fit_is_recorded_as_none_not_zero():
    artifact = build_artifact(
        [], [("short record", 1.0, 2.0)],
        geocode=lambda place: pytest.fail("no places given"),
        hazard_stat=lambda lat, lon, hazard: _Stat(None),
        code_fingerprint="x", fit_cache="bypassed",
    )

    [row] = artifact["results"]
    assert row["slope_per_decade"] is None and row["p_value"] is None


def test_parse_point_reads_label_and_coordinates():
    assert parse_point("Delhi (UI preset)=28.61,77.21") == ("Delhi (UI preset)", 28.61, 77.21)
