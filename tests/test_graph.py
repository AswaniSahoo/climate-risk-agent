"""Tests for the 4-node LangGraph agent (agent/graph.py).

run_agent wires plan -> call get_forecast -> research (IPCC RAG) -> synthesize
into one call that returns a RiskReport. HTTP is mocked (pytest-httpx) and the
IPCC retriever is stubbed offline by default, so these are deterministic. The
wind case proves the refusal path short-circuits BEFORE any forecast call.
"""
import json

import pytest

import agent.graph as graph_mod
import tools.forecast_skill as skill_mod
from agent.contracts import Citation, Hazard, RiskLevel, RiskReport
from agent.graph import run_agent
from rag.answer import CitedAnswer
from rag.chunk import Chunk
from rag.corpus import CorpusError
from tools.climatology import build_hazard_stat
from tools.forecast_skill import forecast_skill, skill_for
from tools.hazard_stats import HazardStat, Representativeness, ReturnLevel, TrendInfo


@pytest.fixture(autouse=True)
def _offline_ipcc(monkeypatch):
    """Default: no corpus on disk -> research degrades loudly, report still ships."""
    def no_corpus():
        raise CorpusError("offline test: no corpus")

    monkeypatch.setattr(graph_mod, "_ipcc_retriever", no_corpus)


@pytest.fixture(autouse=True)
def _isolated_forecast_cache(tmp_path, monkeypatch):
    """Forecast cache -> tmp_path.

    The real one is a repo directory with a one-hour TTL, so a second run of
    this file inside the hour would be served from disk and leave the mocked
    response unused. Per-test isolation makes every run identical.
    """
    from tools.cache_backend import DiskCache, JsonCache

    monkeypatch.setattr(
        "tools.forecast._forecast_cache",
        lambda: JsonCache("forecast", backend=DiskCache(tmp_path / "fc")),
    )

# ~20 years of (illustrative) annual-max 2m_temperature in Kelvin.
_HEAT_MAXIMA = [
    309.6, 310.5, 308.6, 311.0, 307.9, 312.1, 309.0, 310.2, 308.4, 311.5,
    309.8, 307.5, 310.9, 308.1, 312.4, 309.3, 310.7, 308.8, 311.2, 309.5,
]

CANNED = {
    "latitude": 22.26,
    "longitude": 84.85,
    "timezone": "Asia/Kolkata",
    "daily_units": {
        "time": "iso8601",
        "precipitation_sum": "mm",
        "temperature_2m_max": "°C",
        "wind_speed_10m_max": "km/h",
        "wind_gusts_10m_max": "km/h",
    },
    "daily": {
        "time": ["2026-07-02", "2026-07-03", "2026-07-04"],
        # Peaks below are the FALLBACK (absolute-cutoff) bands, which is what a
        # run without an injected HazardStat lands on.
        "precipitation_sum": [14.2, 0.1, 80.0],     # max 80 mm    -> HIGH
        "temperature_2m_max": [46.0, 44.0, 40.0],   # max 46 °C    -> SEVERE
        "wind_speed_10m_max": [70.0, 45.0, 30.0],   # max 70 km/h  -> HIGH (sustained)
        "wind_gusts_10m_max": [95.0, 60.0, 42.0],   # max 95 km/h gust: the GEV metric
    },
}


def test_extreme_precip_query_yields_high_risk_report(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3,
    )

    assert isinstance(report, RiskReport)
    assert report.refusal is None
    assert report.risk_level is RiskLevel.HIGH
    assert report.provenance and report.provenance[0].source == "Open-Meteo"


def test_heatwave_query_yields_severe_risk_report(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3,
    )

    assert report.hazard is Hazard.HEATWAVE
    assert report.risk_level is RiskLevel.SEVERE


def test_wind_query_yields_high_risk_report(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.WIND, horizon_days=3,
    )

    assert report.refusal is None
    assert report.risk_level is RiskLevel.HIGH
    assert report.drivers[0].factor == "wind"
    assert "m/s" in report.drivers[0].detail  # km/h also shown in m/s
    # No climatology injected -> the absolute-cutoff fallback, which says so and
    # grades the SUSTAINED 70 km/h (what those Beaufort-derived cutoffs mean),
    # not the 95 km/h gust.
    basis = _severity_basis(report)
    assert "no ERA5 climatology is available for this location" in basis
    assert "fixed absolute cutoffs" in basis
    assert "sustained 10 m wind 70 km/h grades high (62 to 88 km/h)" in basis


def test_unsupported_hazard_takes_refusal_path_without_forecast(monkeypatch):
    # Simulate a hazard with no data path. No httpx mock on purpose: if the graph
    # wrongly calls get_forecast, this test fails.
    monkeypatch.setattr("agent.graph._ANSWERABLE", {Hazard.HEATWAVE})

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.WIND, horizon_days=3,
    )

    assert report.refusal is not None
    assert report.risk_level is None


def test_heatwave_report_includes_injected_hazard_stats(httpx_mock):
    httpx_mock.add_response(json=CANNED)
    hs = build_hazard_stat(
        list(range(2003, 2023)), _HEAT_MAXIMA, hazard=Hazard.HEATWAVE,
        latitude=22.26, longitude=84.85, timezone="Asia/Kolkata",
    )

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3, hazard_stat=hs,
    )

    assert report.hazard_stats and report.hazard_stats[0].n_years == len(_HEAT_MAXIMA)
    assert report.confidence > 0.3  # climatology grounding beats the raw heuristic
    assert "return level" in report.summary.lower()


def _stat(variable: str, unit: str, two, ten, fifty, hundred) -> HazardStat:
    """A HazardStat carrying the four fitted return levels the bands need."""
    return HazardStat(
        variable=variable, statistic_definition=f"annual max {variable}",
        unit=unit, source="test", model="era5", native_resolution_deg=0.25,
        captures_diurnal_peak=True, timezone="Asia/Kolkata",
        latitude=22.26, longitude=84.85, n_years=63,
        record_start_year=1960, record_end_year=2022, record_max=hundred,
        return_levels=[
            ReturnLevel(return_period_years=2, level=two),
            ReturnLevel(return_period_years=10, level=ten),
            ReturnLevel(return_period_years=50, level=fifty),
            ReturnLevel(return_period_years=100, level=hundred),
        ],
        is_bias_corrected=False,
        representativeness=Representativeness.POINT_INTERPOLATED_REANALYSIS,
        interpretation="test fixture",
    )


def _precip_stat(two, ten, fifty, hundred) -> HazardStat:
    return _stat("precipitation_sum", "mm", two, ten, fifty, hundred)


def _severity_basis(report) -> str:
    return next(d.detail for d in report.drivers if d.factor == "severity_basis")


def test_gev_verdict_overrides_absolute_thresholds(httpx_mock):
    # 80 mm peak = "HIGH" on the absolute Day-1 scale, but at a wet-climate
    # location where an ordinary year already delivers 85 mm it is LOW.
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3,
        hazard_stat=_precip_stat(85.0, 100.0, 140.0, 160.0),
    )

    assert report.risk_level is RiskLevel.LOW  # location-relative, not absolute
    basis = _severity_basis(report)
    assert "below the 2-year level (85.0 mm)" in basis
    assert "ordinary year" in basis


def test_band_between_two_fitted_levels_reports_the_bracket_and_period(httpx_mock):
    # 80 mm sits between the 2-yr (60) and 10-yr (100) levels -> MODERATE, and the
    # explanation must carry the bracket AND the interpolated return period.
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3,
        hazard_stat=_precip_stat(60.0, 100.0, 140.0, 160.0),
    )

    assert report.risk_level is RiskLevel.MODERATE
    basis = _severity_basis(report)
    assert "between the 2- and 10-year levels (60.0 / 100.0 mm)" in basis
    assert "1-in-4-year event" in basis  # log-interpolated: exp(ln2 + 0.5*(ln10-ln2))


def test_heatwave_band_comes_from_the_local_curve(httpx_mock):
    # 46 °C peak against a hot-climate curve: between the 10-yr (44) and 50-yr
    # (47) levels -> HIGH, not the SEVERE the absolute 45 °C cutoff would give.
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3,
        hazard_stat=_stat("temperature_2m_max", "°C", 38.0, 44.0, 47.0, 49.0),
    )

    assert report.risk_level is RiskLevel.HIGH
    assert "between the 10- and 50-year levels (44.0 / 47.0 °C)" in _severity_basis(report)


def test_wind_band_compares_the_gust_against_the_gust_climatology(httpx_mock):
    # The DEBT fix: the forecast GUST (95 km/h) is what gets compared to the
    # ERA5 gust fit. Comparing the 70 km/h sustained speed instead would land a
    # band lower and understate the risk.
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.WIND, horizon_days=3,
        hazard_stat=_stat("wind_gusts_10m_max", "km/h", 70.0, 90.0, 110.0, 125.0),
    )

    assert report.risk_level is RiskLevel.HIGH
    basis = _severity_basis(report)
    assert "Forecast peak 95 km/h" in basis
    assert "between the 10- and 50-year levels (90.0 / 110.0 km/h)" in basis
    assert "absolute cutoffs" not in basis  # the climatology path, not the fallback
    assert "max daily gust 95.0 km/h" in report.drivers[0].detail
    assert "max daily sustained 70.0 km/h" in report.drivers[0].detail


def test_significant_trend_surfaces_in_summary_and_driver(httpx_mock):
    # When the stat carries a significant warming trend, the report must SAY
    # the levels are effective (evaluated at the latest year), not 60-yr averages.
    httpx_mock.add_response(json=CANNED)
    stat = _precip_stat(60.0, 100.0, 140.0, 160.0).model_copy(update={
        "trend": TrendInfo(slope_per_decade=2.5, p_value=0.003,
                           significant=True, evaluated_at_year=2022),
    })

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3, hazard_stat=stat,
    )

    clim = [d for d in report.drivers if d.factor == "climatology"][0]
    assert "non-stationary" in clim.detail
    assert "2022" in report.summary and "trend" in report.summary.lower()


def test_insignificant_trend_reports_the_test_ran(httpx_mock):
    httpx_mock.add_response(json=CANNED)
    stat = _precip_stat(60.0, 100.0, 140.0, 160.0).model_copy(update={
        "trend": TrendInfo(slope_per_decade=0.4, p_value=0.61,
                           significant=False, evaluated_at_year=None),
    })

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3, hazard_stat=stat,
    )

    clim = [d for d in report.drivers if d.factor == "climatology"][0]
    assert "no significant trend" in clim.detail
    assert "p=0.61" in clim.detail


def test_gev_verdict_flags_record_class_event(httpx_mock):
    # Same 80 mm peak at a dry-climate location whose 100-yr event is 70 mm -> SEVERE.
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3,
        hazard_stat=_precip_stat(25.0, 40.0, 60.0, 70.0),
    )

    assert report.risk_level is RiskLevel.SEVERE


# --- research node: IPCC RAG -> real page-level Citations in the report ---

_IPCC_CHUNKS = (
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#2", source="IPCC_AR6_WGI_Chapter11.pdf",
          page=124, text="East Central Asia (ECA) ... median increase of more than 3.5°C"),
    Chunk(chunk_id="IPCC_AR6_WGI_Chapter11.pdf#p124#1", source="IPCC_AR6_WGI_Chapter11.pdf",
          page=124, text="East Central Asia (ECA) significant increases in hot extremes"),
    Chunk(chunk_id="IPCC_AR6_WGI_SPM.pdf#p16#0", source="IPCC_AR6_WGI_SPM.pdf",
          page=16, text="hot extremes have become more frequent and more intense"),
)


class _FakeRetriever:
    def retrieve(self, question, top_k):
        return list(_IPCC_CHUNKS)[:top_k]


def _grounded_ipcc(monkeypatch, answer: CitedAnswer):
    monkeypatch.setattr(graph_mod, "_ipcc_retriever", lambda: _FakeRetriever())
    monkeypatch.setattr(graph_mod, "answer_with_guard", lambda q, chunks, **_: answer)


def test_report_carries_page_level_citations_from_cited_answer(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=CANNED)
    _grounded_ipcc(monkeypatch, CitedAnswer(
        answer="Hot extremes are projected to intensify by more than 3.5°C.",
        citations=["IPCC_AR6_WGI_Chapter11.pdf#p124#2",
                   "IPCC_AR6_WGI_Chapter11.pdf#p124#1",   # same page -> must dedupe
                   "IPCC_AR6_WGI_SPM.pdf#p16#0"],
        abstain=False,
        allowed_ids=[c.chunk_id for c in _IPCC_CHUNKS],
    ))

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3,
    )

    assert Citation(source="IPCC_AR6_WGI_Chapter11.pdf", locator="p124") in report.citations
    assert Citation(source="IPCC_AR6_WGI_SPM.pdf", locator="p16") in report.citations
    assert len(report.citations) == 2  # one Citation per (source, page), not per chunk
    assert "3.5°C" in report.summary  # cited IPCC finding lands in the summary


def test_abstaining_answer_adds_no_citations(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=CANNED)
    _grounded_ipcc(monkeypatch, CitedAnswer(
        answer="", citations=[], abstain=True,
        allowed_ids=[c.chunk_id for c in _IPCC_CHUNKS],
    ))

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3,
    )

    assert report.citations == []  # honest abstention -> no decorative citations
    assert report.risk_level is RiskLevel.SEVERE  # forecast half still works


def test_offline_corpus_degrades_loudly_report_still_ships(httpx_mock, caplog):
    # autouse fixture already makes _ipcc_retriever raise CorpusError
    httpx_mock.add_response(json=CANNED)

    report = run_agent(
        location="Rourkela", latitude=22.26, longitude=84.85,
        hazard=Hazard.HEATWAVE, horizon_days=3,
    )

    assert report.citations == []
    assert report.risk_level is RiskLevel.SEVERE
    assert "IPCC grounding unavailable" in caplog.text  # loud (WARNING), never silent


# --- forecast skill: a day-7 peak is a weaker claim than a day-1 one --------


def _skill_state(hazard: Hazard, horizon_days: int) -> dict:
    return {
        "location": "Rourkela", "latitude": 22.26, "longitude": 84.85,
        "hazard": hazard, "horizon_days": horizon_days,
    }


def _skill_driver(report) -> str:
    return next(d.detail for d in report.drivers if d.factor == "forecast_skill")


@pytest.mark.parametrize(
    "hazard, variable, horizon, pct",
    [
        (Hazard.HEATWAVE, "temperature_2m_max", 1, 85),
        (Hazard.HEATWAVE, "temperature_2m_max", 7, 47),
        (Hazard.EXTREME_PRECIP, "precipitation_sum", 1, 53),
        (Hazard.EXTREME_PRECIP, "precipitation_sum", 7, 3),
        (Hazard.WIND, "wind_gusts_10m_max", 1, 69),
        (Hazard.WIND, "wind_gusts_10m_max", 7, 34),
    ],
)
def test_forecast_node_attaches_the_measured_row_for_the_hazards_own_variable(
    httpx_mock, hazard, variable, horizon, pct
):
    # Wind reads the GUST row, because the gust is the quantity it is graded on.
    httpx_mock.add_response(json=CANNED)

    skill = graph_mod.call(_skill_state(hazard, horizon))["forecast"].skill

    assert skill is not None
    assert skill.variable == variable
    assert skill.lead_day == horizon and skill.requested_horizon_days == horizon
    assert skill.extrapolated is False
    assert skill.extreme_hit_rate == skill_for(variable, horizon).extreme_hit_rate
    assert round(skill.extreme_hit_rate * 100) == pct
    assert "13 cities" in skill.source and "2024-2025" in skill.source


def test_skill_past_the_measured_archive_is_clamped_and_flagged(httpx_mock):
    # The archive stops at lead day 7; the forecast tool accepts 16.
    httpx_mock.add_response(json=CANNED)

    skill = graph_mod.call(_skill_state(Hazard.HEATWAVE, 10))["forecast"].skill

    assert skill.requested_horizon_days == 10
    assert skill.lead_day == 7
    assert skill.extrapolated is True
    assert skill.confidence_weight == forecast_skill("temperature_2m_max", 7).confidence_weight


def test_report_driver_quotes_the_measured_hit_rate(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(**_skill_state(Hazard.HEATWAVE, 7))

    detail = _skill_driver(report)
    assert "47%" in detail  # the measured number, in the report, in words
    assert "GFS Global" in detail and "13 cities" in detail and "2024-2025" in detail
    # 0.3 * (0.4705 / 0.8467), no climatology and no citations to add to it
    assert report.confidence == 0.17


def test_extrapolated_driver_says_the_day_7_value_is_being_reused(httpx_mock):
    httpx_mock.add_response(json=CANNED)

    report = run_agent(**_skill_state(Hazard.HEATWAVE, 10))

    detail = _skill_driver(report)
    assert "past the measured archive" in detail
    assert "day-7 figure is reused" in detail
    assert "47%" in detail


def test_confidence_never_rises_with_the_horizon(httpx_mock):
    """The property that must hold for every hazard: further out is never surer."""
    httpx_mock.add_response(json=CANNED, is_reusable=True)

    confidences = [
        run_agent(**_skill_state(Hazard.HEATWAVE, h)).confidence for h in range(1, 17)
    ]

    assert confidences[0] == 0.3  # day 1 = the pre-skill number, unchanged
    assert max(confidences) <= 0.3  # nothing scores above the old maximum
    assert all(b <= a for a, b in zip(confidences, confidences[1:])), confidences
    assert confidences[-1] < confidences[0]


def test_confidence_keeps_the_flat_term_when_the_table_lacks_the_variable(
    httpx_mock, tmp_path, monkeypatch, caplog
):
    """A gap in our own measurement must not silently penalise the report."""
    thin = {
        "schema_version": 1,
        "provenance": {"model": "gfs_global", "city_count": 13},
        # precipitation only: a heatwave run finds no row for its variable
        "hazards": {"precipitation_sum": skill_mod.load_skill_table().hazards["precipitation_sum"]},
    }
    path = tmp_path / "thin_table.json"
    path.write_text(json.dumps(thin), encoding="utf-8")
    monkeypatch.setattr(skill_mod, "TABLE_PATH", path)
    httpx_mock.add_response(json=CANNED)

    report = run_agent(**_skill_state(Hazard.HEATWAVE, 7))

    assert report.confidence == 0.3  # exactly the old, lead-blind number
    assert all(d.factor != "forecast_skill" for d in report.drivers)
    assert "no measured forecast skill" in caplog.text  # loud, never silent

