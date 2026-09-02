"""Any-location coverage: the agent must work off the India-shaped happy path.

Southern hemisphere, western hemisphere and open ocean all run the SAME graph
with Open-Meteo mocked. The outbound request is inspected on purpose: a dropped
minus sign is the classic geo bug, and it silently relocates a report to the
wrong continent. It must fail here, not in production.
"""
import pytest

import agent.graph as graph_mod
from agent.contracts import Hazard
from agent.graph import run_agent
from agent.location import ResolvedPlace
from rag.corpus import CorpusError
from tools.ar6_regions import region_for

# Sydney (southern), New York (western), mid-Pacific (no AR6 land region). The
# first two are the exact points tests/test_ar6_regions.py already pins.
SYDNEY = (-33.90, 151.20)
NEW_YORK = (40.70, -74.00)
MID_PACIFIC = (0.0, -140.0)


@pytest.fixture(autouse=True)
def _offline_ipcc(monkeypatch):
    """No corpus -> research degrades loudly, the report still ships."""

    def no_corpus():
        raise CorpusError("offline test: no corpus")

    monkeypatch.setattr(graph_mod, "_ipcc_retriever", no_corpus)


def _canned(latitude: float, longitude: float) -> dict:
    return {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "UTC",
        "daily_units": {
            "time": "iso8601",
            "precipitation_sum": "mm",
            "temperature_2m_max": "°C",
            "wind_speed_10m_max": "km/h",
            "wind_gusts_10m_max": "km/h",
        },
        "daily": {
            "time": ["2026-09-02", "2026-09-03", "2026-09-04"],
            "precipitation_sum": [2.0, 0.0, 1.0],
            "temperature_2m_max": [31.0, 29.0, 30.0],
            "wind_speed_10m_max": [22.0, 18.0, 25.0],
            "wind_gusts_10m_max": [38.0, 30.0, 41.0],
        },
    }


def _outbound_coordinates(httpx_mock) -> tuple[str, str]:
    """The latitude/longitude EXACTLY as they left the process, still strings."""
    params = httpx_mock.get_requests()[0].url.params
    return params["latitude"], params["longitude"]


def test_southern_hemisphere_point_sends_a_negative_latitude(httpx_mock):
    latitude, longitude = SYDNEY
    httpx_mock.add_response(json=_canned(latitude, longitude))

    report = run_agent(
        location="Sydney", latitude=latitude, longitude=longitude,
        hazard=Hazard.HEATWAVE, horizon_days=3,
    )

    sent_lat, sent_lon = _outbound_coordinates(httpx_mock)
    assert sent_lat.startswith("-")  # the sign survived the round trip
    assert float(sent_lat) == pytest.approx(latitude)
    assert float(sent_lon) == pytest.approx(longitude)
    assert report.refusal is None and report.risk_level is not None


def test_western_hemisphere_point_sends_a_negative_longitude(httpx_mock):
    latitude, longitude = NEW_YORK
    httpx_mock.add_response(json=_canned(latitude, longitude))

    report = run_agent(
        location="New York", latitude=latitude, longitude=longitude,
        hazard=Hazard.EXTREME_PRECIP, horizon_days=3,
    )

    sent_lat, sent_lon = _outbound_coordinates(httpx_mock)
    assert sent_lon.startswith("-")
    assert float(sent_lat) == pytest.approx(latitude)
    assert float(sent_lon) == pytest.approx(longitude)
    assert report.refusal is None and report.risk_level is not None


def test_mid_pacific_point_still_produces_a_report(httpx_mock):
    """Open ocean: no AR6 region, but forecast + risk banding must still run."""
    latitude, longitude = MID_PACIFIC
    httpx_mock.add_response(json=_canned(latitude, longitude))

    report = run_agent(
        location="0.0000°N, 140.0000°W", latitude=latitude, longitude=longitude,
        hazard=Hazard.WIND, horizon_days=3,
    )

    sent_lat, sent_lon = _outbound_coordinates(httpx_mock)
    assert float(sent_lat) == pytest.approx(latitude)
    assert float(sent_lon) == pytest.approx(longitude)
    assert report.refusal is None  # "no IPCC region" is not a reason to refuse
    assert report.risk_level is not None
    assert report.provenance and report.provenance[0].source == "Open-Meteo"


def test_mid_pacific_point_says_regional_ipcc_context_is_unavailable():
    ocean = ResolvedPlace(name="0.0000°N, 140.0000°W", latitude=0.0, longitude=-140.0)

    assert ocean.notice is not None and "IPCC" in ocean.notice


@pytest.mark.parametrize(
    ("latitude", "longitude", "acronym"),
    [
        (*SYDNEY, "EAU"),  # Eastern Australia — southern hemisphere
        (*NEW_YORK, "ENA"),  # Eastern North America — western hemisphere
    ],
)
def test_hemisphere_cities_map_to_their_ar6_region(latitude, longitude, acronym):
    region_for.cache_clear()
    try:
        from tools.ar6_regions import _land_regions

        _land_regions()
    except Exception as exc:  # offline / download blocked — never fake geometry
        pytest.skip(f"AR6 region data unavailable: {exc}")

    region = region_for(latitude, longitude)

    assert region is not None
    assert region.acronym == acronym
