"""Tests for tools/geocode.py — Open-Meteo geocoding, mocked HTTP (offline)."""
import pytest

from tools.geocode import GeocodeError, GeoLocation, geocode


@pytest.fixture(autouse=True)
def _fresh_cache():
    geocode.cache_clear()  # lru_cache would serve test 1's result to test 2
    yield
    geocode.cache_clear()

_CANNED = {
    "results": [
        {
            "name": "Rourkela",
            "latitude": 22.24975,
            "longitude": 84.88286,
            "country": "India",
            "admin1": "Odisha",
        }
    ]
}


def test_geocode_returns_typed_location(httpx_mock):
    httpx_mock.add_response(json=_CANNED)

    loc = geocode("Rourkela")

    assert isinstance(loc, GeoLocation)
    assert loc.name == "Rourkela"
    assert loc.country == "India"
    assert 22 < loc.latitude < 23 and 84 < loc.longitude < 85


def test_geocode_hits_the_pinned_host_only(httpx_mock):
    httpx_mock.add_response(json=_CANNED)

    geocode("Rourkela")

    request = httpx_mock.get_requests()[0]
    assert request.url.host == "geocoding-api.open-meteo.com"  # SSRF pin


def test_unknown_place_is_a_typed_error(httpx_mock):
    httpx_mock.add_response(json={"results": []})
    with pytest.raises(GeocodeError, match="no match"):
        geocode("Xyzzyville-Nowhere")


def test_missing_results_key_is_a_typed_error(httpx_mock):
    httpx_mock.add_response(json={})
    with pytest.raises(GeocodeError, match="no match"):
        geocode("???")


def test_http_failure_is_a_typed_error(httpx_mock):
    httpx_mock.add_response(status_code=500)
    with pytest.raises(GeocodeError, match="request failed"):
        geocode("Rourkela")
