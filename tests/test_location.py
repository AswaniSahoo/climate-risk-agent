"""Tests for agent/location.py — any point on Earth -> the parts a report needs.

Geocoding HTTP is mocked (pytest-httpx) and the AR6 lookup is stubbed, so these
stay offline and deterministic. The label/notice rules are pure functions on
purpose: the Streamlit script cannot be unit-tested, this can.
"""
import pytest

import agent.location as loc_mod
from agent.location import (
    ResolvedPlace,
    coordinate_error,
    format_coordinates,
    name_still_applies,
    resolve_coordinates,
    resolve_place,
)
from tools.ar6_regions import AR6Region
from tools.geocode import GeocodeError, geocode

_SAS = AR6Region(acronym="SAS", name="South Asia")

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


@pytest.fixture(autouse=True)
def _fresh_geocode_cache():
    geocode.cache_clear()  # lru_cache would serve test 1's result to test 2
    yield
    geocode.cache_clear()


def test_resolve_place_returns_coordinates_and_region(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=_CANNED)
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: _SAS)

    place = resolve_place("Rourkela")

    assert isinstance(place, ResolvedPlace)
    assert place.country == "India"
    assert place.latitude == pytest.approx(22.24975)
    assert place.longitude == pytest.approx(84.88286)
    # the corpus's own vocabulary -> deterministic table-row retrieval
    assert place.label == "Rourkela, South Asia (SAS)"
    assert place.notice is None


def test_resolve_place_hits_the_pinned_geocoding_host(httpx_mock, monkeypatch):
    httpx_mock.add_response(json=_CANNED)
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: _SAS)

    resolve_place("Rourkela")

    assert httpx_mock.get_requests()[0].url.host == "geocoding-api.open-meteo.com"


def test_unresolvable_place_raises_the_typed_geocode_error(httpx_mock, monkeypatch):
    httpx_mock.add_response(json={"results": []})
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: _SAS)

    with pytest.raises(GeocodeError, match="no match"):
        resolve_place("Xyzzyville-Nowhere")


def test_label_falls_back_to_country_when_there_is_no_ar6_region():
    place = ResolvedPlace(
        name="Rourkela", latitude=22.25, longitude=84.88, country="India"
    )

    assert place.label == "Rourkela, India"
    assert "(" not in place.label  # no region acronym to claim


def test_point_outside_every_ar6_region_carries_an_honest_notice():
    ocean = ResolvedPlace(name="0.0000°N, 140.0000°W", latitude=0.0, longitude=-140.0)

    assert ocean.notice is not None
    assert "IPCC" in ocean.notice
    assert "ocean" in ocean.notice.lower()
    assert ocean.label == "0.0000°N, 140.0000°W"  # no country -> no dangling comma


@pytest.mark.parametrize(
    ("lat", "lon", "expected"),
    [
        (22.26, 84.85, "22.2600°N, 84.8500°E"),
        (-33.87, 151.21, "33.8700°S, 151.2100°E"),  # southern hemisphere
        (40.71, -74.01, "40.7100°N, 74.0100°W"),  # western hemisphere
        (0.0, 0.0, "0.0000°N, 0.0000°E"),
    ],
)
def test_format_coordinates_names_the_hemisphere(lat, lon, expected):
    assert format_coordinates(lat, lon) == expected


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(91.0, 0.0), (-90.1, 0.0), (0.0, 181.0), (0.0, -180.5)],
)
def test_out_of_range_coordinates_get_an_error_message(lat, lon):
    message = coordinate_error(lat, lon)
    assert message is not None and "range" in message


def test_in_range_coordinates_have_no_error():
    assert coordinate_error(-33.87, -70.67) is None  # Santiago: both negative


def test_resolve_coordinates_names_the_point_and_maps_the_region(monkeypatch):
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: _SAS)

    place = resolve_coordinates(22.26, 84.85)

    assert place.name == "22.2600°N, 84.8500°E"
    assert place.label == "22.2600°N, 84.8500°E, South Asia (SAS)"
    assert place.notice is None


def test_resolve_coordinates_keeps_a_supplied_name(monkeypatch):
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: _SAS)

    place = resolve_coordinates(22.26, 84.85, name="Rourkela")

    assert place.name == "Rourkela"


def test_resolve_coordinates_keeps_the_country_for_the_no_region_label(monkeypatch):
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: None)

    place = resolve_coordinates(22.26, 84.85, name="Rourkela", country="India")

    assert place.label == "Rourkela, India"
    assert place.notice is not None  # no region -> the caveat still shows


def test_resolve_coordinates_rejects_out_of_range_at_the_boundary(monkeypatch):
    monkeypatch.setattr(loc_mod, "region_for", lambda lat, lon: None)

    with pytest.raises(ValueError, match="latitude"):
        resolve_coordinates(120.0, 0.0)


def test_a_remembered_name_survives_its_own_point():
    assert name_still_applies((22.26, 84.85), 22.26, 84.85)


@pytest.mark.parametrize(
    ("named_at", "lat", "lon"),
    [
        (None, 22.26, 84.85),  # nothing was ever named
        ((22.26, 84.85), 0.0, -140.0),  # dragged into the Pacific
        ((22.26, 84.85), 22.26, -84.85),  # sign flipped: a different place
    ],
)
def test_a_remembered_name_is_dropped_once_the_point_moves(named_at, lat, lon):
    assert not name_still_applies(named_at, lat, lon)
