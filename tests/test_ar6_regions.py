"""Tests for lat/lon -> AR6 region mapping (official Iturbide polygons).

Needs the regionmask data file (downloads ~1 MB once, then cached); tests skip
if it cannot load, so an offline CI run stays green without faking geometry.
"""
import pytest

from tools.ar6_regions import AR6Region, region_for


@pytest.fixture(autouse=True)
def _needs_region_data():
    region_for.cache_clear()
    try:
        from tools.ar6_regions import _land_regions

        _land_regions()
    except Exception as exc:  # offline / download blocked
        pytest.skip(f"AR6 region data unavailable: {exc}")


@pytest.mark.parametrize(
    ("lat", "lon", "acronym", "name"),
    [
        (22.26, 84.85, "SAS", "South Asia"),  # Rourkela
        (52.52, 13.40, "WCE", "Western and Central Europe"),  # Berlin
        (40.70, -74.00, "ENA", "Eastern North America"),  # New York
        (-33.90, 151.20, "EAU", "Eastern Australia"),  # Sydney
    ],
)
def test_known_cities_map_to_expected_regions(lat, lon, acronym, name):
    region = region_for(lat, lon)
    assert isinstance(region, AR6Region)
    assert region.acronym == acronym
    assert region.name == name  # the CHUNKER's spelling — retrieval-critical
    assert region.label == f"{name} ({acronym})"


def test_open_ocean_returns_none():
    assert region_for(0.0, -140.0) is None  # equatorial Pacific


def test_region_name_matches_corpus_vocabulary():
    from rag.chunk import _AR6_REGIONS

    region = region_for(22.26, 84.85)
    assert region is not None
    assert region.name == _AR6_REGIONS[region.acronym]
