"""Tests for lat/lon -> AR6 region mapping (official Iturbide polygons).

The polygons ship with the repo (tools/ar6/*.geojson), so these tests are fully
offline: nothing downloads, and a missing data file is a hard failure rather
than a skip — the file being absent from the image is exactly the bug worth
catching.
"""
import json

import pytest

from tools.ar6_regions import _GEOJSON, AR6Region, _land_regions, region_for


@pytest.fixture(autouse=True)
def _clear_cache():
    region_for.cache_clear()


def test_geojson_is_bundled_parses_and_has_the_expected_land_set():
    assert _GEOJSON.is_file(), f"AR6 polygon file missing at {_GEOJSON}"
    collection = json.loads(_GEOJSON.read_text(encoding="utf-8"))
    # The Atlas v4 file: 58 reference regions, 43 Land + 3 Land-Ocean + 12 Ocean.
    assert len(collection["features"]) == 58
    # 46 land-touching regions — the same count regionmask's ar6.land exposed.
    regions = _land_regions()
    assert len(regions) == 46
    assert all(acronym and name and poly is not None for acronym, name, poly in regions)
    assert len({acronym for acronym, _, _ in regions}) == 46  # no duplicates


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
