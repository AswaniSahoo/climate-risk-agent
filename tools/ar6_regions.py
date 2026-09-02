"""lat/lon -> IPCC AR6 reference region, via the official Iturbide et al. (2020)
polygons (real geometry, never hand-drawn boxes).

Why this matters beyond correctness: the RAG layer's table chunks are anchored
to the exact vocabulary "South Asia (SAS)" (rag/chunk._AR6_REGIONS). Mapping a
point to that same string means the research question names the region the way
the corpus spells it — retrieval of the right table row becomes deterministic
instead of hoping a city name bridges semantically.

The polygons are the Atlas GeoJSON bundled next to this module (680 KB, v4 is
frozen). They used to arrive via regionmask, which drags in geopandas +
rasterio + pyogrio + pyproj + xarray and downloaded the data at runtime.
Reading one file with shapely does the same job: no runtime download, no
geospatial stack, same answers (verified point-for-point against regionmask on
a 2-degree global grid, 0 disagreements in 16,020 points).

Ocean points and unknown regions return None — the caller degrades to
city-name-only retrieval, loudly.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from rag.chunk import _AR6_REGIONS

if TYPE_CHECKING:  # shapely stays a lazy import at runtime (image size)
    from shapely.geometry.base import BaseGeometry

_log = logging.getLogger(__name__)

_GEOJSON = Path(__file__).parent / "ar6" / "IPCC-WGI-reference-regions-v4.geojson"

# regionmask's ar6.land set was every feature the Atlas types as touching land:
# 43 "Land" + 3 "Land-Ocean" = 46. The Land-Ocean ones (CAR, SEA, ...) sit in
# regionmask's land AND ocean sets; the 12 pure "Ocean" ones never did.
_LAND_TYPES = frozenset({"Land", "Land-Ocean"})


class AR6Region(BaseModel):
    """One resolved AR6 reference region, in the corpus's own vocabulary."""

    acronym: str  # e.g. "SAS"
    name: str  # e.g. "South Asia" — matches the table-row chunks verbatim

    @property
    def label(self) -> str:
        return f"{self.name} ({self.acronym})"


@lru_cache(maxsize=1)
def _land_regions() -> tuple[tuple[str, str, BaseGeometry], ...]:
    """(acronym, Atlas name, shapely geometry) for the 46 land-touching regions.

    File order is Atlas id order, which is also regionmask's — first-match-wins
    below therefore resolves overlaps the same way it always did.
    """
    from shapely.geometry import shape

    with _GEOJSON.open(encoding="utf-8") as fh:
        collection = json.load(fh)
    return tuple(
        (props["Acronym"], props["Name"], shape(feature["geometry"]))
        for feature in collection["features"]
        for props in (feature["properties"],)
        if props["Type"] in _LAND_TYPES
    )


@lru_cache(maxsize=512)
def region_for(latitude: float, longitude: float) -> AR6Region | None:
    """The AR6 land region containing this point, or None (ocean / load failure)."""
    from shapely.geometry import Point

    try:
        regions = _land_regions()
    except Exception as exc:  # missing/corrupt data file -> degrade loudly
        _log.warning("AR6 region data unavailable (%s) — proceeding without region", exc)
        return None

    point = Point(longitude, latitude)
    for acronym, fallback_name, polygon in regions:
        if polygon.contains(point):
            # our chunker's full names ("South Asia"), not the Atlas's
            # abbreviated ones ("S.Asia") — the corpus spelling wins
            return AR6Region(acronym=acronym, name=_AR6_REGIONS.get(acronym, fallback_name))
    return None
