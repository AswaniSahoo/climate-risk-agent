"""lat/lon -> IPCC AR6 reference region, via the official Iturbide et al. (2020)
polygons (bundled by regionmask — real geometry, never hand-drawn boxes).

Why this matters beyond correctness: the RAG layer's table chunks are anchored
to the exact vocabulary "South Asia (SAS)" (rag/chunk._AR6_REGIONS). Mapping a
point to that same string means the research question names the region the way
the corpus spells it — retrieval of the right table row becomes deterministic
instead of hoping a city name bridges semantically.

The polygon data downloads once (~1 MB, cached locally / baked into Docker at
build). Ocean points and unknown regions return None — the caller degrades to
city-name-only retrieval, loudly.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from pydantic import BaseModel

from rag.chunk import _AR6_REGIONS

_log = logging.getLogger(__name__)


class AR6Region(BaseModel):
    """One resolved AR6 reference region, in the corpus's own vocabulary."""

    acronym: str  # e.g. "SAS"
    name: str  # e.g. "South Asia" — matches the table-row chunks verbatim

    @property
    def label(self) -> str:
        return f"{self.name} ({self.acronym})"


@lru_cache(maxsize=1)
def _land_regions():
    import regionmask

    return regionmask.defined_regions.ar6.land


@lru_cache(maxsize=512)
def region_for(latitude: float, longitude: float) -> AR6Region | None:
    """The AR6 land region containing this point, or None (ocean / load failure)."""
    from shapely.geometry import Point

    try:
        regions = _land_regions()
    except Exception as exc:  # data download/parse failure -> degrade loudly
        _log.warning("AR6 region data unavailable (%s) — proceeding without region", exc)
        return None

    point = Point(longitude, latitude)
    for abbrev, fallback_name, polygon in zip(
        regions.abbrevs, regions.names, regions.polygons
    ):
        if polygon.contains(point):
            # our chunker's full names ("South Asia"), not regionmask's
            # abbreviated ones ("S.Asia") — the corpus spelling wins
            return AR6Region(acronym=abbrev, name=_AR6_REGIONS.get(abbrev, fallback_name))
    return None
