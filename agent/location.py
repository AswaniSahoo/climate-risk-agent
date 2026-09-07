"""Any point on Earth -> the parts a report needs: coordinates, a name, a region.

Three questions have to be answered about a place before the agent can run:
where is it, what do we call it in the report, and does the IPCC AR6 corpus
carry a *regional* assessment for it. Two front doors ask them — the free-text
NL path (agent/nl.py) and the UI's manual place/coordinate controls — so the
rules live here as pure functions instead of inside the Streamlit script, which
cannot be unit-tested.

It sits in agent/ rather than ui/ because it is domain logic (geocoding + AR6
vocabulary), not presentation: the API layer can reuse it, and mypy type-checks
agent/ while ui/ is a script directory it deliberately skips.
"""
from __future__ import annotations

from pydantic import BaseModel

from tools.ar6_regions import AR6Region, region_for
from tools.geocode import geocode
from tools.validation import validate_coordinates

# Roughly two thirds of the planet is outside every AR6 *land* region. That is a
# real answer, not a failure: the forecast and the ERA5 return levels are
# computed from gridded global reanalysis and stay valid, only the regional
# assessment tables have nothing to say about the point.
NO_REGION_NOTICE = (
    "This point falls outside every IPCC AR6 land region (open ocean, or a gap "
    "between regions). The live forecast and the ERA5 hazard statistics are still "
    "computed for these exact coordinates: only the regional IPCC context is "
    "unavailable, so this report may carry no citations."
)


class ResolvedPlace(BaseModel):
    """One point on Earth, ready for the agent: where it is and what to call it."""

    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""  # state / province, for disambiguation in the report
    region: AR6Region | None = None

    @property
    def label(self) -> str:
        """The `location` string the agent gets.

        With a region, it is spelled the corpus's way ("South Asia (SAS)") so
        retrieval hits the AR6 table row deterministically instead of hoping a
        city name bridges semantically. Without one, it degrades to the place
        name — never to an invented region.
        """
        if self.region is not None:
            return f"{self.name}, {self.region.label}"
        return f"{self.name}, {self.country}".rstrip(", ")

    @property
    def notice(self) -> str | None:
        """The honest caveat for this point, or None when there is nothing to say."""
        return None if self.region is not None else NO_REGION_NOTICE


def format_coordinates(latitude: float, longitude: float) -> str:
    """Name a bare coordinate pair the way a chart would: 22.2600°N, 84.8500°E."""
    ns = "S" if latitude < 0 else "N"
    ew = "W" if longitude < 0 else "E"
    return f"{abs(latitude):.4f}°{ns}, {abs(longitude):.4f}°{ew}"


def coordinate_error(latitude: float, longitude: float) -> str | None:
    """The tool boundary's own complaint about these coordinates, or None.

    The same ranges the network tools enforce, surfaced early so a UI can say
    what is wrong instead of letting a request fail deeper in.
    """
    try:
        validate_coordinates(latitude, longitude)
    except ValueError as exc:
        return str(exc)
    return None


def _with_region(
    name: str, latitude: float, longitude: float, country: str = "", admin1: str = ""
) -> ResolvedPlace:
    return ResolvedPlace(
        name=name,
        latitude=latitude,
        longitude=longitude,
        country=country,
        admin1=admin1,
        region=region_for(latitude, longitude),
    )


def resolve_place(place: str) -> ResolvedPlace:
    """Place name -> coordinates + AR6 region. Raises GeocodeError if unresolvable.

    The error stays typed and propagates: a front door must say "I could not
    find that place", never silently fall back to somewhere else.
    """
    located = geocode(place)
    return _with_region(
        located.name,
        located.latitude,
        located.longitude,
        located.country,
        located.admin1,
    )


def resolve_coordinates(
    latitude: float, longitude: float, *, name: str | None = None, country: str = ""
) -> ResolvedPlace:
    """Hand-entered lat/lon -> the same typed shape (ValueError outside the ranges)."""
    validate_coordinates(latitude, longitude)
    return _with_region(
        name or format_coordinates(latitude, longitude), latitude, longitude, country
    )


def name_still_applies(
    named_at: tuple[float, float] | None, latitude: float, longitude: float
) -> bool:
    """Does a remembered place name still describe this point?

    A UI that geocodes "Rourkela" and then lets the coordinates be edited by
    hand would otherwise keep calling a spot in the Pacific "Rourkela". The name
    is only kept while the point it was resolved for has not moved.
    """
    if named_at is None:
        return False
    return abs(named_at[0] - latitude) < 1e-6 and abs(named_at[1] - longitude) < 1e-6
