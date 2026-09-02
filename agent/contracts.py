"""Output contract for the Climate-Risk Analyst Agent.

`RiskReport` is the strict, typed shape every agent run must produce. Pydantic
validates it at runtime, so a wrong type / missing field / out-of-range value
fails loudly here instead of leaking downstream into the API, UI, or evals.

Built BEFORE the tools and graph on purpose: this is the *target* they exist to
fill.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, model_validator

from tools.hazard_stats import HazardStat


class RiskLevel(str, Enum):
    """Qualitative climate-risk band (IPCC-style). A controlled vocabulary so the
    level is never a free-form string like "kinda high"."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    SEVERE = "severe"


class Hazard(str, Enum):
    """Hazards the agent supports (per MASTER-PLAN). Out-of-scope hazards are
    rejected here, which forces the agent down the refusal path.

    heatwave ← temperature_2m_max, extreme_precip ← precipitation_sum,
    wind ← wind_gusts_10m_max (all from get_forecast). Wind is graded on the
    gust, not the sustained speed, because the ERA5 wind climatology it is
    compared against is a gust fit.
    """

    HEATWAVE = "heatwave"
    EXTREME_PRECIP = "extreme_precip"
    WIND = "wind"


class ChangeDirection(str, Enum):
    """Direction of a projected change, as the IPCC text states it.

    UNKNOWN is a real answer, not a failure: AR6 marks many region/CID pairs
    "low confidence in direction of change", and MIXED covers a sentence that
    states an increase in one part of the region and a decrease in another.
    """

    INCREASE = "increase"
    DECREASE = "decrease"
    NO_CHANGE = "no_change"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class RiskDriver(BaseModel):
    """A single factor pushing the risk level, with a human-readable detail."""

    factor: str
    detail: str


class Citation(BaseModel):
    """A reference to a source document backing a claim.

    RAG-derived and validated at page level against the chunks actually
    retrieved for the question: a citation that cannot be tied to a retrieved
    page forces a refusal rather than a fabricated reference. `source` is the
    document; `locator` is the page.

    `chunk_id` is optional and carries the retrieval unit the claim came from,
    so a consumer can re-check the claim against the exact excerpt rather than
    a whole page. It defaults to None because the page-level citations built in
    agent/graph.py deliberately collapse several chunks of one page into one.
    """

    source: str
    locator: str
    chunk_id: str | None = None


class ProjectedChange(BaseModel):
    """One cited AR6 Ch.12 climatic impact-driver (CID) projection for a region.

    STRUCTURAL CITATION RULE (the same guarantee `CitedAnswer` gives in
    rag/answer.py): every citation must carry a `chunk_id` drawn from
    `retrieved_chunk_ids` — the chunks retrieval actually returned for this
    region/hazard query. A citation to anything else is a schema violation, so
    a fabricated reference cannot be constructed, only rejected. `statement` is
    derived verbatim from those same chunks; nothing here is generated text.
    """

    region_acronym: str
    region_name: str
    hazard: Hazard
    statement: str  # verbatim-derived sentence from a retrieved Ch.12 chunk
    direction: ChangeDirection
    confidence_language: str | None = None  # the IPCC calibrated phrase as written
    warming_level_or_period: str | None = None
    citations: list[Citation]
    retrieved_chunk_ids: list[str]  # the chunk_ids this projection is bound to

    @model_validator(mode="after")
    def _check_citation_integrity(self) -> "ProjectedChange":
        if not self.citations:
            raise ValueError("a projected change must cite at least one retrieved chunk")
        missing = [c.locator for c in self.citations if c.chunk_id is None]
        if missing:
            raise ValueError(f"projected-change citations must carry a chunk_id: {missing}")
        unknown = [
            c.chunk_id for c in self.citations if c.chunk_id not in self.retrieved_chunk_ids
        ]
        if unknown:
            raise ValueError(f"citations {unknown} not among the retrieved chunks")
        return self


class DataProvenance(BaseModel):
    """Where a piece of data came from: source name, URL, when it was fetched,
    and the query params. This is the audit trail the eval harness and a human
    reviewer use to verify every claim."""

    source: str
    url: str
    retrieved_at: datetime
    params: dict = Field(default_factory=dict)


class RiskReport(BaseModel):
    """The agent's structured output contract.

    Invariant: the report either asserts a risk (`risk_level` set) OR refuses
    (`refusal` set) — never both, never neither. Out-of-scope is an explicit,
    valid output, not a crash and not a fabricated risk.
    """

    location: str
    # None is allowed ONLY on refusals: an out-of-scope natural-language query
    # ("wildfire risk?") has no valid Hazard value to carry.
    hazard: Hazard | None
    horizon_days: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)

    risk_level: RiskLevel | None = None
    summary: str = ""
    drivers: list[RiskDriver] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    provenance: list[DataProvenance] = Field(default_factory=list)
    hazard_stats: list[HazardStat] = Field(default_factory=list)
    # Absent (None) whenever the AR6 Ch.12 CID assessment yielded nothing for
    # this region/hazard — the report then says so rather than guessing.
    projected_change: ProjectedChange | None = None
    refusal: str | None = None

    @model_validator(mode="after")
    def _check_refusal_consistency(self) -> "RiskReport":
        if self.refusal is None and self.risk_level is None:
            raise ValueError("risk_level is required unless the report is a refusal")
        if self.refusal is not None and self.risk_level is not None:
            raise ValueError("a refusal report must not assert a risk_level")
        if self.hazard is None and self.refusal is None:
            raise ValueError("hazard may be omitted only on a refusal report")
        return self
