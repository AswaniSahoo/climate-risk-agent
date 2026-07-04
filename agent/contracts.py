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

    heatwave ← temperature_2m_max, extreme_precip ← precipitation_sum (both from
    get_forecast); wind is in-vocabulary but its data path isn't wired yet, so
    a wind query currently takes the refusal path.
    """

    HEATWAVE = "heatwave"
    EXTREME_PRECIP = "extreme_precip"
    WIND = "wind"


class RiskDriver(BaseModel):
    """A single factor pushing the risk level, with a human-readable detail."""

    factor: str
    detail: str


class Citation(BaseModel):
    """A reference to a source document backing a claim.

    Minimal Day-1 placeholder; real RAG-derived citations arrive in Wk2.
    """

    source: str
    locator: str


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
    hazard: Hazard
    horizon_days: int = Field(gt=0)
    confidence: float = Field(ge=0.0, le=1.0)

    risk_level: RiskLevel | None = None
    summary: str = ""
    drivers: list[RiskDriver] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    provenance: list[DataProvenance] = Field(default_factory=list)
    refusal: str | None = None

    @model_validator(mode="after")
    def _check_refusal_consistency(self) -> "RiskReport":
        if self.refusal is None and self.risk_level is None:
            raise ValueError("risk_level is required unless the report is a refusal")
        if self.refusal is not None and self.risk_level is not None:
            raise ValueError("a refusal report must not assert a risk_level")
        return self
