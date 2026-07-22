"""Async FastAPI service over the agent — the programmatic front door.

Why this layer exists: MCP serves AI clients and Streamlit serves humans;
anything programmatic (batch jobs, other services, a future portfolio
endpoint) needs plain HTTP with a typed contract. The response IS the
RiskReport plus the measured telemetry for that run — a caller always learns
what its request cost.

Async design note: the agent and climatology layers are deliberately
synchronous (they're also called from scripts, evals, and MCP where an event
loop may not exist). Here they run via `asyncio.to_thread`, so one slow
Open-Meteo or Gemini call never blocks the event loop — the server keeps
serving /healthz and other requests while a report is being built.

Access control: set env API_KEY -> POST /report and GET /metrics require a
matching `x-api-key` header. Unset -> open, with a loud dev-mode warning.

Run:  uv run uvicorn api.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from agent.contracts import Hazard, RiskReport
from agent.graph import run_agent
from agent.nl import run_agent_nl
from obs.telemetry import Span, snapshot, summarize
from tools.climatology import ClimatologyError, climatology_hazard_stat

_log = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    from obs.log import configure

    configure()  # the API process owns logging config
    if not os.environ.get("API_KEY"):
        _log.warning("API_KEY not set — endpoints are open (dev mode)")
    # Dense-retrieval self-test at startup: fail loud in logs if the embedding
    # model/region is unreachable, instead of silently serving BM25-only.
    # Skipped under pytest so the TestClient startup stays hermetic/offline.
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        from rag.gemini_client import embedding_available

        ok, detail = embedding_available()
        if ok:
            _log.info("dense retrieval self-test OK — %s", detail)
        else:
            _log.warning(
                "DENSE DEGRADED — %s. Serving BM25-only until embeddings are "
                "reachable (check embedding model/region).", detail
            )
    yield


app = FastAPI(
    title="Climate-Risk Analyst Agent API",
    description="Grounded, cited, structured climate-risk reports with per-request telemetry.",
    lifespan=_lifespan,
)


def _require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Header check, read per-request so key rotation needs no restart.

    A deliberately small mechanism: one shared key gates the endpoints that
    spend money (/report) or expose usage (/metrics). /healthz stays open for
    load-balancer probes.
    """
    expected = os.environ.get("API_KEY")
    if expected is None:
        return  # dev mode — warned once at startup
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


class ReportRequest(BaseModel):
    """Typed request contract — the tool layer re-validates coordinates/horizon."""

    location: str
    latitude: float
    longitude: float
    hazard: Hazard
    horizon_days: int = Field(default=7, gt=0)
    use_climatology: bool = True


class ReportResponse(BaseModel):
    report: RiskReport
    telemetry: dict


@app.post("/report", response_model=ReportResponse, dependencies=[Depends(_require_api_key)])
async def create_report(request: ReportRequest) -> ReportResponse:
    hazard_stat = None
    if request.use_climatology:
        try:
            hazard_stat = await asyncio.to_thread(
                climatology_hazard_stat, request.latitude, request.longitude, request.hazard
            )
        except ClimatologyError as exc:
            _log.warning("climatology unavailable (%s) — report proceeds without it", exc)

    try:
        with Span("api-report") as span:
            report = await asyncio.to_thread(
                run_agent,
                location=request.location,
                latitude=request.latitude,
                longitude=request.longitude,
                hazard=request.hazard,
                horizon_days=request.horizon_days,
                hazard_stat=hazard_stat,
            )
    except ValueError as exc:  # tool-boundary validation (coords/horizon)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ReportResponse(report=report, telemetry=span.summary())


class QueryRequest(BaseModel):
    """Free-text front door: 'How risky are heatwaves in Rourkela next week?'"""

    query: str


@app.post("/query", response_model=ReportResponse, dependencies=[Depends(_require_api_key)])
async def create_report_from_query(request: QueryRequest) -> ReportResponse:
    """NL entry: parse -> geocode -> AR6 region -> agent. Unresolvable queries
    return a 200 with a typed REFUSAL report — refusing is a valid output of
    this system, not a server error."""
    with Span("api-query") as span:
        report = await asyncio.to_thread(run_agent_nl, request.query)
    return ReportResponse(report=report, telemetry=span.summary())


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(_require_api_key)])
async def metrics() -> dict:
    """Per-op telemetry rollups for this process (calls, failures, p50/p95, est cost)."""
    return summarize(snapshot())
