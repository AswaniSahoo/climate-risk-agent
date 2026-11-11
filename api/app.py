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

Access control: set env API_KEY -> POST /report, /query, /triage and GET
/metrics require a matching `x-api-key` header. Unset -> open, with a loud
dev-mode warning.

Request limit: POST /report, /query and /triage, the endpoints that can spend
money, allow CRG_API_RATE_LIMIT requests per CRG_API_RATE_WINDOW_S seconds per
client (default 10 per 60 s, "off" to disable) and answer 429 with Retry-After
past that. The client is the connecting address as uvicorn reports it: uvicorn
honours X-Forwarded-For only from FORWARDED_ALLOW_IPS (127.0.0.1 by default in
uvicorn 0.50.0), so behind a proxy, list the proxy there or every caller
shares one limit.
The daily spend limit in rag/spend_guard.py still caps the total.

Run:  uv run uvicorn api.app:app --port 8000
"""
from __future__ import annotations

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from agent.contracts import Hazard, RiskReport
from agent.graph import run_agent
from agent.nl import TriageOutcome, run_agent_nl, triage_query
from obs.telemetry import Span, snapshot, summarize
from rag import spend_guard
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
    # Constant time, so response timing does not leak how much of a guess was
    # right; on bytes, because compare_digest raises on a non-ASCII str.
    if x_api_key is None or not hmac.compare_digest(
        x_api_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="invalid or missing x-api-key")


def _limit_client(request: Request) -> None:
    """Per-client request limit on the endpoints that can spend money."""
    client = request.client.host if request.client is not None else "unknown"
    wait_s = spend_guard.client_rate_limited("api", client)
    if wait_s is None:
        return
    limit, window = spend_guard.api_rate_limit() or (0, spend_guard.DEFAULT_API_RATE_WINDOW_S)
    raise HTTPException(
        status_code=429,
        detail=(f"Too many requests from this client: at most {limit} requests per "
                f"{window} s. Retry in {wait_s} s."),
        headers={"Retry-After": str(wait_s)},
    )


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


@app.post("/report", response_model=ReportResponse,
          dependencies=[Depends(_limit_client), Depends(_require_api_key)])
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


@app.post("/query", response_model=ReportResponse,
          dependencies=[Depends(_limit_client), Depends(_require_api_key)])
async def create_report_from_query(request: QueryRequest) -> ReportResponse:
    """NL entry: parse -> geocode -> AR6 region -> agent. Unresolvable queries
    return a 200 with a typed REFUSAL report — refusing is a valid output of
    this system, not a server error."""
    with Span("api-query") as span:
        report = await asyncio.to_thread(run_agent_nl, request.query)
    return ReportResponse(report=report, telemetry=span.summary())


class TriageResponse(BaseModel):
    triage: TriageOutcome
    telemetry: dict


@app.post("/triage", response_model=TriageResponse,
          dependencies=[Depends(_limit_client), Depends(_require_api_key)])
async def triage_question(request: QueryRequest) -> TriageResponse:
    """Read a question without answering it: hazard, horizon, place, and the
    refusal `/query` would give, in well under a second.

    For a client to pre-fill its controls while the person is still looking at
    the question, and to show an out-of-scope refusal without starting the
    pipeline. No geocoding, forecast or Gemini call happens here; with
    CRG_JEV=triage|all at most one Jev request does, and none when the lexical
    parser settles the question. Gated like `/query` because Jev bills per token.
    """
    with Span("api-triage") as span:
        outcome = await asyncio.to_thread(triage_query, request.query)
    return TriageResponse(triage=outcome, telemetry=span.summary())


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


@app.get("/metrics", dependencies=[Depends(_require_api_key)])
async def metrics() -> dict:
    """Per-op telemetry rollups for this process (calls, failures, p50/p95, est cost)."""
    return summarize(snapshot())
