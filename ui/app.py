"""Streamlit UI: one page over the agent — pick a location + hazard, get a
grounded, cited RiskReport.

The UI reads ONLY the RiskReport contract (never internal state), so every
path the agent supports renders honestly: refusals render as refusals, a
degraded RAG layer renders as a citation-less report, abstentions add nothing.

Run:  uv run streamlit run ui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/app.py` puts ui/ (not the repo root) on sys.path — same
# entry-point shim the MCP servers use.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from obs.log import configure

configure()  # the UI process owns logging config (library layers just log)

# imports below the sys.path shim + logging config on purpose (script entrypoint)
from agent import graph as agent_graph  # noqa: E402
from agent.contracts import Hazard, RiskLevel  # noqa: E402
from tools import climatology  # noqa: E402
from tools.climatology import ClimatologyError  # noqa: E402
from tools.forecast import ForecastError  # noqa: E402

# Demo locations (geocoding is a documented DEBT item; these cover the
# India-weighted eval set plus one non-Indian sanity point).
LOCATIONS: dict[str, tuple[float, float]] = {
    "Rourkela, India": (22.26, 84.85),
    "Mumbai, India": (19.08, 72.88),
    "Delhi, India": (28.61, 77.21),
    "Chennai, India": (13.08, 80.27),
    "Kolkata, India": (22.57, 88.36),
    "Berlin, Germany": (52.52, 13.40),
}

# Severity → (badge color, Material icon). Colors match the semantic palette
# in .streamlit/config.toml, so the badge is themed consistently in light/dark.
_LEVEL_STYLE: dict[RiskLevel, tuple[str, str]] = {
    RiskLevel.LOW: ("green", ":material/check_circle:"),
    RiskLevel.MODERATE: ("yellow", ":material/warning:"),
    RiskLevel.HIGH: ("orange", ":material/priority_high:"),
    RiskLevel.SEVERE: ("red", ":material/emergency:"),
}

st.set_page_config(
    page_title="Climate-Risk Analyst Agent",
    page_icon="🌍",
    layout="wide",
)
st.title("🌍 Climate-Risk Analyst Agent")
st.caption(
    "Ask about heat, extreme rainfall or wind risk anywhere on Earth. You get a "
    "structured report built from a live forecast, 60+ years of ERA5 climate "
    "statistics, and the IPCC AR6 assessment — with page-level citations, and an "
    "honest refusal when the evidence is not there."
)

with st.expander("How to use this (start here)", icon=":material/help:"):
    st.markdown(
        """
**Ask in plain language**, for example:

- *How risky are heatwaves in Berlin over the next 7 days?*
- *Is extreme rainfall a concern in Mumbai next week?*
- *What is the wind risk in Chennai over the next 10 days?*

**What it covers.** Three hazards only: **heat / heatwaves**, **extreme
precipitation**, and **wind**. Anything else (drought, flooding, cyclones,
wildfire, sea level) is deliberately **refused** rather than guessed at, and it
will tell you so.

**How to read the result.**

| Part | What it means |
| --- | --- |
| Risk level | Where the forecast peak falls on *this location's own* ERA5 return-level curve, not a fixed threshold |
| Return levels | The 1-in-10 / 50 / 100 year severity for this exact spot, with a 90% bootstrap confidence interval |
| Warming-trend banner | Shown only when a statistical test finds a real trend; the levels are then "effective" at today's climate |
| IPCC citations | Every citation is machine-checked against the pages actually retrieved. No citation means it declined to claim something it could not ground |
| Confidence | Rises with better data representativeness and with IPCC grounding; capped, because a forecast is never certain |

**Refusals are a feature.** An empty citation list or a refusal means the system
would rather say nothing than invent a number. On a 105-question held-out
benchmark it produced **zero fabricated answers**.
        """
    )

# On a hosted deploy without the baked corpus (Streamlit Community Cloud, a
# fresh clone, etc.) fetch the IPCC PDFs once. Skipped when PYTEST_CURRENT_TEST
# is set, so tests and CI stay hermetic and offline; locally the corpus is
# already on disk. Cheap file check per rerun; download runs only when missing.
import os as _os  # noqa: E402

from rag.corpus import corpus_present  # noqa: E402

if not _os.environ.get("PYTEST_CURRENT_TEST") and not corpus_present():
    with st.spinner("First run: fetching the IPCC AR6 corpus (~50 MB, one time)…"):
        from scripts.download_ipcc import main as _download_corpus  # noqa: E402

        _download_corpus()

# Dense-retrieval self-test (once per session). A wrong embedding region/model
# must surface LOUDLY here — not hide behind a citation-less report while every
# query silently 404s to BM25-only. Skipped under pytest (hermetic UI tests).
if not _os.environ.get("PYTEST_CURRENT_TEST"):
    if "dense_ok" not in st.session_state:
        from rag.gemini_client import embedding_available  # noqa: E402

        st.session_state.dense_ok, st.session_state.dense_detail = embedding_available()
    if not st.session_state.dense_ok:
        st.error(
            f"**Degraded mode:** {st.session_state.dense_detail}. Retrieval is "
            "running **BM25-only** (measured ~82% vs 87% hybrid R@3), so IPCC "
            "citations may be sparse. This is an embedding config issue "
            "(model / region), not a data problem.",
            icon=":material/warning:",
        )

# One-click examples: a first-time visitor should be able to see a real report
# without inventing a question. Each writes the query into the input via
# session_state, so the text stays editable afterwards.
_EXAMPLES = {
    "🌡️ Heat in Berlin": "How risky are heatwaves in Berlin over the next 7 days?",
    "🌧️ Rainfall in Mumbai": "Is extreme rainfall a concern in Mumbai over the next 7 days?",
    "💨 Wind in Chennai": "What is the wind risk in Chennai over the next 10 days?",
    "🚫 Out of scope": "What is the wildfire risk in Sydney next week?",
}
st.caption("Try an example:")
_cols = st.columns(len(_EXAMPLES))
for _col, (_label, _query) in zip(_cols, _EXAMPLES.items()):
    if _col.button(_label, width="stretch"):
        st.session_state["nl_query"] = _query

# Natural-language front door: any place on Earth, plain English.
nl_query = st.text_input(
    "Ask in plain language",
    key="nl_query",
    placeholder="How risky are heatwaves in Rourkela over the next 10 days?",
    help="Deterministic parsing → geocoding → AR6 region mapping → the agent. "
         "Unsupported hazards and unknown places refuse honestly.",
)
ask = st.button("Ask", type="primary", icon=":material/travel_explore:")

with st.sidebar:
    st.caption("…or configure the assessment manually")
    location = st.selectbox("Location", list(LOCATIONS))
    hazard = st.selectbox(
        "Hazard", list(Hazard), format_func=lambda h: h.value.replace("_", " ")
    )
    horizon = st.slider("Forecast horizon (days)", 1, 16, 7)
    use_climatology = st.checkbox(
        "Ground with ERA5 climatology (GEV return levels)", value=True,
        help="First call fetches 60+ years of daily extremes (~3 s), then cached.",
    )
    run = st.button(
        "Assess risk", type="primary", icon=":material/troubleshoot:", width="stretch",
    )

report = None
# The live forecast is the agent's core input; if Open-Meteo is down (503s
# happen) even after the tool's retries, show a clean message instead of a raw
# traceback — the same graceful posture the climatology/RAG layers already take.
_FORECAST_DOWN = (
    "The forecast service (Open-Meteo) is temporarily unavailable. "
    "This is an upstream outage, not a problem with your request — please try again in a moment."
)
if ask and nl_query.strip():
    from agent.nl import run_agent_nl  # noqa: E402
    from obs.telemetry import Span

    with st.spinner("Parsing → geocoding → AR6 region → agent…"):
        with Span("report") as span:
            try:
                report = run_agent_nl(nl_query)
            except ForecastError:
                st.error(_FORECAST_DOWN, icon=":material/cloud_off:")
elif run:
    latitude, longitude = LOCATIONS[location]

    hazard_stat = None
    if use_climatology:
        try:
            with st.spinner("Fitting ERA5 GEV climatology…"):
                hazard_stat = climatology.climatology_hazard_stat(
                    latitude, longitude, hazard
                )
        except ClimatologyError as exc:
            st.warning(f"Climatology unavailable ({exc}) — continuing without it.")

    from obs.telemetry import Span

    with st.spinner("Running agent (forecast → IPCC research → synthesis)…"):
        with Span("report") as span:
            try:
                report = agent_graph.run_agent(
                    location=location, latitude=latitude, longitude=longitude,
                    hazard=hazard, horizon_days=horizon, hazard_stat=hazard_stat,
                )
            except ForecastError:
                st.error(_FORECAST_DOWN, icon=":material/cloud_off:")

if report is not None:
    if report.refusal is not None:
        st.error(f"**Refused:** {report.refusal}", icon=":material/block:")
        st.caption("Out-of-scope is an explicit, valid output — not a fabricated risk.")
    else:
        color, icon = _LEVEL_STYLE[report.risk_level]

        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader(
                    f"{report.risk_level.value.upper()} — "
                    f"{report.hazard.value.replace('_', ' ')} risk in {report.location}"
                )
                st.badge(report.risk_level.value.upper(), color=color, icon=icon)
            st.write(report.summary)

        metric_cols = st.columns(3 if report.hazard_stats else 2, border=True)
        metric_cols[0].metric("Confidence", f"{report.confidence:.0%}")
        metric_cols[1].metric("Forecast horizon", f"{report.horizon_days} d")
        if report.hazard_stats:
            stat = report.hazard_stats[0]
            metric_cols[2].metric(
                "Record max (series)", f"{round(stat.record_max, 1)} {stat.variable}",
            )

        col_left, col_right = st.columns(2)

        with col_left:
            with st.container(border=True):
                st.markdown("**Risk drivers**")
                if report.drivers:
                    for d in report.drivers:
                        st.markdown(f"- **{d.factor}**: {d.detail}")
                else:
                    st.caption("No drivers reported.")

        with col_right:
            with st.container(border=True):
                st.markdown("**IPCC AR6 citations** (page-level, validator-guaranteed)")
                if report.citations:
                    for c in report.citations:
                        st.markdown(
                            f":gray-badge[:material/description: {c.source}] "
                            f":blue-badge[{c.locator}]"
                        )
                else:
                    st.caption(
                        "No IPCC citations in this report (RAG layer offline or the "
                        "answerer honestly abstained — it never invents)."
                    )

        if report.hazard_stats:
            stat = report.hazard_stats[0]
            with st.container(border=True):
                st.markdown(f"**ERA5 climatology** — {stat.n_years} years, {stat.variable}")
                table = {
                    "return_period_years": [
                        r.return_period_years for r in stat.return_levels
                    ],
                    "level": [round(r.level, 1) for r in stat.return_levels],
                }
                if all(r.ci_low is not None for r in stat.return_levels):
                    table["ci"] = [
                        f"{r.ci_low:.1f} – {r.ci_high:.1f}" for r in stat.return_levels
                    ]
                st.dataframe(
                    table,
                    column_config={
                        "return_period_years": st.column_config.NumberColumn(
                            "Return period (yr)", format="%d",
                        ),
                        "level": st.column_config.NumberColumn(
                            f"Level ({stat.variable})", format="%.1f",
                        ),
                        "ci": st.column_config.TextColumn("90% CI (bootstrap)"),
                    },
                    hide_index=True,
                )
                st.caption(
                    f"Record max in series: {round(stat.record_max, 1)} — "
                    f"representativeness: {stat.representativeness.value}"
                )
                if stat.trend is not None:
                    if stat.trend.significant:
                        st.warning(
                            f"Warming trend detected: {stat.trend.slope_per_decade:+.1f} "
                            f"{stat.unit}/decade (p={stat.trend.p_value:.3f}). Return levels "
                            f"above are EFFECTIVE at {stat.trend.evaluated_at_year} — "
                            "today's climate, not the historical average.",
                            icon=":material/trending_up:",
                        )
                    else:
                        st.caption(
                            f"Non-stationarity tested: no significant trend "
                            f"(p={stat.trend.p_value:.2f}) — stationary fit reported."
                        )

    with st.expander("Cost & latency (measured telemetry)", icon=":material/speed:"):
        s = span.summary()
        obs_cols = st.columns(4)
        obs_cols[0].metric("Wall time", f"{s['wall_ms']/1000:.1f} s")
        obs_cols[1].metric("Model calls", s["calls"], help="Live Gemini calls (cache hits excluded)")
        obs_cols[2].metric("Cache hits", s["cache_hits"])
        obs_cols[3].metric("Est. cost", f"${s['est_cost_usd']:.4f}",
                           help="Estimated from token counts × configured prices — not a bill.")
        st.caption(
            f"tokens in/out: {s['tokens_in']}/{s['tokens_out']} · "
            f"retries: {s['retries']} · failures: {s['failures']} — every model "
            "call is measured at the SDK seam; none can opt out."
        )

    with st.expander("Data provenance (audit trail)", icon=":material/fact_check:"):
        for p in report.provenance:
            st.markdown(f"- **{p.source}** — `{p.url}` at {p.retrieved_at:%Y-%m-%d %H:%M} UTC")
            st.json(p.params, expanded=False)

    with st.expander("Raw RiskReport JSON (the contract)", icon=":material/code:"):
        st.code(report.model_dump_json(indent=2), language="json")
