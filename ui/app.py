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

from agent import graph as agent_graph
from agent.contracts import Hazard, RiskLevel
from tools import climatology
from tools.climatology import ClimatologyError

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

_LEVEL_BADGE = {
    RiskLevel.LOW: "🟢 LOW",
    RiskLevel.MODERATE: "🟡 MODERATE",
    RiskLevel.HIGH: "🟠 HIGH",
    RiskLevel.SEVERE: "🔴 SEVERE",
}

st.set_page_config(page_title="Climate-Risk Analyst Agent", page_icon="🌍")
st.title("🌍 Climate-Risk Analyst Agent")
st.caption(
    "Live forecast (Open-Meteo) + ERA5 GEV return levels + IPCC AR6 citations "
    "→ one structured, validated RiskReport. Measured: 91% retrieval R@3, "
    "0 confabulated answers on a 45-question frozen benchmark."
)

with st.sidebar:
    location = st.selectbox("Location", list(LOCATIONS))
    hazard = st.selectbox(
        "Hazard", list(Hazard), format_func=lambda h: h.value.replace("_", " ")
    )
    horizon = st.slider("Forecast horizon (days)", 1, 16, 7)
    use_climatology = st.checkbox(
        "Ground with ERA5 climatology (GEV return levels)", value=True,
        help="First call fetches 60+ years of daily extremes (~3 s), then cached.",
    )
    run = st.button("Assess risk", type="primary")

if run:
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

    with st.spinner("Running agent (forecast → IPCC research → synthesis)…"):
        report = agent_graph.run_agent(
            location=location, latitude=latitude, longitude=longitude,
            hazard=hazard, horizon_days=horizon, hazard_stat=hazard_stat,
        )

    if report.refusal is not None:
        st.error(f"**Refused:** {report.refusal}")
        st.caption("Out-of-scope is an explicit, valid output — not a fabricated risk.")
    else:
        st.subheader(f"{_LEVEL_BADGE[report.risk_level]} — {report.hazard.value.replace('_', ' ')}")
        st.metric("Confidence", f"{report.confidence:.0%}")
        st.write(report.summary)

        if report.drivers:
            st.markdown("**Risk drivers**")
            for d in report.drivers:
                st.markdown(f"- **{d.factor}**: {d.detail}")

        if report.citations:
            st.markdown("**IPCC AR6 citations** (page-level, validator-guaranteed)")
            for c in report.citations:
                st.markdown(f"- 📄 `{c.source}` — {c.locator}")
        else:
            st.caption(
                "No IPCC citations in this report (RAG layer offline or the "
                "answerer honestly abstained — it never invents)."
            )

        if report.hazard_stats:
            stat = report.hazard_stats[0]
            st.markdown(f"**ERA5 climatology** ({stat.n_years} years, {stat.variable})")
            st.table(
                {
                    "return period (yr)": [r.return_period_years for r in stat.return_levels],
                    "level": [round(r.level, 1) for r in stat.return_levels],
                }
            )
            st.caption(
                f"Record max in series: {round(stat.record_max, 1)} — "
                f"representativeness: {stat.representativeness.value}"
            )

    with st.expander("Data provenance (audit trail)"):
        for p in report.provenance:
            st.markdown(f"- **{p.source}** — `{p.url}` at {p.retrieved_at:%Y-%m-%d %H:%M} UTC")
            st.json(p.params, expanded=False)

    with st.expander("Raw RiskReport JSON (the contract)"):
        st.code(report.model_dump_json(indent=2), language="json")
