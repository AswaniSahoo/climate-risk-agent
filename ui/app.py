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
    "Live forecast (Open-Meteo) + ERA5 GEV return levels + IPCC AR6 citations "
    "→ one structured, validated RiskReport. Measured: 91% retrieval R@3, "
    "0 confabulated answers on a 45-question frozen benchmark."
)

with st.sidebar:
    st.caption("Configure the assessment")
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

    from obs.telemetry import Span

    with st.spinner("Running agent (forecast → IPCC research → synthesis)…"):
        with Span("report") as span:
            report = agent_graph.run_agent(
                location=location, latitude=latitude, longitude=longitude,
                hazard=hazard, horizon_days=horizon, hazard_stat=hazard_stat,
            )

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
