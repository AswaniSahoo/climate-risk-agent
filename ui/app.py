"""Streamlit UI: one page over the agent: pick a location and hazard, get a
grounded, cited RiskReport.

The UI reads ONLY the RiskReport contract (never internal state), so every
path the agent supports renders honestly: refusals render as refusals, a
degraded RAG layer renders as a citation-less report, abstentions add nothing.

Run:  uv run streamlit run ui/app.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

# `streamlit run ui/app.py` puts ui/ (not the repo root) on sys.path: same
# entry-point shim the MCP servers use.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from obs.log import configure

configure()  # the UI process owns logging config (library layers just log)

_log = logging.getLogger(__name__)  # tracebacks go HERE, never onto the page

# imports below the sys.path shim + logging config on purpose (script entrypoint)
# agent.graph is NOT imported here: it pulls langgraph (~17 s cold) and nothing
# on the boot path needs it. It is imported where a report is actually run.
from agent.contracts import Hazard  # noqa: E402
from agent.location import (  # noqa: E402
    coordinate_error,
    name_still_applies,
    resolve_coordinates,
    resolve_place,
)

# Pydantic + stdlib only (no langgraph), so this stays off the expensive path:
# it is the shared vocabulary between the agent's step events and this renderer.
from agent.progress import (  # noqa: E402
    CLIMATOLOGY,
    LOCATE,
    StepEvent,
    StepStatus,
    emit_step,
    error_sentence,
    format_elapsed,
    step_meaning,
    track,
)
from tools import climatology  # noqa: E402
from tools.climatology import ClimatologyError  # noqa: E402
from tools.geocode import GeocodeError  # noqa: E402
from ui.report_plot import return_level_plot, specimen_plot  # noqa: E402
from ui.theme import (  # noqa: E402
    APP_CSS,
    FAVICON_SVG,
    MARK_SVG,
    PIN_RED,
    band_heading,
    citation_list,
    driver_list,
    method_strip,
    plate_caption,
    quoted_statement,
    specimen_card,
    why_line,
)

# Shortcut buttons, not the menu: any place on Earth works via the Place box or
# the coordinate inputs. These are the eval-set cities whose ERA5/answer caches
# are pre-warmed, so a first click returns instantly.
LOCATIONS: dict[str, tuple[float, float]] = {
    "Rourkela, India": (22.26, 84.85),
    "Mumbai, India": (19.08, 72.88),
    "Delhi, India": (28.61, 77.21),
    "Chennai, India": (13.08, 80.27),
    "Kolkata, India": (22.57, 88.36),
    "Berlin, Germany": (52.52, 13.40),
}

# The question and the example report share one split on the first screen.
_SPLIT = [11, 9]

st.set_page_config(
    page_title="Climate-Risk Analyst Agent",
    page_icon=FAVICON_SVG,
    layout="wide",
)

st.html(APP_CSS)

# Seeded BEFORE any input widgets exist, so suggestion chips and preset buttons
# can populate them reliably.
st.session_state.setdefault("lat_input", 22.26)
st.session_state.setdefault("lon_input", 84.85)
st.session_state.setdefault("place_name", "Rourkela")
st.session_state.setdefault("place_country", "India")
st.session_state.setdefault("place_coords", (22.26, 84.85))
st.session_state.setdefault("place_query", "Rourkela, India")
st.session_state.setdefault("horizon_input", 7)

# Is an answer on its way this run? A button's click is in session state before the
# button is drawn, so the page can be laid out for the answer instead of the intro.
_answering = bool(st.session_state.get("assess")) or bool(
    (st.session_state.get("ask_btn") or st.session_state.get("_nl_submitted"))
    and str(st.session_state.get("nl_query", "")).strip()
)

# First visit: the question on the left, an example report on the right (on a phone
# they stack in that order). With an answer coming, the question bar sits alone above it.
if _answering:
    _lead, _side = st.container(key="cra_lead_wide"), None
else:
    _lead, _side = st.columns(_SPLIT, gap="large")

with _lead:
    st.markdown(
        f'<div class="cra-head">{MARK_SVG}<h1 class="cra-title">Climate-Risk Analyst Agent</h1></div>'
        '<p class="cra-lede">Heatwave, extreme rainfall and wind risk for any place on Earth, '
        "graded against that place's own climate record.</p>",
        unsafe_allow_html=True,
    )

# The full account of the method and how to use it: everything the first screen
# says in one line per step, kept word for word behind one expander.
_METHOD_MD = """
Every assessment cross-examines operational forecast signals against 60+ years of extreme
value distributions and the IPCC AR6 assessment tables:

1. **Operational forecast: numerical forecast ensemble.** Fetches high-resolution weather
   models (Open-Meteo) up to 16 days out for daily temperature extremes, heavy rainfall
   accumulation, or maximum wind gusts at your exact coordinate.
2. **Historical reanalysis: ERA5 extreme value climatology.** Fits 63 years (1960 to 2022)
   of daily historical extremes using Generalized Extreme Value (GEV) parametric
   distributions. Calibrates local 10, 50, and 100-year return levels with 90% bootstrap
   confidence intervals and tests for non-stationary warming trends.
3. **Peer-reviewed synthesis: IPCC AR6 grounding & verification.** Cross-references
   physical drivers against Chapter 11 and regional trend projections against Chapter 12.
   Every assertion requires machine-checked page citations; ungrounded statements are
   strictly omitted.

**Climatological rarity: relative severity vs. raw weather.** Conventional weather apps
only report raw predictions. The Climate-Risk Analyst determines how extreme that forecast
is relative to the historical climatology of that specific location. For instance, 38 °C
in Berlin triggers a severe risk rating because it surpasses the local 50-year return
level, whereas the same temperature in Rourkela represents expected seasonal weather.

**Audited abstention: zero fabricated answers.** Three physical hazards are supported:
heatwaves, extreme precipitation, and wind. Out-of-scope hazards (cyclones, wildfires,
drought, sea-level rise) and ungrounded queries trigger explicit typed refusals rather than
hallucinated estimates.

**Ask in plain language**, for example:

- How risky are heatwaves in Berlin over the next 7 days?
- Is extreme rainfall a concern in Mumbai next week?
- What is the wind risk in Chennai over the next 10 days?

**Or pick the point yourself** in the sidebar: geocode any place name, or type a
latitude/longitude. The map marks the selected point (display-only: it cannot
be clicked to move the pin).

**What it covers.** Three hazards only: **heat / heatwaves**, **extreme
precipitation**, and **wind**. Anything else (drought, flooding, cyclones,
wildfire, sea level) is deliberately **refused** rather than guessed at, and it
will tell you so.

**How to read the result.**

| Part | What it means |
| --- | --- |
| Risk level | Where the forecast peak falls on this location's own ERA5 return-level curve, not a fixed threshold |
| Return levels | The 1-in-10 / 50 / 100 year severity for this exact spot, with a 90% bootstrap confidence interval |
| Warming-trend banner | Shown only when a statistical test finds a real trend; the levels are then "effective" at today's climate |
| IPCC citations | Every citation is machine-checked against the pages actually retrieved. No citation means it declined to claim something it could not ground |
| Confidence | Rises with better data representativeness and with IPCC grounding; capped, because a forecast is never certain |

**Refusals are a feature.** An empty citation list or a refusal means the system
would rather say nothing than invent a number. On a 105-question held-out
benchmark it produced **zero fabricated answers**.
"""

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
# must surface LOUDLY here: not hide behind a citation-less report while every
# query silently 404s to BM25-only. Skipped under pytest (hermetic UI tests).
if not _os.environ.get("PYTEST_CURRENT_TEST"):
    if "dense_ok" not in st.session_state:
        from rag.gemini_client import embedding_available  # noqa: E402

        st.session_state.dense_ok, st.session_state.dense_detail = embedding_available()
    if not st.session_state.dense_ok:
        with _lead:
            st.error(
                f"**Degraded mode:** {st.session_state.dense_detail}. Retrieval is "
                "running **BM25-only** (R@3 76% against 82% for the hybrid retriever, "
                "measured on the 60-question dev set), so IPCC citations may be sparse. "
                "This is an embedding config issue (model / region), not a data problem.",
                icon=":material/warning:",
            )

def _remember_place(name: str, country: str, latitude: float, longitude: float) -> None:
    """Pin a name to the point it was resolved for (see location.name_still_applies).

    Also writes the sidebar's Place box, so it never names a different place than the
    map. Callers run before that box is drawn (or in a callback), as Streamlit requires.
    """
    st.session_state["lat_input"] = float(latitude)
    st.session_state["lon_input"] = float(longitude)
    st.session_state["place_name"] = name
    st.session_state["place_country"] = country
    st.session_state["place_coords"] = (float(latitude), float(longitude))
    st.session_state["place_query"] = f"{name}, {country}" if country else name


def _sync_controls(query: str, *, locate: bool) -> None:
    """Point the sidebar at what a plain-language question asks about.

    Hazard and horizon always; the place too when `locate`, since a chip already knows
    its point. Runs before the sidebar widgets exist in this run.
    """
    from agent.nl import parse_query

    parsed = parse_query(query)
    if parsed.hazard is not None:
        st.session_state["hazard_input"] = parsed.hazard
    if 1 <= parsed.horizon_days <= 16:
        st.session_state["horizon_input"] = parsed.horizon_days
    if locate and parsed.place:
        try:
            found = resolve_place(parsed.place)
        except GeocodeError:
            return
        _remember_place(found.name, found.country, found.latitude, found.longitude)


def _resolve_and_set_place(query_text: str) -> bool:
    """Geocode and synchronize coordinates immediately for a place name."""
    clean = query_text.strip()
    if not clean:
        return False
    try:
        found = resolve_place(clean)
        _remember_place(found.name, found.country, found.latitude, found.longitude)
        st.session_state["_place_geocode_error"] = None
        return True
    except GeocodeError as exc:
        st.session_state["_place_geocode_error"] = str(exc)
        return False


def _on_nl_query_submit() -> None:
    """Invoked when user hits Enter in the plain language input."""
    st.session_state["_nl_submitted"] = True


def _on_place_query_change() -> None:
    """Invoked when user changes the place query text input."""
    _resolve_and_set_place(st.session_state.get("place_query", ""))


# One-click examples: each chip fills the question, points the sidebar and the map
# at its place and hazard, and runs it.
_SUGGESTIONS = [
    (
        "Heatwave · Berlin (7d)",
        "How risky are heatwaves in Berlin over the next 7 days?",
        "Berlin",
        "Germany",
        52.52,
        13.40,
    ),
    (
        "Rainfall · Mumbai (7d)",
        "Is extreme rainfall a concern in Mumbai over the next 7 days?",
        "Mumbai",
        "India",
        19.08,
        72.88,
    ),
    (
        "Wind gusts · Chennai (10d)",
        "What is the wind risk in Chennai over the next 10 days?",
        "Chennai",
        "India",
        13.08,
        80.27,
    ),
    (
        "Wildfire · Sydney (refusal)",
        "What is the wildfire risk in Sydney next week?",
        "Sydney",
        "Australia",
        -33.87,
        151.21,
    ),
]
with _lead:
    _ask_slot = st.container()
    with st.container(gap="xxsmall"):
        st.caption("Or run an example")
        _chip_slot = st.container(key="cra_chips")

with _chip_slot:
    # No icons on the chips: Streamlit renders a Material icon as its ligature text with no
    # aria-hidden, so a screen reader would read "thermostat Heatwave" as the button's name.
    for _label, _query, _city, _country, _lat, _lon in _SUGGESTIONS:
        if st.button(_label, type="tertiary"):
            st.session_state["nl_query"] = _query
            _remember_place(_city, _country, _lat, _lon)
            _sync_controls(_query, locate=False)
            st.session_state["_nl_place_set"] = _query  # no second geocode for it
            st.session_state["_nl_submitted"] = True
            st.rerun()

with _ask_slot:
    _query_col, _ask_col = st.columns([6, 1], gap="xsmall", vertical_alignment="bottom")
    # Natural-language front door: any place on Earth, plain English.
    nl_query = _query_col.text_input(
        "Ask in plain language",
        key="nl_query",
        placeholder="How risky are heatwaves in Rourkela over the next 10 days?",
        help="Deterministic parsing → geocoding → AR6 region mapping → the agent. "
             "Unsupported hazards and unknown places refuse honestly.",
        on_change=_on_nl_query_submit,
    )
    ask = _ask_col.button("Ask", key="ask_btn", type="primary", width="stretch")
# Popped before the `or`: typing a question and then clicking Ask lands in one rerun
# with both set, and a flag left behind would rerun the question on the next click.
_submitted = bool(st.session_state.pop("_nl_submitted", False))
nl_triggered = bool(ask or _submitted)

# A question points the sidebar (Place box, coordinates, map, hazard, horizon) at
# what it asks about, before the sidebar is drawn, so the two never disagree.
if nl_triggered and nl_query.strip():
    if st.session_state.pop("_nl_place_set", None) != nl_query:
        _sync_controls(nl_query, locate=True)

with st.sidebar:
    st.markdown('<h2 class="cra-side-title">Manual assessment</h2>', unsafe_allow_html=True)
    with st.container(gap="xsmall"):
        st.markdown(
            '<div class="cra-side-label"><span>Preset locations</span>'
            '<span class="cra-side-meta">cached</span></div>',
            unsafe_allow_html=True,
        )
        with st.container(key="cra_presets"):
            for _name in LOCATIONS:
                _city, _, _country = _name.partition(", ")
                if st.button(_city, key=f"ex_{_name}", width="stretch", help=_name):
                    _remember_place(_city, _country, *LOCATIONS[_name])
                    st.rerun()

    place_query = st.text_input(
        "Place", key="place_query", placeholder="Any town, city or region on Earth",
        help="Geocoded via Open-Meteo. The resolved name, country and coordinates are "
             "shown on the map, so a wrong match is visible rather than silent.",
        on_change=_on_place_query_change,
    )
    # A callback, not an `if st.button`: it runs before the Place box is drawn, so it may
    # rewrite the box to the name it found.
    st.button("Find place", width="stretch", on_click=_on_place_query_change)

    _geocode_err = st.session_state.pop("_place_geocode_error", None)
    if _geocode_err:
        st.warning(f"Could not resolve that place: {_geocode_err}", icon=":material/wrong_location:")

    # Ranges mirror tools/validation.validate_coordinates; coordinate_error is the
    # same check, so a value typed past the widget still refuses instead of flying.
    latitude = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0, step=0.01, format="%.4f",
        key="lat_input", help="Decimal degrees, −90 to 90 (negative = southern).",
    )
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, step=0.01, format="%.4f",
        key="lon_input", help="Decimal degrees, −180 to 180 (negative = western).",
    )

    selected = None
    _coord_error = coordinate_error(latitude, longitude)
    if _coord_error is not None:
        st.error(_coord_error, icon=":material/error:")
    else:
        # A name only describes the point it was resolved for. Edit the boxes and
        # it is dropped for the coordinates themselves, rather than labelling a
        # spot in the Pacific "Rourkela".
        _named = name_still_applies(
            st.session_state.get("place_coords"), latitude, longitude
        )
        selected = resolve_coordinates(
            latitude, longitude,
            name=st.session_state["place_name"] if _named else None,
            country=st.session_state["place_country"] if _named else "",
        )

    # Where the assessment will run, right under the inputs that set it. Display-only
    # by design: Streamlit's map selection reports which OBJECTS a click picked
    # (per-layer `indices` / `objects`), never the coordinates of the click itself,
    # so a click cannot place a new point. The Place box and the coordinates do that.
    if selected is not None:
        with st.container(border=True, key="cra_plate", gap=None):
            st.map(
                {"lat": [selected.latitude], "lon": [selected.longitude]},
                zoom=3, color=PIN_RED, size=50_000, height=200,
            )
            with st.container(key="cra_plate_info"):
                st.markdown(
                    plate_caption(
                        selected.name,
                        selected.country,
                        selected.region.label if selected.region else None,
                        selected.latitude,
                        selected.longitude,
                    ),
                    unsafe_allow_html=True,
                )
                if selected.notice is not None:
                    st.info(selected.notice, icon=":material/public_off:")

    hazard = st.selectbox(
        "Hazard", list(Hazard), format_func=lambda h: h.value.replace("_", " ").capitalize(),
        key="hazard_input",
    )
    horizon = st.slider("Forecast horizon (days)", 1, 16, key="horizon_input")
    use_climatology = st.checkbox(
        "Ground with ERA5 climatology (GEV return levels)", value=True,
        help="The first report for a new place fetches 63 years of ERA5 daily extremes and "
             "fits them: about 1 to 2 minutes, then cached.",
    )
    # Secondary on purpose: the Ask box is the page's one primary action.
    run = st.button("Assess risk", key="assess", width="stretch")
    st.caption("First report for a new place: about 1 to 2 minutes, then cached.")

report = None
failure: str | None = None


class _StepPanel:
    """Renders the agent's StepEvents into an open st.status container.

    One `st.empty()` slot per step, so a step's line is REPLACED when it
    finishes rather than a second line appearing beneath it: the panel stays a
    checklist a person can read, not a scrolling log.
    """

    def __init__(self, status: Any) -> None:
        self._status = status
        self._slots: dict[str, Any] = {}

    def _slot(self, node: str) -> Any:
        if node not in self._slots:
            self._slots[node] = self._status.empty()
        return self._slots[node]

    def __call__(self, event: StepEvent) -> None:
        if event.status is StepStatus.STARTED:
            # The collapsed title says what is happening right now, so the panel
            # is useful even when the reader has it shut.
            self._status.update(label=f"{event.label}…")
            head = f":gray-badge[running] **{event.label}**"
        elif event.status is StepStatus.FINISHED:
            badge = f" :gray-badge[{event.detail}]" if event.detail else ""
            head = f":green-badge[{format_elapsed(event.seconds)}] **{event.label}**{badge}"
        elif event.status is StepStatus.SKIPPED:
            badge = f" :gray-badge[{event.detail}]" if event.detail else ""
            head = f":gray-badge[skipped] **{event.label}**{badge}"
        else:  # FAILED: the sentence for the reader goes on the panel's title
            head = f":red-badge[failed] **{event.label}**"
        self._slot(event.node).markdown(f"{head}  \n:small[{step_meaning(event.node)}]")


def _run_with_panel(work: Any) -> tuple[Any, str | None]:
    """Run `work(on_step)` under a live status panel. Returns (report, message).

    The panel's own title carries the outcome: total time when it worked, one
    plain sentence when it did not. The traceback goes to the log, never onto
    the page, which is the whole point of `error_sentence`.
    """
    started = time.perf_counter()
    with st.status("Building your report…", expanded=True) as status:
        try:
            result = work(_StepPanel(status))
        except Exception as exc:
            _log.exception("report run failed")
            sentence = error_sentence(exc)
            status.update(label=sentence, state="error", expanded=True)
            return None, sentence
        if getattr(result, "refusal", None):
            title = "Refused: the question is outside what this agent can ground"
        else:
            title = f"Report ready in {format_elapsed(time.perf_counter() - started)}"
        status.update(label=title, state="complete", expanded=False)
        return result, None


if nl_triggered and nl_query.strip():
    from agent.nl import run_agent_nl  # noqa: E402
    from obs.telemetry import Span  # noqa: E402

    with Span("report") as span:
        report, failure = _run_with_panel(
            lambda on_step: run_agent_nl(nl_query, on_step=on_step)
        )
    if failure is not None:
        st.error(failure, icon=":material/error:")
elif run and selected is not None and hazard is not None:
    from agent import graph as agent_graph  # noqa: E402
    from obs.telemetry import Span  # noqa: E402

    sel = selected
    haz = hazard
    # Collected rather than written inline: a warning drawn inside the status
    # container would vanish when the panel collapses on success.
    notices: list[str] = []

    def _assess(on_step):
        # The point was already resolved, synchronously, by the sidebar widgets.
        # Reporting it anyway keeps both entry points showing the same steps.
        emit_step(on_step, LOCATE, StepStatus.FINISHED, detail="resolved in the sidebar")
        hazard_stat = None
        if use_climatology:
            try:
                with track(on_step, CLIMATOLOGY):
                    hazard_stat = climatology.climatology_hazard_stat(
                        sel.latitude, sel.longitude, haz
                    )
            except ClimatologyError as exc:
                # Loud but non-fatal, exactly as before: the band falls back to
                # absolute cutoffs and the report states which basis it used.
                notices.append(f"Climatology unavailable ({exc}). Continuing without it.")
        else:
            emit_step(on_step, CLIMATOLOGY, StepStatus.SKIPPED, detail="ERA5 grounding off")
        return agent_graph.run_agent(
            location=sel.label,
            latitude=sel.latitude, longitude=sel.longitude,
            hazard=haz, horizon_days=horizon, hazard_stat=hazard_stat,
            on_step=on_step,
        )

    with Span("report") as span:
        report, failure = _run_with_panel(_assess)
    for notice in notices:
        st.warning(notice, icon=":material/warning:")
    if failure is not None:
        st.error(failure, icon=":material/error:")

if report is not None:
    if report.refusal is not None:
        with st.container(key="cra_refusal"):
            st.error(f"**Refused:** {report.refusal}", icon=":material/block:")
            st.caption("Out-of-scope is an explicit, valid output: not a fabricated risk.")
    else:
        _plot_html = return_level_plot(report)
        # The agent writes the forecast and ERA5 numbers first, then any IPCC prose.
        # Same words in the same order, but only the numbers sit beside the plot.
        _numbers, _, _ipcc_prose = report.summary.partition(" IPCC AR6: ")
        # The agent's own sentence for why the band is what it is, lifted under the answer.
        _why = next((d.detail for d in report.drivers if d.factor == "severity_basis"), None)

        # Headings go h1, h2, h3 for a screen reader; the band itself is the visible title.
        st.markdown('<h2 class="cra-sr-only">Report</h2>', unsafe_allow_html=True)
        with st.container(border=True, key="cra_verdict"):
            _verdict_cols = st.columns(2, gap="large") if _plot_html else [st.container()]
            with _verdict_cols[0]:
                with st.container(gap="xxsmall"):
                    st.markdown(band_heading(report.risk_level), unsafe_allow_html=True)
                    st.subheader(
                        f"{report.hazard.value.replace('_', ' ').capitalize()} "
                        f"risk in {report.location}",
                        anchor=False,
                    )
                    if _why:
                        st.markdown(why_line(_why), unsafe_allow_html=True)
                st.caption(_numbers)
                metric_cols = st.columns(3 if report.hazard_stats else 2)
                metric_cols[0].metric("Confidence", f"{report.confidence:.0%}")
                metric_cols[1].metric("Forecast horizon", f"{report.horizon_days} d")
                if report.hazard_stats:
                    stat = report.hazard_stats[0]
                    metric_cols[2].metric(
                        "Record max (series)", f"{round(stat.record_max, 1)} {stat.unit}",
                    )
            if _plot_html:
                with _verdict_cols[1]:
                    st.markdown(_plot_html, unsafe_allow_html=True)
            if _ipcc_prose:
                _prose_key = "cra_prose_long" if len(_ipcc_prose) > 480 else "cra_prose"
                with st.container(key=_prose_key):
                    st.write(f"IPCC AR6: {_ipcc_prose}")

        # Collapsed, and below the answer: it answers "what am I looking at?" for a
        # first reader without pushing the band down the page for anyone.
        with st.expander("How to read this report", icon=":material/menu_book:"):
            st.markdown(
                """
- **Risk band**: where the forecast peak lands on this location's own ERA5
  return-level curve (the 2 / 10 / 50-year edges), not a fixed threshold. With no
  fitted curve it falls back to absolute cutoffs, and the `severity_basis` driver
  says which of the two you are reading.
- **Confidence**: composed from what the report actually has, namely how
  representative the ERA5 series is, the **measured** forecast skill at this lead
  day (a peak 10 days out is worth less than one tomorrow), and whether an IPCC
  citation was produced. Capped, because a forecast is never certain.
- **Citations**: page-level (`file · p123`) and structurally validated against
  the pages actually retrieved. An empty list means the answerer declined to
  claim something it could not ground.
- **Projected change**: a verbatim AR6 Chapter 12 sentence for the reference
  region containing this point. Nothing is paraphrased, and an absence is stated
  in words rather than quietly filled in.
- **Limits**: what this cannot do, and where the numbers stop being valid, in
  [LIMITATIONS.md](https://github.com/AswaniSahoo/climate-risk-agent/blob/main/LIMITATIONS.md).
- **Cost & latency**: wall time, live model calls (cache hits excluded), token
  counts and an **estimated** dollar figure. Tokens are measured at the SDK seam;
  the dollars come from a price table, so they are an estimate, not a bill.
                """
            )

        col_left, col_right = st.columns(2)

        with col_left:
            with st.container(border=True):
                st.markdown("**Risk drivers**")
                if report.drivers:
                    st.markdown(driver_list(report.drivers), unsafe_allow_html=True)
                else:
                    st.caption("No drivers reported.")

        with col_right:
            with st.container(border=True):
                st.markdown("**IPCC AR6 citations** (page-level, validator-guaranteed)")
                if report.citations:
                    st.markdown(citation_list(report.citations), unsafe_allow_html=True)
                else:
                    st.caption(
                        "No IPCC citations in this report (RAG layer offline or the "
                        "answerer honestly abstained; it never invents)."
                    )

            # Display-only: the report JSON already carries the whole thing.
            with st.container(border=True):
                st.markdown("**Projected change (IPCC AR6 Ch.12)**")
                _pc = report.projected_change
                if _pc is not None:
                    st.markdown(
                        f":gray-badge[{_pc.region_name} ({_pc.region_acronym})] "
                        f":gray-badge[{_pc.direction.value.replace('_', ' ')}]"
                        + (
                            f" :gray-badge[{_pc.confidence_language}]"
                            if _pc.confidence_language
                            else ""
                        )
                        + (
                            f" :gray-badge[{_pc.warming_level_or_period}]"
                            if _pc.warming_level_or_period
                            else ""
                        )
                    )
                    st.markdown(quoted_statement(_pc.statement), unsafe_allow_html=True)
                    st.markdown(citation_list(_pc.citations), unsafe_allow_html=True)
                else:
                    st.caption(
                        "Regional projections were not found in the corpus for this "
                        "hazard and AR6 region: nothing is asserted in their place."
                    )

        if report.hazard_stats:
            stat = report.hazard_stats[0]
            with st.container(border=True):
                st.markdown(f"**ERA5 climatology** ({stat.n_years} years, {stat.variable})")
                _table_col, _notes_col = st.columns([7, 5], gap="large")
                table = {
                    "return_period_years": [
                        r.return_period_years for r in stat.return_levels
                    ],
                    "level": [round(r.level, 1) for r in stat.return_levels],
                }
                if all(r.ci_low is not None for r in stat.return_levels):
                    table["ci"] = [
                        f"{r.ci_low:.1f} to {r.ci_high:.1f}" for r in stat.return_levels
                    ]
                _table_col.dataframe(
                    table,
                    column_config={
                        "return_period_years": st.column_config.NumberColumn(
                            "Return period (yr)", format="%d",
                        ),
                        "level": st.column_config.NumberColumn(
                            f"Level ({stat.unit})", format="%.1f",
                        ),
                        "ci": st.column_config.TextColumn("90% CI (bootstrap)"),
                    },
                    hide_index=True,
                )
                _notes_col.caption(
                    f"Record max in series: {round(stat.record_max, 1)} {stat.unit} · "
                    f"representativeness: {stat.representativeness.value.replace('_', ' ')}"
                )
                if stat.trend is not None and not stat.trend.significant:
                    _notes_col.caption(
                        f"Non-stationarity tested: no significant trend "
                        f"(p={stat.trend.p_value:.2f}) · stationary fit reported."
                    )
                if stat.trend is not None and stat.trend.significant:
                    st.warning(
                        f"Warming trend detected: {stat.trend.slope_per_decade:+.1f} "
                        f"{stat.unit}/decade (p={stat.trend.p_value:.3f}). Return levels "
                        f"above are effective at {stat.trend.evaluated_at_year} "
                        "(today's climate, not the historical average).",
                        icon=":material/trending_up:",
                    )

    with st.expander("Cost & latency (measured telemetry)", icon=":material/speed:"):
        s = span.summary()
        obs_cols = st.columns(4)
        obs_cols[0].metric("Wall time", f"{s['wall_ms']/1000:.1f} s")
        obs_cols[1].metric("Model calls", s["calls"], help="Live Gemini calls (cache hits excluded)")
        obs_cols[2].metric("Cache hits", s["cache_hits"])
        obs_cols[3].metric("Est. cost", f"${s['est_cost_usd']:.4f}",
                           help="Estimated from token counts × configured prices (not a bill).")
        st.caption(
            f"tokens in/out: {s['tokens_in']}/{s['tokens_out']} · "
            f"retries: {s['retries']} · failures: {s['failures']} · every model "
            "call is measured at the SDK seam; none can opt out."
        )

    with st.expander("Data provenance (audit trail)", icon=":material/fact_check:"):
        for p in report.provenance:
            st.markdown(f"- **{p.source}**: `{p.url}` at {p.retrieved_at:%Y-%m-%d %H:%M} UTC")
            st.json(p.params, expanded=False)

    with st.expander("Raw RiskReport JSON (the contract)", icon=":material/code:"):
        st.code(report.model_dump_json(indent=2), language="json")

# First visit, or nothing came back: the method in three lines and the example report.
if report is None and failure is None:
    with _lead if _side is not None else st.container():
        st.markdown(
            method_strip([
                ("Forecast", "Open-Meteo, up to 16 days ahead"),
                ("60+ years of ERA5 extremes", "this point's own return levels"),
                ("IPCC AR6 pages", "every claim cites its page"),
            ]),
            unsafe_allow_html=True,
        )
        with st.expander("How it works, in full"):
            st.markdown(_METHOD_MD)
    if _side is not None:
        with _side, st.container(border=True, key="cra_specimen"):
            st.markdown(specimen_card(*specimen_plot()), unsafe_allow_html=True)
