"""Streamlit UI: one page over the agent: pick a location and hazard, get a
grounded, cited RiskReport.

The UI reads ONLY the RiskReport contract (never internal state), so every
path the agent supports renders honestly: refusals render as refusals, a
degraded RAG layer renders as a citation-less report, abstentions add nothing.

Run:  uv run streamlit run ui/app.py
"""
from __future__ import annotations

import base64
import logging
import sys
import time
from pathlib import Path
from typing import Any, Literal

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
from agent.contracts import Hazard, RiskLevel  # noqa: E402
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

# Severity -> (badge color, Material icon). Colors match the semantic palette
# in .streamlit/config.toml, so the badge is themed consistently in light/dark.
_BadgeColor = Literal["green", "yellow", "orange", "red"]
_LEVEL_STYLE: dict[RiskLevel, tuple[_BadgeColor, str]] = {
    RiskLevel.LOW: ("green", ":material/check_circle:"),
    RiskLevel.MODERATE: ("yellow", ":material/warning:"),
    RiskLevel.HIGH: ("orange", ":material/priority_high:"),
    RiskLevel.SEVERE: ("red", ":material/emergency:"),
}

_ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets"
_FAVICON_PATH = _ASSETS_DIR / "favicon.png"
if not _FAVICON_PATH.is_file():
    _FAVICON_PATH = Path("assets/favicon.png")

_LOGO_B64 = ""
if _FAVICON_PATH.is_file():
    try:
        _LOGO_B64 = base64.b64encode(_FAVICON_PATH.read_bytes()).decode("utf-8")
    except Exception as _exc:
        _log.warning("Could not read brand logo: %s", _exc)

st.set_page_config(
    page_title="Climate-Risk Analyst Agent",
    page_icon=str(_FAVICON_PATH) if _FAVICON_PATH.is_file() else "assets/favicon.png",
    layout="wide",
)

if hasattr(st, "logo") and _FAVICON_PATH.is_file():
    try:
        st.logo(str(_FAVICON_PATH), icon_image=str(_FAVICON_PATH))
    except Exception:
        pass

st.markdown(
    """
    <style>
    /* Scientific Climate Risk Intelligence Theme (Dark & Light) */
    :root {
      --cra-bg: #111613;
      --cra-surface: rgba(26, 34, 30, 0.72);
      --cra-surface-card: rgba(22, 29, 25, 0.85);
      --cra-surface-card-hover: rgba(28, 38, 33, 0.95);
      --cra-border: rgba(82, 183, 136, 0.18);
      --cra-border-subtle: rgba(82, 183, 136, 0.10);
      --cra-border-hover: rgba(82, 183, 136, 0.35);
      --cra-text: #ECEAE4;
      --cra-text-muted: #9BA59F;
      --cra-text-subtle: #6E7972;
      --cra-primary: #3E886D;
      --cra-primary-dark: #2D6A4F;
      --cra-primary-hover: #4D9E80;
      --cra-primary-glow: rgba(62, 136, 109, 0.25);
      --cra-badge-bg: rgba(62, 136, 109, 0.14);
      --cra-badge-border: rgba(62, 136, 109, 0.28);
      --cra-badge-text: #52B788;
      --cra-dot: #52B788;
      --cra-callout-red-bg: rgba(229, 115, 115, 0.08);
      --cra-callout-red-border: rgba(229, 115, 115, 0.24);
      --cra-callout-red-text: #EF9A9A;
      --cra-callout-green-bg: rgba(62, 136, 109, 0.09);
      --cra-callout-green-border: rgba(62, 136, 109, 0.24);
      --cra-callout-green-text: #66BB6A;
      --cra-shadow-card: 0 4px 20px -2px rgba(0, 0, 0, 0.4), 0 1px 3px 0 rgba(0, 0, 0, 0.2);
      --cra-shadow-hover: 0 8px 30px -4px rgba(0, 0, 0, 0.55), 0 2px 6px 0 rgba(0, 0, 0, 0.25);
      --cra-inner-refract: inset 0 1px 0 rgba(255, 255, 255, 0.06);
    }

    [data-theme="light"], .stApp[data-theme="light"], [data-base-theme="light"], .stApp[data-base-theme="light"] {
      --cra-bg: #F9F8F5;
      --cra-surface: rgba(255, 255, 255, 0.82);
      --cra-surface-card: rgba(255, 255, 255, 0.90);
      --cra-surface-card-hover: rgba(255, 255, 255, 0.98);
      --cra-border: rgba(36, 94, 72, 0.14);
      --cra-border-subtle: rgba(36, 94, 72, 0.08);
      --cra-border-hover: rgba(36, 94, 72, 0.28);
      --cra-text: #1E2621;
      --cra-text-muted: #556058;
      --cra-text-subtle: #8A948D;
      --cra-primary: #245E48;
      --cra-primary-dark: #1B4736;
      --cra-primary-hover: #1B4736;
      --cra-primary-glow: rgba(36, 94, 72, 0.16);
      --cra-badge-bg: rgba(36, 94, 72, 0.08);
      --cra-badge-border: rgba(36, 94, 72, 0.18);
      --cra-badge-text: #245E48;
      --cra-dot: #2D6A4F;
      --cra-callout-red-bg: rgba(198, 146, 20, 0.05);
      --cra-callout-red-border: rgba(198, 146, 20, 0.18);
      --cra-callout-red-text: #9E2A2B;
      --cra-callout-green-bg: rgba(45, 106, 79, 0.05);
      --cra-callout-green-border: rgba(45, 106, 79, 0.18);
      --cra-callout-green-text: #2D6A4F;
      --cra-shadow-card: 0 4px 20px -2px rgba(30, 38, 33, 0.04), 0 1px 3px 0 rgba(30, 38, 33, 0.02);
      --cra-shadow-hover: 0 8px 28px -4px rgba(30, 38, 33, 0.07), 0 2px 6px 0 rgba(30, 38, 33, 0.02);
      --cra-inner-refract: inset 0 1px 0 rgba(255, 255, 255, 0.8);
    }

    @media (prefers-color-scheme: light) {
      :root:not([data-theme="dark"]):not([data-base-theme="dark"]) {
        --cra-bg: #F9F8F5;
        --cra-surface: rgba(255, 255, 255, 0.82);
        --cra-surface-card: rgba(255, 255, 255, 0.90);
        --cra-surface-card-hover: rgba(255, 255, 255, 0.98);
        --cra-border: rgba(36, 94, 72, 0.14);
        --cra-border-subtle: rgba(36, 94, 72, 0.08);
        --cra-border-hover: rgba(36, 94, 72, 0.28);
        --cra-text: #1E2621;
        --cra-text-muted: #556058;
        --cra-text-subtle: #8A948D;
        --cra-primary: #245E48;
        --cra-primary-dark: #1B4736;
        --cra-primary-hover: #1B4736;
        --cra-primary-glow: rgba(36, 94, 72, 0.16);
        --cra-badge-bg: rgba(36, 94, 72, 0.08);
        --cra-badge-border: rgba(36, 94, 72, 0.18);
        --cra-badge-text: #245E48;
        --cra-dot: #2D6A4F;
        --cra-callout-red-bg: rgba(198, 146, 20, 0.05);
        --cra-callout-red-border: rgba(198, 146, 20, 0.18);
        --cra-callout-red-text: #9E2A2B;
        --cra-callout-green-bg: rgba(45, 106, 79, 0.05);
        --cra-callout-green-border: rgba(45, 106, 79, 0.18);
        --cra-callout-green-text: #2D6A4F;
        --cra-shadow-card: 0 4px 20px -2px rgba(30, 38, 33, 0.04), 0 1px 3px 0 rgba(30, 38, 33, 0.02);
        --cra-shadow-hover: 0 8px 28px -4px rgba(30, 38, 33, 0.07), 0 2px 6px 0 rgba(30, 38, 33, 0.02);
        --cra-inner-refract: inset 0 1px 0 rgba(255, 255, 255, 0.8);
      }
    }

    /* Brand Header with Logo */
    .cra-header-container {
      display: flex;
      align-items: center;
      gap: 1.15rem;
      margin-top: 0.2rem;
      margin-bottom: 0.6rem;
    }
    .cra-brand-logo {
      width: 50px;
      height: 50px;
      border-radius: 12px;
      border: 1px solid var(--cra-border);
      box-shadow: 0 4px 14px -2px var(--cra-primary-glow), 0 1px 3px rgba(0, 0, 0, 0.15);
      object-fit: contain;
      image-rendering: -webkit-optimize-contrast;
      background: var(--cra-surface-card);
      padding: 2px;
      flex-shrink: 0;
      transition: transform 0.2s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.2s cubic-bezier(0.16, 1, 0.3, 1);
    }
    .cra-brand-logo:hover {
      transform: translateY(-1px) scale(1.03);
      box-shadow: 0 8px 22px -4px var(--cra-primary-glow);
    }
    .cra-header-content {
      display: flex;
      flex-direction: column;
      justify-content: center;
    }
    .cra-title {
      font-size: 2.3rem;
      font-weight: 700;
      letter-spacing: -0.025em;
      line-height: 1.12;
      margin: 0;
      color: var(--cra-text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }
    @media (max-width: 640px) {
      .cra-header-container {
        gap: 0.85rem;
      }
      .cra-brand-logo {
        width: 44px;
        height: 44px;
      }
      .cra-title {
        font-size: 1.7rem;
      }
    }

    /* Eyebrow Pill Badge */
    .cra-eyebrow {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.26rem 0.75rem;
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--cra-badge-text);
      background: var(--cra-badge-bg);
      border: 1px solid var(--cra-badge-border);
      border-radius: 9999px;
      margin-bottom: 0.45rem;
      width: fit-content;
    }
    @keyframes cra-pulse {
      0%, 100% { opacity: 1; transform: scale(1); }
      50% { opacity: 0.45; transform: scale(0.85); }
    }
    .cra-dot {
      width: 6px;
      height: 6px;
      border-radius: 50%;
      background: var(--cra-dot);
      box-shadow: 0 0 6px var(--cra-dot);
      animation: cra-pulse 2.8s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* Double-Bezel Card Depth */
    div[data-testid="stVerticalBlockBorderWrapper"] {
      border-radius: 14px !important;
      border: 1px solid var(--cra-border) !important;
      background: var(--cra-surface-card) !important;
      backdrop-filter: blur(14px) !important;
      -webkit-backdrop-filter: blur(14px) !important;
      box-shadow: var(--cra-shadow-card), var(--cra-inner-refract) !important;
      transition: border-color 0.22s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.22s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {
      border-color: var(--cra-border-hover) !important;
      box-shadow: var(--cra-shadow-hover), var(--cra-inner-refract) !important;
    }

    /* Tactile Physics for Buttons */
    .stButton > button {
      border-radius: 10px !important;
      font-weight: 500 !important;
      letter-spacing: -0.01em !important;
      border: 1px solid var(--cra-border) !important;
      background: var(--cra-surface) !important;
      color: var(--cra-text) !important;
      box-shadow: var(--cra-inner-refract) !important;
      transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
    }
    .stButton > button:hover {
      transform: translateY(-1px) !important;
      border-color: var(--cra-border-hover) !important;
      box-shadow: 0 4px 12px -2px var(--cra-primary-glow), var(--cra-inner-refract) !important;
      color: var(--cra-badge-text) !important;
    }
    .stButton > button:active {
      transform: translateY(0) scale(0.985) !important;
    }
    .stButton > button[kind="primary"] {
      background: var(--cra-primary) !important;
      border-color: var(--cra-primary-dark) !important;
      color: #F9F8F5 !important;
      box-shadow: 0 2px 8px -1px var(--cra-primary-glow) !important;
    }
    .stButton > button[kind="primary"]:hover {
      background: var(--cra-primary-hover) !important;
      box-shadow: 0 6px 18px -2px var(--cra-primary-glow) !important;
      color: #FFFFFF !important;
    }

    /* Metric Tabular Numbers & Clean Hierarchy */
    div[data-testid="stMetric"] {
      padding: 8px 4px !important;
    }
    div[data-testid="stMetricLabel"] {
      font-size: 0.72rem !important;
      text-transform: uppercase !important;
      letter-spacing: 0.08em !important;
      font-weight: 600 !important;
      color: var(--cra-text-muted) !important;
    }
    div[data-testid="stMetricValue"] {
      font-feature-settings: "tnum" 1 !important;
      font-variant-numeric: tabular-nums !important;
      letter-spacing: -0.025em !important;
      font-weight: 600 !important;
      color: var(--cra-text) !important;
    }

    /* Status & Expander Widgets */
    div[data-testid="stStatusWidget"] {
      border-radius: 12px !important;
      border: 1px solid var(--cra-border) !important;
      background: var(--cra-surface-card) !important;
    }
    div[data-testid="stExpander"] {
      border-radius: 12px !important;
      border: 1px solid var(--cra-border) !important;
      background: var(--cra-surface-card) !important;
    }

    /* Badges */
    span[data-testid="stBadge"] {
      border-radius: 6px !important;
      font-weight: 500 !important;
      letter-spacing: 0.02em !important;
    }

    /* Input Fields Focus State */
    div[data-baseweb="input"] {
      border-radius: 10px !important;
      border: 1px solid var(--cra-border) !important;
      background: var(--cra-surface) !important;
      transition: border-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    div[data-baseweb="input"]:focus-within {
      border-color: var(--cra-primary) !important;
      box-shadow: 0 0 0 3px var(--cra-primary-glow) !important;
    }

    /* Empty State Pipeline Styles */
    .cra-pipeline-card {
      padding: 1.1rem 1.2rem;
      border-radius: 12px;
      background: var(--cra-badge-bg);
      border: 1px solid var(--cra-border-subtle);
      margin-bottom: 0.85rem;
      transition: all 0.2s ease;
    }
    .cra-pipeline-card:hover {
      background: rgba(62, 136, 109, 0.12);
      border-color: var(--cra-border);
    }
    .cra-step-badge {
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--cra-badge-text);
    }
    .cra-step-title {
      font-size: 0.96rem;
      font-weight: 600;
      letter-spacing: -0.01em;
      margin: 0.25rem 0 0.4rem 0;
      color: var(--cra-text);
    }
    .cra-step-desc {
      font-size: 0.85rem;
      line-height: 1.5;
      color: var(--cra-text-muted);
    }
    .cra-callout {
      padding: 1.1rem 1.2rem;
      border-radius: 12px;
      background: var(--cra-callout-red-bg);
      border: 1px solid var(--cra-callout-red-border);
      margin-bottom: 0.85rem;
    }
    .cra-callout-green {
      padding: 1.1rem 1.2rem;
      border-radius: 12px;
      background: var(--cra-callout-green-bg);
      border: 1px solid var(--cra-callout-green-border);
      margin-bottom: 0.85rem;
    }

    /* Sidebar Instrument Deck */
    .cra-sidebar-header {
      padding: 0.35rem 0 0.85rem 0;
      margin-bottom: 0.65rem;
      border-bottom: 1px solid var(--cra-border-subtle);
    }
    .cra-sidebar-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.35rem;
      padding: 0.22rem 0.65rem;
      font-size: 0.64rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      color: var(--cra-badge-text);
      background: var(--cra-badge-bg);
      border: 1px solid var(--cra-badge-border);
      border-radius: 9999px;
      margin-bottom: 0.45rem;
    }
    .cra-sidebar-title {
      font-size: 1.08rem;
      font-weight: 600;
      letter-spacing: -0.015em;
      color: var(--cra-text);
      margin-bottom: 2px;
    }
    .cra-sidebar-desc {
      font-size: 0.78rem;
      line-height: 1.42;
      color: var(--cra-text-muted);
    }
    .cra-section-label {
      display: flex;
      align-items: center;
      justify-content: space-between;
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: var(--cra-text-muted);
      margin: 0.85rem 0 0.45rem 0;
    }
    .cra-fast-badge {
      font-size: 0.62rem;
      font-weight: 600;
      padding: 0.12rem 0.45rem;
      border-radius: 4px;
      background: var(--cra-badge-bg);
      color: var(--cra-badge-text);
      border: 1px solid var(--cra-badge-border);
      letter-spacing: 0.04em;
    }

    /* Tactile Sidebar Preset Buttons */
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
      border-radius: 8px !important;
      font-size: 0.82rem !important;
      font-weight: 500 !important;
      padding: 0.32rem 0.5rem !important;
      border: 1px solid var(--cra-border) !important;
      background: var(--cra-surface) !important;
      color: var(--cra-text) !important;
      box-shadow: var(--cra-inner-refract) !important;
    }
    [data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
      background: var(--cra-badge-bg) !important;
      border-color: var(--cra-border-hover) !important;
      color: var(--cra-badge-text) !important;
      transform: translateY(-1px) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"] {
      border-radius: 10px !important;
      font-weight: 600 !important;
      letter-spacing: -0.01em !important;
      background: var(--cra-primary) !important;
      border: 1px solid var(--cra-primary-dark) !important;
      color: #F9F8F5 !important;
      box-shadow: 0 4px 14px -2px var(--cra-primary-glow), var(--cra-inner-refract) !important;
    }
    [data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
      background: var(--cra-primary-hover) !important;
      border-color: var(--cra-primary) !important;
      box-shadow: 0 6px 20px -2px var(--cra-primary-glow), var(--cra-inner-refract) !important;
      color: #FFFFFF !important;
      transform: translateY(-1px) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

_SVG_FALLBACK_LOGO = (
    '<svg class="cra-brand-logo" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect width="48" height="48" rx="12" fill="#1A221E" stroke="#3E886D" stroke-width="1.5"/>'
    '<circle cx="24" cy="24" r="14" stroke="#52B788" stroke-width="1.5" stroke-dasharray="2 3"/>'
    '<circle cx="24" cy="24" r="8" stroke="#3E886D" stroke-width="1.5"/>'
    '<circle cx="24" cy="24" r="3" fill="#52B788"/>'
    '<path d="M24 10V14M24 34V38M10 24H14M34 24H38" stroke="#52B788" stroke-width="1.5" stroke-linecap="round"/>'
    '</svg>'
)

_logo_html = (
    f'<img src="data:image/png;base64,{_LOGO_B64}" class="cra-brand-logo" alt="Climate-Risk Analyst Logo" />'
    if _LOGO_B64
    else _SVG_FALLBACK_LOGO
)

st.markdown(
    f"""
    <div class="cra-header-container">
        {_logo_html}
        <div class="cra-header-content">
            <div class="cra-eyebrow"><span class="cra-dot"></span>DECISION-GRADE CLIMATE INTELLIGENCE</div>
            <h1 class="cra-title">Climate-Risk Analyst Agent</h1>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    "Ask about heat, extreme rainfall or wind risk anywhere on Earth. You get a "
    "structured report built from a live forecast, 60+ years of ERA5 climate "
    "statistics, and the IPCC AR6 assessment, complete with page-level citations and an "
    "honest refusal when the evidence is not there."
)

with st.expander("How to use this (start here)", icon=":material/help:"):
    st.markdown(
        """
**Ask in plain language**, for example:

- *How risky are heatwaves in Berlin over the next 7 days?*
- *Is extreme rainfall a concern in Mumbai next week?*
- *What is the wind risk in Chennai over the next 10 days?*

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
# must surface LOUDLY here: not hide behind a citation-less report while every
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

def _remember_place(name: str, country: str, latitude: float, longitude: float) -> None:
    """Pin a name to the point it was resolved for (see location.name_still_applies)."""
    st.session_state["lat_input"] = float(latitude)
    st.session_state["lon_input"] = float(longitude)
    st.session_state["place_name"] = name
    st.session_state["place_country"] = country
    st.session_state["place_coords"] = (float(latitude), float(longitude))


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


# Seeded BEFORE any input widgets exist, so suggestion chips and preset buttons
# can populate them reliably.
st.session_state.setdefault("lat_input", 22.26)
st.session_state.setdefault("lon_input", 84.85)
st.session_state.setdefault("place_name", "Rourkela")
st.session_state.setdefault("place_country", "India")
st.session_state.setdefault("place_coords", (22.26, 84.85))
st.session_state.setdefault("place_query", "Rourkela, India")

# One-click examples: each chip populates the query text AND synchronizes the
# coordinates, place name, and map to the target location immediately.
_SUGGESTIONS = [
    (
        "Heatwave · Berlin (7d)",
        ":material/thermostat:",
        "How risky are heatwaves in Berlin over the next 7 days?",
        "Berlin",
        "Germany",
        52.52,
        13.40,
    ),
    (
        "Rainfall · Mumbai (7d)",
        ":material/rainy:",
        "Is extreme rainfall a concern in Mumbai over the next 7 days?",
        "Mumbai",
        "India",
        19.08,
        72.88,
    ),
    (
        "Wind gusts · Chennai (10d)",
        ":material/air:",
        "What is the wind risk in Chennai over the next 10 days?",
        "Chennai",
        "India",
        13.08,
        80.27,
    ),
    (
        "Wildfire · Sydney (Refusal)",
        ":material/block:",
        "What is the wildfire risk in Sydney next week?",
        "Sydney",
        "Australia",
        -33.87,
        151.21,
    ),
]
st.caption("Suggested inquiries (click to populate query):")
_cols = st.columns(len(_SUGGESTIONS))
for _col, (_label, _icon, _query, _city, _country, _lat, _lon) in zip(_cols, _SUGGESTIONS):
    if _col.button(_label, icon=_icon, width="stretch"):
        st.session_state["nl_query"] = _query
        _remember_place(_city, _country, _lat, _lon)
        st.rerun()

# Natural-language front door: any place on Earth, plain English.
nl_query = st.text_input(
    "Ask in plain language",
    key="nl_query",
    placeholder="How risky are heatwaves in Rourkela over the next 10 days?",
    help="Deterministic parsing → geocoding → AR6 region mapping → the agent. "
         "Unsupported hazards and unknown places refuse honestly.",
    on_change=_on_nl_query_submit,
)
ask = st.button("Ask", type="primary", icon=":material/travel_explore:")
nl_triggered = bool(ask or st.session_state.pop("_nl_submitted", False))

# When plain language inquiry is run, resolve its target location early so the
# map and coordinates immediately reflect the queried place.
if nl_triggered and nl_query.strip():
    from agent.nl import parse_query
    _parsed_early = parse_query(nl_query)
    if _parsed_early.place:
        try:
            _found_early = resolve_place(_parsed_early.place)
            _remember_place(
                _found_early.name,
                _found_early.country,
                _found_early.latitude,
                _found_early.longitude,
            )
        except GeocodeError:
            pass

with st.sidebar:
    st.markdown(
        """
        <div class="cra-sidebar-header">
            <div class="cra-sidebar-pill">
                <span class="cra-dot"></span>
                <span>CONTROL DECK</span>
            </div>
            <div class="cra-sidebar-title">Assessment Controls</div>
            <div class="cra-sidebar-desc">Configure coordinates, hazard domain, and climatology baseline manually.</div>
        </div>
        <div class="cra-section-label">
            <span>PRESET LOCATIONS</span>
            <span class="cra-fast-badge">CACHED</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _example_names = list(LOCATIONS)
    for _start in range(0, len(_example_names), 3):
        for _col, _name in zip(st.columns(3), _example_names[_start:_start + 3]):
            _city, _, _country = _name.partition(", ")
            if _col.button(_city, key=f"ex_{_name}", width="stretch", help=_name):
                _remember_place(_city, _country, *LOCATIONS[_name])
                st.rerun()

    place_query = st.text_input(
        "Place", key="place_query", placeholder="Any town, city or region on Earth",
        help="Geocoded via Open-Meteo. The resolved name, country and coordinates are "
             "shown on the map, so a wrong match is visible rather than silent.",
        on_change=_on_place_query_change,
    )
    if st.button("Find place", icon=":material/search:", width="stretch"):
        if _resolve_and_set_place(place_query):
            st.rerun()

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

    hazard = st.selectbox(
        "Hazard", list(Hazard), format_func=lambda h: h.value.replace("_", " ")
    )
    horizon = st.slider("Forecast horizon (days)", 1, 16, 7)
    use_climatology = st.checkbox(
        "Ground with ERA5 climatology (GEV return levels)", value=True,
        help="First call fetches 60+ years of daily extremes (~3 s), then cached.",
    )
    run = st.button(
        "Assess risk", key="assess", type="primary",
        icon=":material/troubleshoot:", width="stretch",
    )
    st.caption(
        "First report for a new place takes about 1 to 2 minutes: 60+ years of ERA5 "
        "extremes are fetched and fitted once, then cached. Repeat visits are instant."
    )

# Where the assessment will run. Display-only by design: Streamlit's map
# selection reports which OBJECTS a click picked (per-layer `indices` /
# `objects`), never the coordinates of the click itself, so a click cannot
# place a new point. The Place box and the coordinate inputs do that.
if selected is not None:
    with st.container(border=True):
        _map_col, _info_col = st.columns([2, 1])
        with _map_col:
            st.map(
                {"lat": [selected.latitude], "lon": [selected.longitude]},
                zoom=3, color="#E25C48", height=240,
            )
        with _info_col:
            st.markdown(f"**{selected.name}**")
            st.caption(
                " · ".join(
                    part for part in (
                        selected.country,
                        selected.region.label if selected.region else None,
                        f"{selected.latitude:.4f}, {selected.longitude:.4f}",
                    ) if part
                )
            )
            if selected.notice is not None:
                st.info(selected.notice, icon=":material/public_off:")

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
            head = f":green-badge[running] **{event.label}**"
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
        st.error(f"**Refused:** {report.refusal}", icon=":material/block:")
        st.caption("Out-of-scope is an explicit, valid output: not a fabricated risk.")
    else:
        color, icon = _LEVEL_STYLE[report.risk_level]

        # Collapsed by default: it answers "what am I looking at?" for a first
        # reader without pushing the actual report down the page for a repeat one.
        with st.expander("How to read this report", icon=":material/menu_book:"):
            st.markdown(
                """
- **Risk band**: where the forecast peak lands on *this location's own* ERA5
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
- **Projected change**: a *verbatim* AR6 Chapter 12 sentence for the reference
  region containing this point. Nothing is paraphrased, and an absence is stated
  in words rather than quietly filled in.
- **Limits**: what this cannot do, and where the numbers stop being valid, in
  [LIMITATIONS.md](https://github.com/AswaniSahoo/climate-risk-agent/blob/main/LIMITATIONS.md).
- **Cost & latency**: wall time, live model calls (cache hits excluded), token
  counts and an **estimated** dollar figure. Tokens are measured at the SDK seam;
  the dollars come from a price table, so they are an estimate, not a bill.
                """
            )

        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                st.subheader(
                    f"{report.risk_level.value.upper()} · "
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
                            f":gray-badge[{c.locator}]"
                        )
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
                        f":gray-badge[:material/public: {_pc.region_name} "
                        f"({_pc.region_acronym})] "
                        f":gray-badge[{_pc.direction.value.replace('_', ' ')}]"
                        + (
                            f" :green-badge[{_pc.confidence_language}]"
                            if _pc.confidence_language
                            else ""
                        )
                        + (
                            f" :orange-badge[{_pc.warming_level_or_period}]"
                            if _pc.warming_level_or_period
                            else ""
                        )
                    )
                    st.caption(f"“{_pc.statement}”")
                    for _c in _pc.citations:
                        st.markdown(
                            f":gray-badge[:material/description: {_c.source}] "
                            f":gray-badge[{_c.locator}]"
                        )
                else:
                    st.caption(
                        "Regional projections were not found in the corpus for this "
                        "hazard and AR6 region: nothing is asserted in their place."
                    )

        if report.hazard_stats:
            stat = report.hazard_stats[0]
            with st.container(border=True):
                st.markdown(f"**ERA5 climatology** ({stat.n_years} years, {stat.variable})")
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
                    f"Record max in series: {round(stat.record_max, 1)} · "
                    f"representativeness: {stat.representativeness.value}"
                )
                if stat.trend is not None:
                    if stat.trend.significant:
                        st.warning(
                            f"Warming trend detected: {stat.trend.slope_per_decade:+.1f} "
                            f"{stat.unit}/decade (p={stat.trend.p_value:.3f}). Return levels "
                            f"above are EFFECTIVE at {stat.trend.evaluated_at_year} "
                            "(today's climate, not the historical average).",
                            icon=":material/trending_up:",
                        )
                    else:
                        st.caption(
                            f"Non-stationarity tested: no significant trend "
                            f"(p={stat.trend.p_value:.2f}) · stationary fit reported."
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

elif failure is None and not run and not ask:
    col_pipeline, col_principles = st.columns([11, 9], gap="medium")
    with col_pipeline:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 0.9rem;">
                    <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--cra-badge-text); margin-bottom: 4px;">
                        EMPIRICAL EVALUATION PIPELINE
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 600; letter-spacing: -0.015em; margin-bottom: 6px; color: var(--cra-text);">
                        How the agent grounds risk assessments
                    </div>
                    <div style="font-size: 0.86rem; color: var(--cra-text-muted); line-height: 1.5;">
                        Every assessment cross-examines operational forecast signals against 60+ years of extreme value distributions and the IPCC AR6 assessment tables:
                    </div>
                </div>
                <div class="cra-pipeline-card">
                    <div class="cra-step-badge">LAYER 01 · OPERATIONAL FORECAST</div>
                    <div class="cra-step-title">Numerical Forecast Ensemble</div>
                    <div class="cra-step-desc">
                        Fetches high-resolution weather models (Open-Meteo) up to 16 days out for daily temperature extremes, heavy rainfall accumulation, or maximum wind gusts at your exact coordinate.
                    </div>
                </div>
                <div class="cra-pipeline-card">
                    <div class="cra-step-badge">LAYER 02 · HISTORICAL REANALYSIS</div>
                    <div class="cra-step-title">ERA5 Extreme Value Climatology</div>
                    <div class="cra-step-desc">
                        Fits 60+ years (1940 to 2023) of daily historical extremes using Generalized Extreme Value (GEV) parametric distributions. Calibrates local 10, 50, and 100-year return levels with 90% bootstrap confidence intervals and tests for non-stationary warming trends.
                    </div>
                </div>
                <div class="cra-pipeline-card">
                    <div class="cra-step-badge">LAYER 03 · PEER-REVIEWED SYNTHESIS</div>
                    <div class="cra-step-title">IPCC AR6 Grounding & Verification</div>
                    <div class="cra-step-desc">
                        Cross-references physical drivers against Chapter 11 and regional trend projections against Chapter 12. Every assertion requires machine-checked page citations; ungrounded statements are strictly omitted.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col_principles:
        with st.container(border=True):
            st.markdown(
                """
                <div style="margin-bottom: 0.75rem;">
                    <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.12em; color: var(--cra-badge-text); margin-bottom: 4px;">
                        DECISION PRINCIPLES
                    </div>
                    <div style="font-size: 1.1rem; font-weight: 600; letter-spacing: -0.015em; margin-bottom: 6px; color: var(--cra-text);">
                        Scientific Guarantees
                    </div>
                </div>
                <div class="cra-callout">
                    <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--cra-callout-red-text); margin-bottom: 2px;">
                        CLIMATOLOGICAL RARITY
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; color: var(--cra-text);">
                        Relative Severity vs. Raw Weather
                    </div>
                    <div style="font-size: 0.84rem; line-height: 1.48; color: var(--cra-text-muted);">
                        Conventional weather apps only report raw predictions. The Climate-Risk Analyst determines how extreme that forecast is relative to the historical climatology of that specific location. For instance, 38 °C in Berlin triggers a severe risk rating because it surpasses the local 50-year return level, whereas the same temperature in Rourkela represents expected seasonal weather.
                    </div>
                </div>
                <div class="cra-callout-green">
                    <div style="font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: var(--cra-callout-green-text); margin-bottom: 2px;">
                        AUDITED ABSTENTION
                    </div>
                    <div style="font-size: 0.95rem; font-weight: 600; margin-bottom: 4px; color: var(--cra-text);">
                        Zero Fabricated Answers
                    </div>
                    <div style="font-size: 0.84rem; line-height: 1.48; color: var(--cra-text-muted);">
                        Three physical hazards are supported: heatwaves, extreme precipitation, and wind. Out-of-scope hazards (cyclones, wildfires, drought, sea-level rise) and ungrounded queries trigger explicit typed refusals rather than hallucinated estimates.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
