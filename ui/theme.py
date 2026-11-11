"""The look of the app in one place: CSS tokens, the logo mark, small HTML pieces.

The world is a weather bureau's surface-analysis chart: neutral paper, black ink,
hairline rules and one red pencil. Surfaces stay neutral so that color can mean
something: the map pin, the risk band and the return-level plot.

Streamlit's own widgets are themed in .streamlit/config.toml. That file and the
tokens below carry the same values on purpose, so keep the two in step.

`light-dark()` resolves against the `color-scheme` Streamlit sets on `.stApp` from
the active theme, never against the visitor's OS. A `prefers-color-scheme` query
can disagree with the theme in use, which is how a light-mode OS once got dark
text on a dark page here.

Every value that reaches these HTML strings is escaped: place names come from a
geocoder and report text from the agent, and they all render through
`unsafe_allow_html`.

Design method after unslop by Damir Sokolovsky (@heydamir), CC BY 4.0.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from html import escape

from agent.contracts import Citation, RiskDriver, RiskLevel

# st.map takes one literal color, so this one value has to read on both basemaps.
PIN_RED = "#D1381F"

# The station circle of a surface chart, filled in oktas. Here the fill is the
# band, so a report reads in greyscale and for color-blind readers as well.
_LEVEL_OKTAS: dict[RiskLevel, int] = {
    RiskLevel.LOW: 2,
    RiskLevel.MODERATE: 4,
    RiskLevel.HIGH: 6,
    RiskLevel.SEVERE: 8,
}


def station_circle(level: RiskLevel, *, attrs: str = "") -> str:
    """A station circle whose fill shows the band. `attrs` sizes it when nested in a plot."""
    fraction = _LEVEL_OKTAS[level] / 8
    if fraction >= 1:
        fill = '<circle class="cra-station-fill" cx="12" cy="12" r="10"/>'
    else:
        angle = 2 * math.pi * fraction
        x, y = 12 + 10 * math.sin(angle), 12 - 10 * math.cos(angle)
        large_arc = 1 if fraction > 0.5 else 0
        fill = (
            f'<path class="cra-station-fill" '
            f'd="M12 12L12 2A10 10 0 {large_arc} 1 {x:.2f} {y:.2f}Z"/>'
        )
    return (
        f'<svg class="cra-station" viewBox="0 0 24 24" aria-hidden="true" {attrs}>'
        f'<circle class="cra-station-ring" cx="12" cy="12" r="10"/>{fill}</svg>'
    )


# The original logo's idea, open concentric arcs around a dot, drawn flat. On a
# surface chart it reads as isobars around a low. One ink, one red dot.
_MARK_SHAPES = (
    '<circle cx="16" cy="16" r="13" fill="none" stroke="{ink}" stroke-width="2" '
    'stroke-linecap="round" pathLength="100" stroke-dasharray="78 22" '
    'transform="rotate(-6 16 16)"/>'
    '<circle cx="16" cy="16" r="8" fill="none" stroke="{ink}" stroke-width="2" '
    'stroke-linecap="round" pathLength="100" stroke-dasharray="70 30" '
    'transform="rotate(190 16 16)"/>'
    '<circle cx="16" cy="16" r="3" fill="{dot}"/>'
)

MARK_SVG = (
    '<svg class="cra-brand-logo" viewBox="0 0 32 32" role="img" '
    'aria-label="Climate-Risk Analyst logo">'
    + _MARK_SHAPES.format(ink="currentColor", dot="var(--cra-red)")
    + "</svg>"
)

# Inline on purpose: .dockerignore and .gcloudignore both exclude assets/, so a
# file-based tab icon never reached the deployed image. The paper tile keeps the
# ink mark visible on a dark browser tab.
FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="6" fill="#F7F7F4"/>'
    + _MARK_SHAPES.format(ink="#1C1C1A", dot="#B8321A")
    + "</svg>"
)


def band_heading(level: RiskLevel) -> str:
    """The answer: the risk band, set as the largest thing in a report."""
    name = level.value
    return (
        f'<div class="cra-band cra-lvl-{name}">'
        f"{station_circle(level)}<span>{escape(name.capitalize())}</span></div>"
    )


def driver_list(drivers: Iterable[RiskDriver]) -> str:
    """Drivers as a log: factor on the left in mono, what it says on the right."""
    rows = "".join(
        f'<div class="cra-driver"><dt>{escape(d.factor)}</dt><dd>{escape(d.detail)}</dd></div>'
        for d in drivers
    )
    return f'<dl class="cra-drivers">{rows}</dl>'


# What a reader sees for each evidence strength (docs/jev-ui-integration.md). An
# unchecked page gets no label at all: silence is not a verdict.
EVIDENCE_LABELS: dict[str, str] = {
    "strong": "Supports the statement",
    "moderate": "Partly supports",
    "weak": "Does not clearly support this statement",
    "contradicted": "May contradict the statement",
}


def citation_list(
    citations: Iterable[Citation], evidence: Mapping[tuple[str, str], str] | None = None
) -> str:
    """One row per citation, in report order, with the pages in one right-aligned column.

    `evidence` maps (source, locator) to a strength from the citation check. It
    only adds a label under a row: no row is hidden, added or moved by it.
    """
    rows = []
    for c in citations:
        strength = (evidence or {}).get((c.source, c.locator), "")
        label = EVIDENCE_LABELS.get(strength)
        tag = (f'<span class="cra-cite-ev cra-ev-{strength}">{escape(label)}</span>'
               if label else "")
        rows.append(
            f'<div class="cra-cite"><span class="cra-cite-src">{escape(c.source)}</span>'
            f'<span class="cra-cite-loc">{escape(c.locator)}</span>{tag}</div>'
        )
    return f'<div class="cra-cites">{"".join(rows)}</div>'


def prose_with_flags(prose: str, flagged: Iterable[str], note: str) -> tuple[str, list[str]]:
    """The IPCC prose as one paragraph, each flagged sentence marked where it stands.

    The mark is a dotted underline with `note` as its hover text and, for a screen
    reader, the same note spoken after the sentence. Returns the HTML and the
    flagged sentences it could not find in the prose, so a caller can list them.
    """
    flat = re.sub(r"\s+", " ", prose).strip()
    spans: list[tuple[int, int]] = []
    missing: list[str] = []
    for sentence in flagged:
        at = flat.find(sentence) if sentence else -1
        end = at + len(sentence)
        if at < 0 or any(at < stop and start < end for start, stop in spans):
            missing.append(sentence)
            continue
        spans.append((at, end))
    parts, pos = [], 0
    for start, stop in sorted(spans):
        parts.append(escape(flat[pos:start]))
        parts.append(
            f'<span class="cra-unsupported" title="{escape(note)}">{escape(flat[start:stop])}'
            f'<span class="cra-sr-only"> ({escape(note)})</span></span>'
        )
        pos = stop
    parts.append(escape(flat[pos:]))
    return f'<p class="cra-prose">{"".join(parts)}</p>', missing


def question_reading(hazard: str, horizon_days: int, place: str, note: str | None) -> str:
    """How the question was read, shown before the report runs: hazard, horizon, place."""
    days = f"next {horizon_days} day{'' if horizon_days == 1 else 's'}"
    parts = (f'<span class="cra-read-hazard">{escape(hazard)}</span>', escape(days), escape(place))
    note_html = f'<p class="cra-read-note">{escape(note)}</p>' if note else ""
    return (
        '<div class="cra-read"><h2 class="cra-read-title">How your question was read</h2>'
        f'<p class="cra-read-line">{" · ".join(parts)}</p>{note_html}</div>'
    )


def quoted_statement(statement: str) -> str:
    """A verbatim IPCC sentence, set as a quotation rather than as small print."""
    return f'<blockquote class="cra-quote">“{escape(statement)}”</blockquote>'


def plate_caption(
    name: str, country: str, region_label: str | None, latitude: float, longitude: float
) -> str:
    """The line under the map: which point the assessment will run on."""
    meta = " · ".join(part for part in (country, region_label) if part)
    meta_html = f'<div class="cra-plate-meta">{escape(meta)}</div>' if meta else ""
    return (
        '<div class="cra-plate">'
        '<div class="cra-plate-kicker">Where the assessment will run</div>'
        '<div class="cra-plate-row">'
        f'<span class="cra-plate-name">{escape(name)}</span>'
        f'<span class="cra-plate-coords">{latitude:.4f}, {longitude:.4f}</span>'
        f"</div>{meta_html}</div>"
    )


def method_strip(steps: Iterable[tuple[str, str]]) -> str:
    """The method as three numbered steps, one line each. The full account sits in an expander."""
    items = "".join(
        f'<div class="cra-step" role="listitem"><span class="cra-step-n">{n}</span>'
        f'<span class="cra-step-name">{escape(name)}</span>'
        f'<span class="cra-step-line">{escape(line)}</span></div>'
        for n, (name, line) in enumerate(steps, start=1)
    )
    return f'<div class="cra-steps" role="list" aria-label="How a report is built">{items}</div>'


def specimen_card(level: RiskLevel, plot_html: str) -> str:
    """The first screen's preview of a report: the real band heading, plot style and
    citation rows, around a schematic reading. The card says it is not a forecast."""
    sources = "".join(
        f'<div class="cra-cite"><span class="cra-cite-src">{escape(src)}</span>'
        f'<span class="cra-cite-loc">page-level</span></div>'
        for src in ("IPCC AR6 WGI Chapter 11", "IPCC AR6 WGI Chapter 12")
    )
    return (
        f'<div class="cra-specimen cra-lvl-{level.value}">'
        '<div class="cra-specimen-head"><h2 class="cra-specimen-title">Example report</h2>'
        '<span class="cra-specimen-meta">Schematic, not a forecast</span></div>'
        f"{band_heading(level)}"
        '<div class="cra-specimen-sub">Heatwave risk at your point, next 7 days</div>'
        f'{plot_html}<div class="cra-cites">{sources}</div></div>'
    )


def why_line(sentence: str) -> str:
    """The one sentence that says why the band is what it is, set under the answer."""
    return f'<p class="cra-why">{escape(sentence)}</p>'


# No "<" anywhere inside the style text, comments included. st.html sanitizes with
# DOMPurify, which silently drops a whole <style> whose text looks like markup.
APP_CSS = """
<style>
:root {
  --cra-ink: light-dark(#1C1C1A, #ECECE6);
  --cra-ink-2: light-dark(#54544E, #ABABA4);
  --cra-ink-3: light-dark(#696962, #8F8F88);
  --cra-paper: light-dark(#F7F7F4, #161614);
  --cra-line: light-dark(rgb(28 28 26 / 13%), rgb(236 236 230 / 13%));
  --cra-line-strong: light-dark(rgb(28 28 26 / 30%), rgb(236 236 230 / 30%));
  --cra-red: light-dark(#B8321A, #E8674A);
  --cra-band-low: light-dark(#2F6B45, #7FB08A);
  --cra-band-moderate: light-dark(#8A6200, #D4A83A);
  --cra-band-high: light-dark(#A84A08, #E58A45);
  --cra-band-severe: var(--cra-red);
  --cra-mono: "Source Code Pro", ui-monospace, monospace;
}
.cra-lvl-low { --cra-lvl: var(--cra-band-low); }
.cra-lvl-moderate { --cra-lvl: var(--cra-band-moderate); }
.cra-lvl-high { --cra-lvl: var(--cra-band-high); }
.cra-lvl-severe { --cra-lvl: var(--cra-band-severe); }

/* Streamlit gives markdown a -1rem margin to cancel a paragraph's bottom margin. Our
   HTML blocks have no such margin, so without this they run into the next element. */
[data-testid="stMarkdownContainer"]:has(> [class*="cra-"]) { margin-bottom: 0; }

/* Captions are a text role, not an opacity */
[data-testid="stCaptionContainer"] { opacity: 1; color: var(--cra-ink-3); }

/* Page and sidebar rhythm */
[data-testid="stMainBlockContainer"] {
  padding: 56px 32px 64px;
  max-width: 1480px;
}
[data-testid="stSidebarHeader"] { height: 44px; }
[data-testid="stSidebarUserContent"] { padding-bottom: 32px; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 56px 16px 48px; }
}

/* Primary actions are ink, never red: in a risk tool a red button reads as an alarm */
button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"]:focus:not(:active) {
  background: var(--cra-ink);
  border-color: var(--cra-ink);
  color: var(--cra-paper);
}
button[data-testid="stBaseButton-primary"]:hover {
  background: var(--cra-ink-2);
  border-color: var(--cra-ink-2);
  color: var(--cra-paper);
}
button[data-testid="stBaseButton-primary"]:focus-visible {
  outline: 2px solid var(--cra-red);
  outline-offset: 2px;
}

/* Keyboard focus the theme would otherwise leave faint or invisible */
[data-testid="stTextInputRootElement"]:focus-within,
[data-testid="stNumberInputContainer"]:focus-within,
[data-testid="stTextAreaRootElement"]:focus-within,
[data-testid="stSelectbox"] .react-aria-ComboBox > div:focus-within {
  box-shadow: 0 0 0 1px var(--cra-red);
}
section[data-testid="stMain"]:focus-visible,
[data-testid="stDeckGlJsonChart"] canvas:focus-visible {
  outline: 2px solid var(--cra-red);
  outline-offset: -2px;
}
/* Read by screen readers, not drawn */
.cra-sr-only {
  position: absolute;
  width: 1px;
  height: 1px;
  margin: -1px;
  padding: 0;
  overflow: hidden;
  clip: rect(0 0 0 0);
  white-space: nowrap;
  border: 0;
}
/* Streamlit adds a hidden "Link to heading" anchor to every heading: one extra tab stop each */
[data-testid="stHeaderActionElements"] { display: none; }

/* Figures */
[data-testid="stMetricValue"] {
  font-family: var(--cra-mono);
  font-variant-numeric: tabular-nums;
}
[data-testid="stMetricValue"] > div { overflow: visible; }

/* Header */
.cra-head { display: flex; align-items: center; gap: 12px; }
.cra-brand-logo { width: 40px; height: 40px; flex: none; color: var(--cra-ink); }
h1.cra-title {
  margin: 0;
  padding: 0;
  font-size: 30px;
  line-height: 36px;
  font-weight: 500;
  color: var(--cra-ink);
}
p.cra-lede {
  margin: 12px 0 0;
  max-width: 52ch;
  font-size: 17px;
  line-height: 26px;
  color: var(--cra-ink-2);
}
@media (max-width: 640px) {
  h1.cra-title { font-size: 26px; line-height: 32px; }
  p.cra-lede { font-size: 16px; line-height: 24px; }
}
/* While a report is on the page, the question bar stays one reading width */
.st-key-cra_lead_wide { max-width: 760px; }

/* Method: three steps on one hairline, the full account in the expander under it */
.cra-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0 24px;
  border-top: 1px solid var(--cra-line);
}
.cra-step {
  display: grid;
  grid-template-columns: 16px minmax(0, 1fr);
  align-content: start;
  gap: 0 8px;
  padding-top: 12px;
}
.cra-step-n {
  grid-row: span 2;
  font-family: var(--cra-mono);
  font-size: 12px;
  line-height: 20px;
  font-variant-numeric: tabular-nums;
  color: var(--cra-ink-3);
}
.cra-step-name { font-size: 14px; line-height: 20px; font-weight: 500; color: var(--cra-ink); }
.cra-step-line { font-size: 13px; line-height: 20px; color: var(--cra-ink-2); }
@media (max-width: 640px) {
  .cra-steps { grid-template-columns: minmax(0, 1fr); }
  .cra-step { padding: 8px 0; border-bottom: 1px solid var(--cra-line); }
}

/* Example report: the first screen's one moment */
.st-key-cra_specimen { padding: 20px 20px 16px; }
.cra-specimen-head {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 4px 16px;
  margin-bottom: 12px;
}
h2.cra-specimen-title { margin: 0; padding: 0; font-size: 14px; line-height: 20px; font-weight: 500; color: var(--cra-ink); }
.cra-specimen-meta {
  font-family: var(--cra-mono);
  font-size: 12px;
  line-height: 16px;
  font-weight: 450;
  color: var(--cra-ink-3);
}
.cra-specimen .cra-band { font-size: 32px; line-height: 40px; }
.cra-specimen .cra-band .cra-station { width: 24px; height: 24px; }
.cra-specimen-sub { margin: 0 0 12px; font-size: 15px; line-height: 22px; color: var(--cra-ink-2); }
.cra-specimen .cra-plot { margin-bottom: 12px; }

/* Sidebar */
h2.cra-side-title { margin: 0; padding: 0; font-size: 16px; line-height: 24px; font-weight: 500; color: var(--cra-ink); }
.cra-side-label {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 14px;
  line-height: 20px;
  color: var(--cra-ink);
}
.cra-side-meta {
  font-family: var(--cra-mono);
  font-size: 12px;
  font-weight: 450;
  color: var(--cra-ink-3);
}
.st-key-cra_presets {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
[class*="st-key-ex_"] button { padding-inline: 4px; min-height: 32px; }

/* Example questions: an even grid, not a ragged wrap */
.st-key-cra_chips {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(max(216px, calc(50% - 8px)), 1fr));
  gap: 0 16px;
}
.st-key-cra_chips button { min-height: 32px; justify-content: flex-start; }

/* Map plate: the first screen's one moment, so the map runs to the frame */
.st-key-cra_plate { padding: 0; gap: 0; overflow: hidden; }
.st-key-cra_plate_info { padding: 12px 16px 16px; gap: 12px; }
.cra-plate-kicker {
  margin-bottom: 4px;
  font-family: var(--cra-mono);
  font-size: 12px;
  line-height: 16px;
  font-weight: 450;
  color: var(--cra-ink-3);
}
.cra-plate-row {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: baseline;
  gap: 0 16px;
}
.cra-plate-name { font-size: 20px; line-height: 28px; font-weight: 500; color: var(--cra-ink); }
.cra-plate-coords {
  font-family: var(--cra-mono);
  font-size: 13px;
  font-variant-numeric: tabular-nums;
  color: var(--cra-ink-2);
}
.cra-plate-meta { font-size: 14px; line-height: 20px; color: var(--cra-ink-2); }

/* The answer */
.cra-band {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 48px;
  line-height: 56px;
  font-weight: 500;
  letter-spacing: -0.02em;
  color: var(--cra-lvl);
}
.cra-band .cra-station { width: 36px; height: 36px; flex: none; }
/* Band, then place, then the one sentence of why: three steps down in size */
.st-key-cra_verdict h3 { padding: 4px 0 0; font-size: 24px; line-height: 32px; }
p.cra-why { margin: 8px 0; max-width: 60ch; font-size: 17px; line-height: 26px; color: var(--cra-ink); }
.cra-station { color: var(--cra-lvl); }
.cra-station-ring { fill: var(--cra-paper); stroke: currentColor; stroke-width: 2; }
.cra-station-fill { fill: currentColor; }
.st-key-cra_verdict > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
  > [data-testid="stColumn"] { min-width: min(100%, 360px); }
.st-key-cra_verdict > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
  > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"] { height: 100%; }
.st-key-cra_verdict > [data-testid="stLayoutWrapper"] > [data-testid="stHorizontalBlock"]
  > [data-testid="stColumn"]:first-child > [data-testid="stVerticalBlock"] > :last-child {
  margin-top: auto;
}
/* Long prose is set in two columns, the way a printed bulletin would */
.st-key-cra_prose { max-width: 76ch; }
.st-key-cra_prose_long p { column-width: 40ch; column-gap: 32px; }

/* A refusal is a finished answer, not an app failure, so it is ink and not alarm */
.st-key-cra_refusal [data-testid="stAlertContainer"] {
  background: transparent;
  color: var(--cra-ink);
  border: 1px solid var(--cra-line-strong);
}

/* Return-level plot */
.cra-plot { margin: 0; max-width: 640px; container-type: inline-size; }
.cra-plot-title { font-size: 14px; line-height: 20px; font-weight: 600; color: var(--cra-ink); }
.cra-plot-sub { margin-bottom: 8px; font-size: 13px; line-height: 20px; color: var(--cra-ink-3); }
.cra-plot-note { margin: 8px 0 0; font-size: 13px; line-height: 20px; color: var(--cra-ink-3); }
.cra-plot-chart { display: block; width: 100%; height: auto; overflow: visible; }
.cra-plot-compact { display: none; }
@container (max-width: 419px) {
  .cra-plot-full { display: none; }
  .cra-plot-compact { display: block; }
}
.cra-plot-chart text {
  font-family: var(--cra-mono);
  font-weight: 450;
  font-variant-numeric: tabular-nums;
  fill: var(--cra-ink-2);
  paint-order: stroke;
  stroke: var(--cra-paper);
  stroke-width: 3px;
  stroke-linejoin: round;
}
.cra-plot-chart text.t-axis, .cra-plot-chart text.t-record { fill: var(--cra-ink-3); }
.cra-plot-chart text.t-point { fill: var(--cra-ink); }
.cra-plot-chart text.t-peak { fill: var(--cra-lvl); font-weight: 500; }
.cra-plot-chart text.t-band { fill: var(--cra-ink-3); stroke: none; }
.cra-plot-chart text.t-band-on { fill: var(--cra-lvl); font-weight: 500; stroke: none; }
.cra-plot-chart .m-grid { stroke: var(--cra-line); stroke-width: 1; }
.cra-plot-chart .m-edge { stroke: var(--cra-line-strong); stroke-width: 1; }
.cra-plot-chart .m-ribbon { fill: var(--cra-ink); fill-opacity: 0.09; }
.cra-plot-chart .m-curve { fill: none; stroke: var(--cra-ink); stroke-width: 1.75; }
.cra-plot-chart .m-point { fill: var(--cra-paper); stroke: var(--cra-ink); stroke-width: 1.5; }
.cra-plot-chart .m-record { stroke: var(--cra-ink-3); stroke-width: 1; stroke-dasharray: 2 3; }
.cra-plot-chart .m-peak { stroke: var(--cra-lvl); stroke-width: 1.75; }
.cra-plot-chart .m-drop { stroke: var(--cra-lvl); stroke-width: 1; stroke-dasharray: 2 3; }
.cra-plot-chart .m-band-on { fill: var(--cra-lvl); fill-opacity: 0.14; }

/* Report rows */
.cra-drivers { margin: 0; }
.cra-driver {
  display: grid;
  grid-template-columns: 112px minmax(0, 1fr);
  gap: 0 16px;
  padding: 8px 0;
  border-top: 1px solid var(--cra-line);
}
.cra-driver dt {
  font-family: var(--cra-mono);
  font-size: 12px;
  line-height: 20px;
  font-weight: 450;
  color: var(--cra-ink-3);
}
.cra-driver dd { margin: 0; font-size: 14px; line-height: 20px; color: var(--cra-ink); }
.cra-cite {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 4px 0;
  border-top: 1px solid var(--cra-line);
  font-family: var(--cra-mono);
  font-size: 12px;
  line-height: 20px;
  font-weight: 450;
  color: var(--cra-ink-2);
}
.cra-cite-src { min-width: 0; overflow-wrap: anywhere; }
.cra-cite-loc { flex: none; font-variant-numeric: tabular-nums; color: var(--cra-ink); }
blockquote.cra-quote {
  margin: 0;
  padding: 0 0 0 12px;
  border-left: 2px solid var(--cra-line-strong);
  opacity: 1;
  font-size: 14px;
  line-height: 20px;
  color: var(--cra-ink);
}

/* Evidence labels (CRG_JEV=verify): a line under the page they describe, never a reorder */
.cra-cite { flex-wrap: wrap; row-gap: 0; }
.cra-cite-ev { flex-basis: 100%; color: var(--cra-ink-2); }
.cra-ev-strong { color: var(--cra-ink); }
.cra-ev-weak { color: var(--cra-ink-3); }
.cra-ev-contradicted { color: var(--cra-red); font-weight: 600; }
.cra-unsupported {
  text-decoration: underline dotted var(--cra-red);
  text-decoration-thickness: 2px;
  text-underline-offset: 4px;
}

/* How the question was read (CRG_JEV=triage): under the question, before the answer */
.st-key-cra_reading { max-width: 760px; gap: 8px; }
.cra-read { padding-top: 12px; border-top: 1px solid var(--cra-line); }
h2.cra-read-title {
  margin: 0 0 4px;
  padding: 0;
  font-family: var(--cra-mono);
  font-size: 12px;
  line-height: 16px;
  font-weight: 450;
  color: var(--cra-ink-3);
}
p.cra-read-line { margin: 0; font-size: 17px; line-height: 26px; color: var(--cra-ink); }
.cra-read-hazard { font-weight: 600; }
p.cra-read-note { margin: 4px 0 0; max-width: 64ch; font-size: 14px; line-height: 20px; color: var(--cra-ink-2); }
.st-key-cra_reading_alts { flex-wrap: wrap; gap: 0 16px; }
.st-key-cra_reading_alts button { min-height: 32px; }
@media (max-width: 640px) {
  .cra-driver { grid-template-columns: minmax(0, 1fr); }
  .cra-band { font-size: 40px; line-height: 48px; }
  .st-key-cra_verdict h3 { font-size: 20px; line-height: 28px; }
}
</style>
"""
