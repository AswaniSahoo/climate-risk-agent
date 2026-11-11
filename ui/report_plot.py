"""The return-level plot: the report's one chart, and the product's idea drawn.

A risk band is decided by where the forecast peak lands on this location's own
return-level curve (agent/risk_bands.py). This draws exactly that reading: the
fitted levels, the peak as a line running across until it meets the curve, then
down into the band it falls in.

Honest by construction:

- The x axis is log return period and the segments are straight, which is the
  same "linear in log(T)" interpolation `band_from_return_period` grades with.
  The crossing is the agent's own reading, not a second opinion.
- Nothing is extrapolated. A peak past either end of the fitted curve sits in the
  open zone beyond that end, the same way the agent reads it off that edge.
- The peak comes from the report's `severity_basis` driver, because RiskReport has
  no numeric field for it. The absolute-cutoff fallback words that driver
  differently, so a report not graded on this curve gets the curve with no marker
  and no highlighted band. tests/test_ui.py pins the sentence this parses.

Plain strings and no Streamlit import, so all of this can be unit-tested.
"""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from html import escape

from agent.contracts import RiskLevel, RiskReport
from agent.risk_bands import _BAND_EDGES
from tools.hazard_stats import HazardStat, ReturnLevel
from ui.theme import station_circle

_PEAK_SENTENCE = re.compile(r"^Forecast peak (-?\d+(?:\.\d+)?(?:e[+-]?\d+)?) (\S+) ")


@dataclass(frozen=True)
class _Frame:
    """One drawing size. Text is never scaled small: narrow columns get their own frame."""

    css_class: str
    width: int
    height: int
    left: int
    right: int
    top: int = 8
    bottom: int = 56  # tick row, then the band strip
    font_size: int = 12


_FULL = _Frame("cra-plot-full", width=480, height=300, left=44, right=12)
_COMPACT = _Frame("cra-plot-compact", width=320, height=280, left=36, right=8)
_STRIP_HEIGHT = 24


def _text_width(text: str, font_size: float) -> float:
    """Advance width of a label: the plot's text is monospaced, 0.6 em per character."""
    return len(text) * font_size * 0.6


@dataclass(frozen=True)
class _Box:
    """The rough rectangle a label occupies, enough to keep labels off each other."""

    left: float
    right: float
    top: float
    bottom: float

    def overlaps(self, other: "_Box") -> bool:
        return (
            self.left < other.right
            and other.left < self.right
            and self.top < other.bottom
            and other.top < self.bottom
        )


def _box(x: float, baseline: float, text: str, font_size: float, *, anchor_end: bool = False) -> _Box:
    width = _text_width(text, font_size)
    left = x - width if anchor_end else x
    return _Box(left, left + width, baseline - font_size, baseline + 3)


def forecast_peak(report: RiskReport, stat: HazardStat) -> tuple[float, str] | None:
    """(value, as written) of the peak this report was graded with on `stat`, else None."""
    for driver in report.drivers:
        if driver.factor != "severity_basis":
            continue
        match = _PEAK_SENTENCE.match(driver.detail)
        if match and match.group(2) == stat.unit:
            return float(match.group(1)), match.group(1)
    return None


def _usable_curve(stat: HazardStat) -> list[ReturnLevel] | None:
    """The fitted levels in return-period order, or None if they cannot be drawn."""
    curve = sorted(stat.return_levels, key=lambda r: r.return_period_years)
    if len(curve) < 2 or curve[0].return_period_years <= 0:
        return None
    if not all(math.isfinite(r.level) for r in curve):
        return None
    return curve


def _has_ribbon(curve: Sequence[ReturnLevel]) -> bool:
    for r in curve:
        if r.ci_low is None or r.ci_high is None:
            return False
        if not (math.isfinite(r.ci_low) and math.isfinite(r.ci_high)):
            return False
    return True


def _y_ticks(values: Sequence[float], target: int = 4) -> list[float]:
    """Round ticks that span `values`, about `target` intervals of them."""
    low, high = min(values), max(values)
    raw = (high - low or 1.0) / target
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if raw <= m * magnitude)
    first = math.floor(low / step)
    last = math.ceil(high / step)
    return [i * step for i in range(first, last + 1)]


def _crossing_x(
    value: float, curve: Sequence[ReturnLevel], x_of: Callable[[float], float],
    domain: tuple[float, float],
) -> float | None:
    """Where the peak meets the curve, using the same bracket the agent uses."""
    first, last = curve[0], curve[-1]
    if value < first.level:
        return (x_of(domain[0]) + x_of(first.return_period_years)) / 2
    if value >= last.level:
        return (x_of(last.return_period_years) + x_of(domain[1])) / 2
    for a, b in zip(curve, curve[1:]):
        if a.level <= value < b.level:
            start, end = x_of(a.return_period_years), x_of(b.return_period_years)
            return start + (value - a.level) / (b.level - a.level) * (end - start)
    return None  # a non-monotone curve: draw it, but claim no reading from it


def _drawing(
    frame: _Frame, stat: HazardStat, curve: Sequence[ReturnLevel],
    peak: tuple[float, str] | None, level: RiskLevel | None, label: str,
) -> str:
    plot_left, plot_right = frame.left, frame.width - frame.right
    plot_top, plot_bottom = frame.top, frame.height - frame.bottom

    domain = (
        min(1.0, curve[0].return_period_years / 2),
        max(200.0, curve[-1].return_period_years * 2),
    )
    log_min, log_span = math.log(domain[0]), math.log(domain[1] / domain[0])

    def x_of(period: float) -> float:
        return plot_left + (math.log(period) - log_min) / log_span * (plot_right - plot_left)

    ribbon = _has_ribbon(curve)
    values = [r.level for r in curve] + [stat.record_max]
    if ribbon:
        values += [r.ci_low for r in curve] + [r.ci_high for r in curve]
    if peak is not None:
        values.append(peak[0])
    ticks = _y_ticks(values)
    y_min, y_max = ticks[0], ticks[-1]

    def y_of(value: float) -> float:
        return plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)

    parts: list[str] = []

    for tick in ticks:
        y = y_of(tick)
        parts.append(
            f'<line class="m-grid" x1="{plot_left}" x2="{plot_right}" y1="{y:.1f}" y2="{y:.1f}"/>'
        )
        parts.append(
            f'<text class="t-axis" x="{plot_left - 6}" y="{y + 4:.1f}" text-anchor="end">'
            f"{tick:g}</text>"
        )

    if ribbon:
        upper = [f"{x_of(r.return_period_years):.1f},{y_of(r.ci_high):.1f}" for r in curve]
        lower = [f"{x_of(r.return_period_years):.1f},{y_of(r.ci_low):.1f}" for r in reversed(curve)]
        parts.append(f'<polygon class="m-ribbon" points="{" ".join(upper + lower)}"/>')

    crossing = _crossing_x(peak[0], curve, x_of, domain) if peak is not None else None
    drawn_peak = peak is not None and crossing is not None
    strip_top = plot_bottom + 24

    if drawn_peak and level is not None:
        edges = [domain[0], *(edge for edge, _ in _BAND_EDGES), domain[1]]
        bands = [*(band for _, band in _BAND_EDGES), RiskLevel.SEVERE]
        for edge in edges[1:-1]:
            x = x_of(edge)
            parts.append(
                f'<line class="m-edge" x1="{x:.1f}" x2="{x:.1f}" y1="{plot_top}" '
                f'y2="{strip_top + _STRIP_HEIGHT}"/>'
            )
        for start, end, band in zip(edges, edges[1:], bands):
            left, right = x_of(start), x_of(end)
            active = band is level
            if active:
                parts.append(
                    f'<rect class="m-band-on" x="{left:.1f}" y="{strip_top}" '
                    f'width="{right - left:.1f}" height="{_STRIP_HEIGHT}"/>'
                )
            parts.append(
                f'<text class="{"t-band-on" if active else "t-band"}" '
                f'x="{(left + right) / 2:.1f}" y="{strip_top + 16}" text-anchor="middle">'
                f"{escape(band.value.capitalize())}</text>"
            )

    record_y = y_of(stat.record_max)
    parts.append(
        f'<line class="m-record" x1="{plot_left}" x2="{plot_right}" '
        f'y1="{record_y:.1f}" y2="{record_y:.1f}"/>'
    )

    points = " ".join(f"{x_of(r.return_period_years):.1f},{y_of(r.level):.1f}" for r in curve)
    parts.append(f'<polyline class="m-curve" points="{points}"/>')

    peak_y = None
    if drawn_peak and peak is not None and crossing is not None:
        peak_y = y_of(peak[0])
        parts.append(
            f'<line class="m-drop" x1="{crossing:.1f}" x2="{crossing:.1f}" '
            f'y1="{peak_y:.1f}" y2="{strip_top}"/>'
        )
        parts.append(
            f'<line class="m-peak" x1="{plot_left}" x2="{crossing:.1f}" '
            f'y1="{peak_y:.1f}" y2="{peak_y:.1f}"/>'
        )

    # Labels, placed by what the geometry leaves free. A return-level curve rises to the
    # right, so the area above the curve to its left and below it to its right stay open.
    font = frame.font_size
    placed: list[_Box] = []

    if drawn_peak and peak is not None and crossing is not None and peak_y is not None:
        text = f"Forecast peak {peak[1]} {stat.unit}"
        width = _text_width(text, font)
        if crossing - plot_left >= width + 12 or crossing + 12 + width > plot_right:
            label_x = plot_left + 4.0  # left of the crossing, above the line
            label_y = peak_y + 16 if peak_y - 6 < plot_top + 10 else peak_y - 6
        else:
            label_x = crossing + 12  # no room there: right of it, below the line
            label_y = peak_y + 16 if peak_y + 16 <= plot_bottom else peak_y - 6
        placed.append(_box(label_x, label_y, text, font))
        parts.append(
            f'<text class="t-peak" x="{label_x:.1f}" y="{label_y:.1f}">{escape(text)}</text>'
        )

    record_text = f"Record {stat.record_max:.1f}"
    record_label_y = record_y + 15 if record_y - 6 < plot_top + 10 else record_y - 6
    record_box = _box(plot_left + 4.0, record_label_y, record_text, font)
    if any(record_box.overlaps(other) for other in placed):
        right_box = _box(plot_right, record_y - 6, record_text, font, anchor_end=True)
        placed.append(right_box)
        parts.append(
            f'<text class="t-record" x="{plot_right}" y="{record_y - 6:.1f}" text-anchor="end">'
            f"{escape(record_text)}</text>"
        )
    else:
        placed.append(record_box)
        parts.append(
            f'<text class="t-record" x="{plot_left + 4}" y="{record_label_y:.1f}">'
            f"{escape(record_text)}</text>"
        )

    for r in curve:
        x, y = x_of(r.return_period_years), y_of(r.level)
        parts.append(f'<circle class="m-point" cx="{x:.1f}" cy="{y:.1f}" r="3"/>')
        parts.append(
            f'<text class="t-axis" x="{x:.1f}" y="{plot_bottom + 16}" text-anchor="middle">'
            f"{r.return_period_years}</text>"
        )
        text = f"{r.level:.1f}"
        below_right = _box(x + 6, y + 15, text, font)
        if any(below_right.overlaps(other) for other in placed):
            upper_left = _box(x - 6, y - 8, text, font, anchor_end=True)
            if not any(upper_left.overlaps(other) for other in placed):
                placed.append(upper_left)
                parts.append(
                    f'<text class="t-point" x="{x - 6:.1f}" y="{y - 8:.1f}" text-anchor="end">'
                    f"{text}</text>"
                )
                continue
        placed.append(below_right)
        parts.append(f'<text class="t-point" x="{x + 6:.1f}" y="{y + 15:.1f}">{text}</text>')

    if drawn_peak and level is not None and crossing is not None and peak_y is not None:
        parts.append(
            station_circle(
                level,
                attrs=f'x="{crossing - 8:.1f}" y="{peak_y - 8:.1f}" width="16" height="16"',
            )
        )

    return (
        f'<svg class="cra-plot-chart {frame.css_class}" '
        f'viewBox="0 0 {frame.width} {frame.height}" font-size="{frame.font_size}" '
        f'role="img" aria-label="{escape(label, quote=True)}">{"".join(parts)}</svg>'
    )


def return_level_plot(report: RiskReport) -> str | None:
    """The plot as one HTML string, or None when the report has no curve to draw."""
    if not report.hazard_stats:
        return None
    stat = report.hazard_stats[0]
    curve = _usable_curve(stat)
    if curve is None:
        return None

    peak = forecast_peak(report, stat)
    unit = escape(stat.unit)
    level = report.risk_level if peak is not None else None

    levels_text = ", ".join(
        f"{r.return_period_years}-year {r.level:.1f} {stat.unit}" for r in curve
    )
    basis = next((d.detail for d in report.drivers if d.factor == "severity_basis"), "")
    label = f"Return levels for this location: {levels_text}." + (
        f" {basis}" if peak is not None else ""
    )

    title = (
        "Forecast peak on this location's return-level curve"
        if peak is not None
        else "This location's return-level curve"
    )
    notes = ["Straight segments between fitted levels, the same reading the band uses."]
    if _has_ribbon(curve):
        notes.append("The ribbon is the 90% bootstrap CI.")
    trend = stat.trend
    if trend is not None and trend.significant and trend.evaluated_at_year is not None:
        notes.append(f"Levels effective at {trend.evaluated_at_year}.")
    if peak is None:
        notes.append("No peak is marked: this report's band was not read from this curve.")

    level_class = f" cra-lvl-{level.value}" if level is not None else ""
    return (
        f'<figure class="cra-plot{level_class}"><figcaption>'
        f'<div class="cra-plot-title">{escape(title)}</div>'
        f'<div class="cra-plot-sub">Level in {unit}, by return period in years (log scale)</div>'
        "</figcaption>"
        f"{_drawing(_FULL, stat, curve, peak, level, label)}"
        f"{_drawing(_COMPACT, stat, curve, peak, level, label)}"
        f'<div class="cra-plot-note">{escape(" ".join(notes))}</div></figure>'
    )


def _specimen_svg(frame: _Frame, peak: float, level: RiskLevel, label: str) -> str:
    left, right = frame.left, frame.width - frame.right
    top, bottom = frame.top, frame.height - frame.bottom
    strip_top = bottom + 24
    span = math.log(200.0)

    def x_of(period: float) -> float:
        return left + math.log(period) / span * (right - left)

    def y_of(fraction: float) -> float:
        return bottom - fraction * (bottom - top)

    def shape(period: float) -> float:
        return 0.12 + 0.78 * (math.log(period) / span) ** 0.8

    crossing = x_of(math.exp(((peak - 0.12) / 0.78) ** 1.25 * span))
    peak_y = y_of(peak)
    parts = [
        f'<line class="m-grid" x1="{left}" x2="{right}" y1="{y_of(f):.1f}" y2="{y_of(f):.1f}"/>'
        for f in (0.0, 0.5, 1.0)
    ]
    edges = [1.0, *(edge for edge, _ in _BAND_EDGES), 200.0]
    bands = [*(band for _, band in _BAND_EDGES), RiskLevel.SEVERE]
    for edge in edges[1:-1]:
        x = x_of(edge)
        # Broken around the tick row, so a rule never runs through its own number.
        parts.append(
            f'<line class="m-edge" x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{bottom + 2}"/>'
            f'<line class="m-edge" x1="{x:.1f}" x2="{x:.1f}" y1="{strip_top}" '
            f'y2="{strip_top + _STRIP_HEIGHT}"/>'
        )
        parts.append(
            f'<text class="t-axis" x="{x:.1f}" y="{bottom + 16}" text-anchor="middle">{edge:g}</text>'
        )
    parts.append(
        f'<text class="t-axis" x="{right}" y="{bottom + 16}" text-anchor="end">years</text>'
    )
    for start, end, band in zip(edges, edges[1:], bands):
        x0, x1 = x_of(start), x_of(end)
        active = band is level
        if active:
            parts.append(
                f'<rect class="m-band-on" x="{x0:.1f}" y="{strip_top}" '
                f'width="{x1 - x0:.1f}" height="{_STRIP_HEIGHT}"/>'
            )
        parts.append(
            f'<text class="{"t-band-on" if active else "t-band"}" x="{(x0 + x1) / 2:.1f}" '
            f'y="{strip_top + 16}" text-anchor="middle">{escape(band.value.capitalize())}</text>'
        )
    curve = " ".join(
        f"{x_of(t):.1f},{y_of(shape(t)):.1f}"
        for t in (1.25, 1.6, 2, 3, 5, 8, 12, 20, 35, 50, 80, 120, 170)
    )
    parts.append(f'<polyline class="m-curve" points="{curve}"/>')
    parts.append(
        f'<text class="t-axis" x="{right}" y="{y_of(0.36):.1f}" text-anchor="end">Return levels</text>'
    )
    parts.append(
        f'<line class="m-drop" x1="{crossing:.1f}" x2="{crossing:.1f}" '
        f'y1="{peak_y:.1f}" y2="{strip_top}"/>'
    )
    parts.append(
        f'<line class="m-peak" x1="{left}" x2="{crossing:.1f}" y1="{peak_y:.1f}" y2="{peak_y:.1f}"/>'
    )
    parts.append(f'<text class="t-peak" x="{left + 4}" y="{peak_y - 6:.1f}">Forecast peak</text>')
    parts.append(
        station_circle(
            level, attrs=f'x="{crossing - 8:.1f}" y="{peak_y - 8:.1f}" width="16" height="16"'
        )
    )
    return (
        f'<svg class="cra-plot-chart {frame.css_class}" '
        f'viewBox="0 0 {frame.width} {frame.height}" font-size="{frame.font_size}" '
        f'role="img" aria-label="{escape(label, quote=True)}">{"".join(parts)}</svg>'
    )


def specimen_plot() -> tuple[RiskLevel, str]:
    """The first screen's schematic of how a band is read, with the band it reads.

    Shape only, so nothing here can pass for data: no level values are printed, and
    the curve and the peak are drawn rather than measured (the card says so). What is
    real is the reading itself, a log return-period axis split at the band edges in
    agent/risk_bands.py, so the band named is the band that geometry would get.
    """
    peak = 0.62  # height of the drawn peak, as a fraction of the plot
    period = math.exp(((peak - 0.12) / 0.78) ** 1.25 * math.log(200.0))
    level = next((band for edge, band in _BAND_EDGES if period < edge), RiskLevel.SEVERE)
    label = (
        "Schematic, not data: a forecast peak line meets a place's return-level curve "
        f"and drops into the {level.value.capitalize()} band."
    )
    full = _Frame("cra-plot-full", width=480, height=208, left=12, right=12, top=12)
    compact = _Frame("cra-plot-compact", width=320, height=196, left=8, right=8, top=12)
    return level, (
        f'<figure class="cra-plot cra-lvl-{level.value}">'
        f"{_specimen_svg(full, peak, level, label)}{_specimen_svg(compact, peak, level, label)}"
        "</figure>"
    )
