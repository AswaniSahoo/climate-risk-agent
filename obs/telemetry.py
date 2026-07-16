"""Structural observability for every model call: latency, tokens, retries, cost.

Design mirrors the citation validator's philosophy: the property is enforced at
a chokepoint, not by discipline. All Gemini traffic already flows through
`rag/gemini_client.py`, so instrumenting that seam means NO call can escape
being measured — behavior, latency, and reliability become queryable data.

- Records go to an in-memory ring (thread-safe: embeds run in a pool) AND to a
  daily JSONL file (`TELEMETRY_DIR`, default data/telemetry) for offline
  aggregation across sessions.
- `Span` scopes records to one logical operation (one RiskReport) and rolls up
  calls / tokens / retries / cache hits / wall time.
- Costs are ESTIMATES from an env-overridable price table — token counts are
  measured truth; dollar figures are labeled estimates, never authoritative.
"""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.RLock()
_EVENTS: list[dict] = []

# USD per 1M tokens — ESTIMATES (verify against the current Google price sheet;
# override without a code change via env). Cost output is always labeled "est".
_PRICE_PER_MTOK = {
    "gemini-2.5-flash": (
        float(os.environ.get("PRICE_FLASH_IN", "0.30")),
        float(os.environ.get("PRICE_FLASH_OUT", "2.50")),
    ),
    "gemini-embedding-2": (
        float(os.environ.get("PRICE_EMBED_IN", "0.15")),
        0.0,
    ),
}


def _sink_path() -> Path:
    directory = Path(os.environ.get("TELEMETRY_DIR", "data/telemetry"))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"


def record(
    *,
    op: str,
    model: str,
    latency_ms: float,
    tokens_in: int,
    tokens_out: int,
    retries: int,
    ok: bool,
    cached: bool = False,
) -> None:
    """Record one model call (or cache hit). Never raises — observability must
    not be able to take down the observed system."""
    event = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "op": op,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "retries": retries,
        "ok": ok,
        "cached": cached,
    }
    with _LOCK:
        _EVENTS.append(event)
    try:
        with _sink_path().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
    except OSError as exc:
        print(f"[telemetry] disk sink unavailable ({exc}) — keeping in-memory only")


def snapshot() -> list[dict]:
    """A copy of every event recorded by this process."""
    with _LOCK:
        return list(_EVENTS)


def reset() -> None:
    """Clear in-memory events (tests; the JSONL sink is append-only history)."""
    with _LOCK:
        _EVENTS.clear()


def estimate_cost_usd(events: list[dict]) -> float:
    """Estimated spend for `events` — cache hits are free by definition."""
    total = 0.0
    for e in events:
        if e.get("cached"):
            continue
        price_in, price_out = _PRICE_PER_MTOK.get(e["model"], (0.0, 0.0))
        total += e["tokens_in"] / 1e6 * price_in + e["tokens_out"] / 1e6 * price_out
    return round(total, 6)


class Span:
    """Scope telemetry to one logical operation (e.g. one RiskReport)."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[dict] = []
        self._start_index = 0
        self._t0 = 0.0
        self.wall_ms = 0.0

    def __enter__(self) -> "Span":
        with _LOCK:
            self._start_index = len(_EVENTS)
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc_info) -> None:
        self.wall_ms = (time.perf_counter() - self._t0) * 1000.0
        with _LOCK:
            self.events = list(_EVENTS[self._start_index :])

    def summary(self) -> dict:
        live = [e for e in self.events if not e.get("cached")]
        return {
            "span": self.name,
            "wall_ms": round(self.wall_ms, 1),
            "calls": len(live),
            "cache_hits": sum(1 for e in self.events if e.get("cached")),
            "retries": sum(e["retries"] for e in self.events),
            "failures": sum(1 for e in self.events if not e["ok"]),
            "tokens_in": sum(e["tokens_in"] for e in live),
            "tokens_out": sum(e["tokens_out"] for e in live),
            "est_cost_usd": estimate_cost_usd(self.events),
        }


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(q * len(sorted_values)), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(events: list[dict]) -> dict:
    """Per-op rollup: calls, failures, retries, latency percentiles, est cost."""
    by_op: dict[str, list[dict]] = {}
    for e in events:
        by_op.setdefault(e["op"], []).append(e)
    stats: dict[str, dict] = {}
    for op, group in by_op.items():
        latencies = sorted(e["latency_ms"] for e in group if not e.get("cached"))
        stats[op] = {
            "calls": len(group),
            "cache_hits": sum(1 for e in group if e.get("cached")),
            "failures": sum(1 for e in group if not e["ok"]),
            "retries": sum(e["retries"] for e in group),
            "p50_ms": _percentile(latencies, 0.50),
            "p95_ms": _percentile(latencies, 0.95),
            "est_cost_usd": estimate_cost_usd(group),
        }
    return stats
