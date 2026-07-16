"""Offline telemetry report: aggregate every recorded model call across sessions.

Reads the daily JSONL files the seam writes (TELEMETRY_DIR, default
data/telemetry) and prints per-day, per-op rollups: calls, cache hits,
failures, retries, latency percentiles, estimated cost.

Run:  uv run python -m obs.report
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from obs.telemetry import summarize


def main() -> None:
    directory = Path(os.environ.get("TELEMETRY_DIR", "data/telemetry"))
    files = sorted(directory.glob("*.jsonl"))
    if not files:
        print(f"no telemetry yet in {directory}/ — run the agent or an eval first")
        return

    grand: list[dict] = []
    for path in files:
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # torn write on a crashed run — skip, don't die
        grand.extend(events)

        print(f"\n== {path.stem} ({len(events)} events) ==")
        for op, s in summarize(events).items():
            print(
                f"  {op:9s} calls={s['calls']:5d}  cache_hits={s['cache_hits']:4d}  "
                f"failures={s['failures']:3d}  retries={s['retries']:3d}  "
                f"p50={s['p50_ms']:7.1f}ms  p95={s['p95_ms']:8.1f}ms  "
                f"est=${s['est_cost_usd']:.4f}"
            )

    print(f"\n== TOTAL ({len(grand)} events across {len(files)} day(s)) ==")
    for op, s in summarize(grand).items():
        print(
            f"  {op:9s} calls={s['calls']:5d}  cache_hits={s['cache_hits']:4d}  "
            f"failures={s['failures']:3d}  retries={s['retries']:3d}  "
            f"p50={s['p50_ms']:7.1f}ms  p95={s['p95_ms']:8.1f}ms  "
            f"est=${s['est_cost_usd']:.4f}"
        )
    print("\n(costs are estimates: measured tokens x configured per-Mtok prices)")


if __name__ == "__main__":
    main()
