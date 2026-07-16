"""Logging setup: library code logs, entrypoints configure.

Library modules use `logging.getLogger(__name__)` and NEVER call basicConfig —
the entrypoint that owns the process (UI, API, eval runner, script) calls
`configure()` once. Unconfigured, Python's last-resort handler still prints
WARNING+ to stderr, so the loud-fallback guarantee survives even in bare
imports (a degraded layer can never be silent).
"""
from __future__ import annotations

import logging


def configure(level: int = logging.INFO) -> None:
    """Process-level logging config — call once from the entrypoint."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
