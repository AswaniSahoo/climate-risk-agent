"""Shared test fixtures.

Telemetry isolation: the Gemini seam records every call (that's the point),
so any test that touches it would otherwise append to the real
data/telemetry/*.jsonl. Route all test telemetry to a tmp dir and reset the
in-memory ring per test — tests must never pollute production observability data.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_telemetry(tmp_path, monkeypatch):
    from obs import telemetry

    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "telemetry"))
    telemetry.reset()
    yield
    telemetry.reset()
