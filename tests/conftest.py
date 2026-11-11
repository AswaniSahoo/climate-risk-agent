"""Shared test fixtures.

Telemetry isolation: the Gemini seam records every call (that's the point),
so any test that touches it would otherwise append to the real
data/telemetry/*.jsonl. Route all test telemetry to a tmp dir and reset the
in-memory ring per test — tests must never pollute production observability data.
"""
try:
    import pyarrow  # noqa: F401 (initialize Arrow C++ symbols before scipy/geos)
except ImportError:
    pass

import pytest


@pytest.fixture(autouse=True)
def _isolated_telemetry(tmp_path, monkeypatch):
    from obs import telemetry

    monkeypatch.setenv("TELEMETRY_DIR", str(tmp_path / "telemetry"))
    telemetry.reset()
    yield
    telemetry.reset()


@pytest.fixture(autouse=True)
def _jev_off_and_offline(monkeypatch):
    """No test reaches a Jev provider unless it opts in, and none reaches one for real.

    A developer's shell may export a key, a provider or the flag; every Jev
    variable is removed for every test, and the shared client (with its cache,
    circuit and pacing) is dropped on both sides, so one test's Jev state cannot
    leak into the next. Tests that exercise Jev set the flag and a fake key
    themselves and answer every request with pytest-httpx.
    """
    from rag import jev_client

    for name in ("TYPESAFE_API_KEY", "JEV_API_KEY", "JEV_BASE_URL", "CRG_JEV",
                 "CRG_JEV_MODEL", "CRG_JEV_MIN_INTERVAL_S"):
        monkeypatch.delenv(name, raising=False)
    jev_client.reset_client()
    yield
    jev_client.reset_client()
