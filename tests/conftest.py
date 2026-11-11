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

import socket

import pytest

_LOOPBACK = {"", "localhost", "127.0.0.1", "::1", "0.0.0.0", "::"}


def _is_loopback(host) -> bool:
    name = host.decode() if isinstance(host, bytes) else host
    return name is None or str(name) in _LOOPBACK or str(name).startswith("127.")


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """No test may touch the network: a lookup or connect beyond loopback fails it.

    The call itself raises, as an offline machine would, and the attempt is
    recorded so the test then FAILS at teardown even if the code under test
    swallowed the error and degraded quietly, which is how two NL tests sent a
    real Open-Meteo Archive request on every run. HTTP seams are mocked with
    pytest-httpx, which never opens a socket. Yields the recorded attempts.
    """
    attempts: list[str] = []
    real_getaddrinfo = socket.getaddrinfo
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex

    def getaddrinfo(host, *args, **kwargs):
        if _is_loopback(host):
            return real_getaddrinfo(host, *args, **kwargs)
        attempts.append(str(host))
        raise socket.gaierror(socket.EAI_NONAME, f"network access in a test: {host}")

    def _refuse_remote(addr) -> None:
        if isinstance(addr, (str, bytes)) or _is_loopback(addr[0]):  # unix socket or loopback
            return
        attempts.append(str(addr[0]))
        raise OSError(f"network access in a test: {addr[0]}")

    def connect(self, addr):
        _refuse_remote(addr)
        return real_connect(self, addr)

    def connect_ex(self, addr):
        _refuse_remote(addr)
        return real_connect_ex(self, addr)

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    yield attempts
    if attempts:
        pytest.fail(f"this test tried to reach the network: {sorted(set(attempts))}",
                    pytrace=False)


@pytest.fixture(autouse=True)
def _archive_budget_fresh(monkeypatch):
    """Each test starts with an empty Open-Meteo Archive budget and no breaker,
    at the default limits, whatever the shell exports."""
    from tools import archive_budget
    from tools.cache_backend import LocalCounter

    for name in ("CRG_ARCHIVE_DAILY_CALLS", "CRG_ARCHIVE_HOURLY_CALLS"):
        monkeypatch.delenv(name, raising=False)
    archive_budget.reset(LocalCounter())
    yield
    archive_budget.reset(LocalCounter())


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


@pytest.fixture(autouse=True)
def _fresh_spend_guard(monkeypatch):
    """Every test starts on an empty in-process spend counter, the default
    limits and a fresh embedding-probe result (rag/spend_guard.py).

    A developer's shell may export a limit, and one test's spending must not
    refuse the next test's call. The shared Upstash counter is never used under
    pytest; tests of it hand in their own.
    """
    import rag.gemini_client as gc
    from rag import spend_guard

    for name in ("CRG_DAILY_MODEL_BUDGET_USD", "CRG_SESSION_FRESH_REPORTS",
                 "CRG_API_RATE_LIMIT", "CRG_API_RATE_WINDOW_S"):
        monkeypatch.delenv(name, raising=False)
    spend_guard.reset()
    gc._reset_probe()
    yield
    spend_guard.reset()
    gc._reset_probe()
