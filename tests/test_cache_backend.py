"""Tests for the shared cache backend (tools/cache_backend.py).

Cache v2 exists because Cloud Run gives every replica its OWN ephemeral disk:
the pre-v2 caches were per-replica and died on redeploy, so the 42 s ERA5 fetch
+ 12.9 s GEV fit was paid again by the next container. The backend adds a
shared tier (Upstash Redis over REST) WITHOUT taking anything away — keyless
local dev, CI and these tests still run on disk.

Everything here is offline: the REST calls are mocked with pytest-httpx, the
disk tier uses tmp_path. The invariant under test throughout is that a cache
FAILURE is never a request failure.
"""
import json
import logging

import pytest
from pydantic import BaseModel

from obs.telemetry import snapshot
from tools.cache_backend import (
    CACHE_SCHEMA_VERSION,
    CacheUnavailable,
    DiskCache,
    FallbackCache,
    JsonCache,
    UpstashRedisCache,
    _backend_from_env,
    get_traced,
)

_URL = "https://fake-cat-12345.upstash.io"
_TOKEN = "not-a-real-token"


class _Thing(BaseModel):
    name: str
    value: int


class _StubBackend:
    """An in-memory backend whose failure mode is controllable."""

    def __init__(self, name: str, *, fail: bool = False, boom: bool = False) -> None:
        self.name = name
        self._fail = fail  # raises CacheUnavailable (the expected degradation)
        self._boom = boom  # raises something else entirely (the unexpected one)
        self.store: dict[str, str] = {}

    def _maybe_raise(self) -> None:
        if self._boom:
            raise RuntimeError("stub exploded in an unforeseen way")
        if self._fail:
            raise CacheUnavailable("stub backend down")

    def get(self, key: str) -> str | None:
        self._maybe_raise()
        return self.store.get(key)

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        self._maybe_raise()
        self.store[key] = value


# --- disk tier -------------------------------------------------------------

def test_disk_cache_roundtrips_a_value(tmp_path):
    cache = DiskCache(tmp_path)

    assert cache.get("k") is None  # cold
    cache.set("k", "v", None)
    assert cache.get("k") == "v"
    assert cache.name == "disk"


def test_disk_cache_honours_ttl_on_read(tmp_path, monkeypatch):
    import tools.cache_backend as backend_mod

    clock = [1000.0]
    monkeypatch.setattr(backend_mod, "_now", lambda: clock[0])
    cache = DiskCache(tmp_path)
    cache.set("k", "v", 60)

    clock[0] = 1059.0
    assert cache.get("k") == "v"  # still inside the window
    clock[0] = 1061.0
    assert cache.get("k") is None  # expired -> a miss, not a stale answer


def test_disk_cache_corrupt_file_is_a_loud_miss(tmp_path, caplog):
    cache = DiskCache(tmp_path)
    cache.set("k", "v", None)
    [path] = list(tmp_path.glob("*.json"))
    path.write_text("{not json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert cache.get("k") is None  # degrade to a miss, never crash
    assert "corrupt cache entry" in caplog.text


def test_disk_cache_never_writes_outside_its_root(tmp_path):
    # Keys are namespaced and caller-supplied; hashing them is what keeps a
    # traversal attempt inside the cache directory.
    cache = DiskCache(tmp_path)
    cache.set("../../escape", "v", None)

    assert not (tmp_path.parent / "escape.json").exists()
    assert list(tmp_path.glob("*.json"))
    assert cache.get("../../escape") == "v"


# --- Upstash REST tier -----------------------------------------------------

def test_upstash_set_sends_the_documented_command_array(httpx_mock):
    httpx_mock.add_response(url=_URL, method="POST", json={"result": "OK"})

    UpstashRedisCache(_URL, _TOKEN).set("k", "v", 60)

    [request] = httpx_mock.get_requests()
    assert request.headers["Authorization"] == f"Bearer {_TOKEN}"
    assert json.loads(request.content) == ["SET", "k", "v", "EX", 60]


def test_upstash_get_returns_the_result_field(httpx_mock):
    httpx_mock.add_response(url=_URL, method="POST", json={"result": "v"})

    assert UpstashRedisCache(_URL, _TOKEN).get("k") == "v"
    assert json.loads(httpx_mock.get_requests()[0].content) == ["GET", "k"]


def test_upstash_null_result_is_a_miss(httpx_mock):
    httpx_mock.add_response(url=_URL, method="POST", json={"result": None})

    assert UpstashRedisCache(_URL, _TOKEN).get("k") is None


def test_upstash_http_error_becomes_cache_unavailable(httpx_mock):
    httpx_mock.add_response(status_code=500)

    with pytest.raises(CacheUnavailable):
        UpstashRedisCache(_URL, _TOKEN).get("k")


def test_upstash_error_payload_becomes_cache_unavailable(httpx_mock):
    httpx_mock.add_response(json={"error": "WRONGPASS invalid password"})

    with pytest.raises(CacheUnavailable):
        UpstashRedisCache(_URL, _TOKEN).get("k")


def test_upstash_failure_never_names_the_token(httpx_mock):
    httpx_mock.add_response(status_code=401)

    with pytest.raises(CacheUnavailable) as exc:
        UpstashRedisCache(_URL, _TOKEN).get("k")
    assert _TOKEN not in str(exc.value)  # a cache error must not leak the secret


# --- fallback --------------------------------------------------------------

def test_fallback_serves_disk_when_redis_fails_and_warns_once(httpx_mock, tmp_path, caplog):
    httpx_mock.add_response(status_code=500, is_reusable=True)
    disk = DiskCache(tmp_path)
    disk.set("k", "v", None)
    cache = FallbackCache(UpstashRedisCache(_URL, _TOKEN), disk)

    with caplog.at_level(logging.WARNING):
        assert cache.get("k") == "v"
        assert cache.get("k") == "v"

    assert caplog.text.count("falling back") == 1  # ONE warning per process, not per call


def test_fallback_reports_which_backend_served(tmp_path):
    primary, fallback = _StubBackend("redis"), _StubBackend("disk")
    cache = FallbackCache(primary, fallback)

    cache.set("k", "v", None)
    assert get_traced(cache, "k") == ("v", "redis")

    primary.store.clear()  # redis evicted it; disk still has the write-through copy
    assert get_traced(cache, "k") == ("v", "disk")


def test_fallback_write_through_survives_a_dead_primary():
    primary, fallback = _StubBackend("redis", fail=True), _StubBackend("disk")
    cache = FallbackCache(primary, fallback)

    cache.set("k", "v", None)  # must not raise

    assert fallback.store["k"] == "v"


def test_get_traced_falls_back_to_the_backend_name():
    stub = _StubBackend("disk")
    stub.store["k"] = "v"

    assert get_traced(stub, "k") == ("v", "disk")


# --- env selection ---------------------------------------------------------

def test_env_without_credentials_selects_disk(tmp_path, caplog):
    with caplog.at_level(logging.INFO):
        backend = _backend_from_env({"CRG_CACHE_DIR": str(tmp_path)})

    assert isinstance(backend, DiskCache)
    assert "disk" in caplog.text


def test_env_with_credentials_selects_redis_with_disk_fallback(tmp_path, caplog):
    env = {
        "CRG_CACHE_DIR": str(tmp_path),
        "UPSTASH_REDIS_REST_URL": _URL,
        "UPSTASH_REDIS_REST_TOKEN": _TOKEN,
    }

    with caplog.at_level(logging.INFO):
        backend = _backend_from_env(env)

    assert isinstance(backend, FallbackCache)
    assert backend.name == "redis+disk"
    assert _TOKEN not in caplog.text  # never log the secret


def test_half_configured_credentials_fall_back_to_disk(tmp_path):
    backend = _backend_from_env({"CRG_CACHE_DIR": str(tmp_path), "UPSTASH_REDIS_REST_URL": _URL})

    assert isinstance(backend, DiskCache)


# --- JsonCache (the typed façade every caller uses) ------------------------

def test_json_cache_key_is_namespaced_and_versioned():
    cache = JsonCache("hazard_fit", backend=_StubBackend("disk"))

    assert cache.key("abc") == f"crg:{CACHE_SCHEMA_VERSION}:hazard_fit:abc"


def test_json_cache_roundtrips_a_pydantic_model():
    cache = JsonCache("demo", backend=_StubBackend("disk"))

    assert cache.get_model("k", _Thing) is None
    cache.set_model("k", _Thing(name="a", value=1), ttl_s=60)
    assert cache.get_model("k", _Thing) == _Thing(name="a", value=1)


def test_json_cache_records_hit_and_miss_telemetry():
    cache = JsonCache("hazard_fit", backend=_StubBackend("redis"))

    cache.get_model("k", _Thing)  # miss
    cache.set_model("k", _Thing(name="a", value=1), ttl_s=60)
    cache.get_model("k", _Thing)  # hit

    events = [e for e in snapshot() if e["op"] == "cache:hazard_fit"]
    assert [e["cached"] for e in events] == [False, True]
    assert {e["model"] for e in events} == {"redis"}  # a UI can render "hit (redis)"


def test_json_cache_swallows_an_unexpected_backend_failure():
    cache = JsonCache("demo", backend=_StubBackend("disk", boom=True))

    assert cache.get_model("k", _Thing) is None  # a broken cache is a miss...
    cache.set_model("k", _Thing(name="a", value=1), ttl_s=60)  # ...and a no-op write


def test_json_cache_treats_an_unparseable_payload_as_a_miss(caplog):
    stub = _StubBackend("disk")
    cache = JsonCache("demo", backend=stub)
    stub.store[cache.key("k")] = '{"name": "a"}'  # schema drifted: no `value`

    with caplog.at_level(logging.WARNING):
        assert cache.get_model("k", _Thing) is None
    assert "cache" in caplog.text.lower()
