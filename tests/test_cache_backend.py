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
import itertools
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


# --- DiskCache.set is ATOMIC -----------------------------------------------

def test_a_failed_disk_write_leaves_no_partial_file(tmp_path, monkeypatch, caplog):
    """The old `write_text` could leave a truncated JSON file behind, which the
    next reader logs as a corrupt entry. A temp file plus os.replace means a
    failed write leaves the directory exactly as it was."""
    import tools.cache_backend as backend_mod

    cache = DiskCache(tmp_path)
    real_replace = backend_mod.os.replace

    def exploding_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(backend_mod.os, "replace", exploding_replace)
    with caplog.at_level(logging.WARNING):
        cache.set("k", "v")  # never raises: a cache write is not a request
    monkeypatch.setattr(backend_mod.os, "replace", real_replace)

    assert list(tmp_path.iterdir()) == []  # no entry, and no leftover .tmp
    assert cache.get("k") is None
    assert "cache write failed" in caplog.text


def test_concurrent_writes_from_two_threads_both_land(tmp_path):
    import threading

    cache = DiskCache(tmp_path)
    barrier = threading.Barrier(2)

    def write(key: str, value: str) -> None:
        barrier.wait()
        for _ in range(50):
            cache.set(key, value)

    threads = [
        threading.Thread(target=write, args=("alpha", "a" * 2000)),
        threading.Thread(target=write, args=("beta", "b" * 2000)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert cache.get("alpha") == "a" * 2000
    assert cache.get("beta") == "b" * 2000
    assert len(list(tmp_path.glob("*.json"))) == 2  # one file per key...
    assert list(tmp_path.glob("*.tmp")) == []       # ...and nothing half-written


def test_same_key_written_concurrently_is_never_torn(tmp_path):
    """os.replace publishes a whole file at once, so the entry that survives two
    threads racing on one key is one of the two values — never a splice of both,
    which is what interleaved write_text calls produce."""
    import threading

    cache = DiskCache(tmp_path)
    values = ["a" * 5000, "b" * 5000]
    barrier = threading.Barrier(len(values))

    def hammer(value: str) -> None:
        barrier.wait()
        for _ in range(50):
            cache.set("shared", value)

    threads = [threading.Thread(target=hammer, args=(v,)) for v in values]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    [entry] = list(tmp_path.glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))  # parses => not spliced
    assert payload["value"] in values
    assert list(tmp_path.glob("*.tmp")) == []


# --- cache-read telemetry: complete in memory, SAMPLED on disk --------------

def test_cache_hits_are_sampled_on_disk_but_never_in_memory(tmp_path, monkeypatch):
    """Every cache GET used to append a line to the daily JSONL. Misses still do
    (a cost or cold-start investigation is about misses); hits go 1-in-N, and
    all of them stay in the in-memory ring the UI badge reads."""
    import tools.cache_backend as backend_mod
    from obs.telemetry import _sink_path

    monkeypatch.setattr(backend_mod, "_HIT_SAMPLE_N", 5)
    monkeypatch.setattr(backend_mod, "_hit_counter", itertools.count())
    cache = JsonCache("demo", backend=_StubBackend("disk"))
    cache.set_model("k", _Thing(name="a", value=1), ttl_s=60)

    for _ in range(10):
        assert cache.get_model("k", _Thing) is not None

    events = [e for e in snapshot() if e["op"] == "cache:demo"]
    assert len(events) == 10 and all(e["cached"] for e in events)  # memory: complete

    lines = _sink_path().read_text(encoding="utf-8").splitlines()
    persisted = [json.loads(line) for line in lines if '"cache:demo"' in line]
    assert len(persisted) == 2  # disk: 1-in-5


def test_cache_misses_always_reach_the_disk_sink(tmp_path, monkeypatch):
    import tools.cache_backend as backend_mod
    from obs.telemetry import _sink_path

    monkeypatch.setattr(backend_mod, "_HIT_SAMPLE_N", 1000)
    cache = JsonCache("demo", backend=_StubBackend("disk"))

    for _ in range(3):
        assert cache.get_model("absent", _Thing) is None

    lines = _sink_path().read_text(encoding="utf-8").splitlines()
    persisted = [json.loads(line) for line in lines if '"cache:demo"' in line]
    assert len(persisted) == 3 and not any(e["cached"] for e in persisted)

