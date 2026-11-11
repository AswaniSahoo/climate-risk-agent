"""Tests for the Cloud Storage tier and the counter (tools/cache_backend.py).

Everything runs against `FakeBucket`, an in-memory stand-in for the JSON API
served through pytest-httpx, with the semantics the code relies on, each
quoted from the docs in tools/cache_backend.py:
- every write makes a new generation;
- `ifGenerationMatch=0` creates only if no live object exists, and a
  generation that does not match fails with 412 Precondition Failed;
- a media GET returns the data, a plain GET the resource with `generation`
  and the custom `metadata`; a missing object is 404.
No test reaches Google: the token comes from a stub.
"""
import json
import logging
import threading
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pytest

from tools.cache_backend import (
    CacheUnavailable,
    DiskCache,
    FallbackCounter,
    GcsCache,
    GcsCounter,
    JsonCache,
    LocalCounter,
    TieredCache,
    _backend_from_env,
    _with_seed,
)

BUCKET = "climate-risk-agent-cache"
TOKEN = "not-a-real-token"


def _token():
    return {"Authorization": f"Bearer {TOKEN}"}


class FakeBucket:
    """One bucket's objects, with generations, preconditions and injectable faults."""

    def __init__(self) -> None:
        self.objects: dict[str, dict] = {}
        self.next_generation = 1_700_000_000_000_000
        self.lock = threading.Lock()
        self.fail_with: int | None = None  # answer every request with this status
        self.raise_error = False  # answer every request with a transport error
        self.steal_writes = 0  # this many uploads lose a race to a write that lands first
        self.requests: list[tuple[str, str]] = []

    def _name(self, url) -> str:
        return unquote(urlparse(str(url)).path.split("/o/", 1)[1])

    def _resource(self, name: str) -> dict:
        obj = self.objects[name]
        return {"name": name, "bucket": BUCKET, "generation": str(obj["generation"]),
                "metadata": dict(obj["metadata"]), "contentType": obj["contentType"]}

    def _store(self, name: str, data: bytes, metadata: dict, content_type: str) -> None:
        self.next_generation += 1
        self.objects[name] = {"data": data, "metadata": metadata, "contentType": content_type,
                              "generation": self.next_generation}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.raise_error:
            raise httpx.ConnectError("simulated outage", request=request)
        url = urlparse(str(request.url))
        query = {k: v[0] for k, v in parse_qs(url.query).items()}
        with self.lock:
            self.requests.append((request.method, url.path))
            if self.fail_with is not None:
                return httpx.Response(self.fail_with, json={"error": {"code": self.fail_with}})
            if request.method == "GET":
                name = self._name(request.url)
                if name not in self.objects:
                    return httpx.Response(404, json={"error": {"code": 404}})
                if query.get("alt") == "media":
                    return httpx.Response(200, content=self.objects[name]["data"])
                return httpx.Response(200, json=self._resource(name))
            if request.method == "DELETE":
                name = self._name(request.url)
                if self.objects.pop(name, None) is None:
                    return httpx.Response(404, json={"error": {"code": 404}})
                return httpx.Response(204)
            assert request.method == "POST" and "/upload/storage/v1/" in url.path
            assert query["uploadType"] == "multipart"
            meta, data, content_type = _parse_multipart(request)
            name = meta["name"]
            if self.steal_writes:  # someone else wrote first: our generation is stale now
                self.steal_writes -= 1
                if name in self.objects:
                    current = self.objects[name]
                    self._store(name, current["data"], current["metadata"], current["contentType"])
            if "ifGenerationMatch" in query:
                expected = int(query["ifGenerationMatch"])
                live = self.objects.get(name)
                if (expected == 0 and live is not None) or (
                        expected != 0 and (live is None or live["generation"] != expected)):
                    return httpx.Response(412, json={"error": {"code": 412}})
            self._store(name, data, dict(meta.get("metadata", {})), content_type)
            return httpx.Response(200, json=self._resource(name))


def _parse_multipart(request: httpx.Request) -> tuple[dict, bytes, str]:
    content_type = request.headers["Content-Type"]
    assert content_type.startswith("multipart/related; boundary=")
    boundary = content_type.split("boundary=", 1)[1].encode()
    parts = [p for p in request.content.split(b"--" + boundary) if p.strip() not in (b"", b"--")]
    assert len(parts) == 2, "metadata part, then data part"
    head0, body0 = parts[0].split(b"\r\n\r\n", 1)
    head1, body1 = parts[1].split(b"\r\n\r\n", 1)
    assert b"application/json" in head0
    part_type = head1.decode().split("Content-Type:", 1)[1].strip()
    return json.loads(body0), body1.rstrip(b"\r\n"), part_type


@pytest.fixture
def bucket(httpx_mock):
    fake = FakeBucket()
    httpx_mock.add_callback(fake, is_reusable=True, is_optional=True)
    return fake


class Clock:
    def __init__(self, now: float = 1_800_000_000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


# --- the key/value tier -----------------------------------------------------

def test_a_missing_key_is_a_miss(bucket):
    assert GcsCache(BUCKET, token=_token).get("crg:v1:demo:absent") is None


def test_a_value_round_trips_and_an_overwrite_wins(bucket):
    cache = GcsCache(BUCKET, token=_token)

    cache.set("crg:v1:demo:k", "first", ttl_s=60)
    cache.set("crg:v1:demo:k", "second", ttl_s=60)  # overwriting needs delete permission: granted

    assert cache.get("crg:v1:demo:k") == "second"


def test_the_expiry_is_enforced_when_read(bucket):
    clock = Clock()
    cache = GcsCache(BUCKET, token=_token, clock=clock)
    cache.set("crg:v1:demo:k", "v", ttl_s=60)

    clock.now += 59
    assert cache.get("crg:v1:demo:k") == "v"
    clock.now += 2
    assert cache.get("crg:v1:demo:k") is None  # expired, though the object is still there


def test_the_expiry_is_written_with_the_object_as_metadata(bucket):
    clock = Clock()
    GcsCache(BUCKET, token=_token, clock=clock).set("crg:v1:demo:k", "v", ttl_s=60)

    [obj] = bucket.objects.values()
    assert float(obj["metadata"]["expires_at"]) == clock.now + 60


def test_object_names_are_prefixed_and_percent_encoded(bucket):
    GcsCache(BUCKET, token=_token).get("crg:v1:hazard_fit:abc")

    method, path = bucket.requests[-1]
    assert path.endswith("/o/kv%2Fcrg%3Av1%3Ahazard_fit%3Aabc")


def test_a_failing_bucket_raises_cache_unavailable_and_never_names_the_token(bucket, caplog):
    bucket.fail_with = 503
    cache = GcsCache(BUCKET, token=_token)

    with caplog.at_level(logging.DEBUG), pytest.raises(CacheUnavailable) as raised:
        cache.get("crg:v1:demo:k")

    assert TOKEN not in str(raised.value) and TOKEN not in caplog.text


def test_credentials_that_fail_are_cache_unavailable(bucket):
    def broken():
        raise RuntimeError("no application default credentials")

    with pytest.raises(CacheUnavailable):
        GcsCache(BUCKET, token=broken).get("crg:v1:demo:k")


# --- the tier order: shipped seed, then disk, then the bucket ----------------

def test_a_bucket_hit_fills_the_disk_with_the_time_it_has_left(bucket, tmp_path, monkeypatch):
    import tools.cache_backend as backend_mod

    clock = Clock()
    monkeypatch.setattr(backend_mod, "_now", clock)  # one clock for disk, bucket and tiers
    far = GcsCache(BUCKET, token=_token)
    far.set("crg:v1:demo:k", "shared", ttl_s=3600)
    clock.now += 600
    disk = DiskCache(tmp_path)
    tiered = TieredCache(disk, far)

    assert tiered.get_traced("crg:v1:demo:k") == ("shared", "gcs")
    asked = len(bucket.requests)
    assert tiered.get_traced("crg:v1:demo:k") == ("shared", "disk")  # the disk answers now
    assert len(bucket.requests) == asked
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert payload["expires_at"] == pytest.approx(clock.now + 3000, abs=2)


def test_writes_go_through_to_disk_and_bucket(bucket, tmp_path):
    disk = DiskCache(tmp_path)
    tiered = TieredCache(disk, GcsCache(BUCKET, token=_token))

    tiered.set("crg:v1:demo:k", "v", ttl_s=60)

    assert disk.get("crg:v1:demo:k") == "v"
    assert GcsCache(BUCKET, token=_token).get("crg:v1:demo:k") == "v"


def test_a_bucket_outage_degrades_to_disk_logs_and_stays_away_for_a_while(bucket, tmp_path, caplog):
    clock = Clock()
    disk = DiskCache(tmp_path)
    tiered = TieredCache(disk, GcsCache(BUCKET, token=_token), retry_after_s=60, clock=clock)
    bucket.raise_error = True

    with caplog.at_level(logging.WARNING):
        tiered.set("crg:v1:demo:k", "v", ttl_s=60)  # never raises: a cache write is not a request
        assert tiered.get("crg:v1:demo:other") is None  # a miss, not an error
    assert disk.get("crg:v1:demo:k") == "v"
    assert "Cloud Storage" in caplog.text
    asked = len(bucket.requests)

    tiered.get("crg:v1:demo:other")
    assert len(bucket.requests) == asked  # left alone while the outage window lasts
    bucket.raise_error = False
    clock.now += 61
    tiered.get("crg:v1:demo:other")
    assert len(bucket.requests) == asked + 1  # and tried again after it


def test_the_environment_puts_the_seed_first_then_disk_then_the_bucket(tmp_path, monkeypatch):
    seed_dir = tmp_path / "seed"
    seed_dir.mkdir()
    key = JsonCache("demo").key("seeded")
    (seed_dir / "s.json").write_text(json.dumps({"entries": [{"key": key, "value": {"x": 1}}]}))
    env = {"CRG_CACHE_DIR": str(tmp_path / "kv"), "CRG_CACHE_GCS_BUCKET": BUCKET,
           "CRG_CACHE_SEED_DIR": str(seed_dir)}

    backend = _with_seed(_backend_from_env(env), env)

    assert backend.name == "precomputed+disk+gcs"
    assert backend.get(key) == '{"x":1}'  # answered by the seed: the bucket is never asked


def test_without_the_bucket_setting_the_disk_tier_stands_alone(tmp_path):
    backend = _backend_from_env({"CRG_CACHE_DIR": str(tmp_path)})

    assert isinstance(backend, DiskCache)


# --- the counter: INCRBY plus EXPIRE by compare-and-swap ---------------------

def test_the_first_add_creates_the_counter_and_later_adds_accumulate(bucket):
    counter = GcsCounter(BUCKET, token=_token)

    assert counter.add("crg:v1:budget:k", 5, ttl_s=60) == 5
    assert counter.add("crg:v1:budget:k", 7, ttl_s=60) == 12
    assert counter.add("crg:v1:budget:k", -2, ttl_s=60) == 10  # a refund, as INCRBY allows


def test_the_counter_restarts_after_its_lifetime_and_every_add_renews_it(bucket):
    clock = Clock()
    counter = GcsCounter(BUCKET, token=_token, clock=clock)

    counter.add("crg:v1:budget:k", 5, ttl_s=60)
    clock.now += 50
    assert counter.add("crg:v1:budget:k", 1, ttl_s=60) == 6  # renewed to now + 60
    clock.now += 50
    assert counter.add("crg:v1:budget:k", 0, ttl_s=60) == 6  # still alive: 100 s after the first add
    clock.now += 61
    assert counter.add("crg:v1:budget:k", 1, ttl_s=60) == 1  # expired: starts again


def test_a_lost_race_is_retried_with_jitter_and_nothing_is_lost(bucket):
    slept: list[float] = []
    counter = GcsCounter(BUCKET, token=_token, sleep=slept.append, jitter=lambda: 0.5)
    counter.add("crg:v1:budget:k", 1, ttl_s=60)
    bucket.steal_writes = 3  # the next three uploads each find a newer generation

    assert counter.add("crg:v1:budget:k", 1, ttl_s=60) == 2

    assert len(slept) == 3 and slept == sorted(slept)  # backing off, one wait per 412


def test_concurrent_adds_from_many_threads_lose_no_update(bucket):
    counter = GcsCounter(BUCKET, token=_token, sleep=lambda s: None, max_attempts=200)
    bucket.steal_writes = 25  # races on top of the ones the threads cause themselves

    def worker():
        for _ in range(20):
            counter.add("crg:v1:budget:k", 1, ttl_s=600)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert counter.add("crg:v1:budget:k", 0, ttl_s=600) == 8 * 20


def test_a_counter_that_keeps_losing_gives_up_as_cache_unavailable(bucket):
    counter = GcsCounter(BUCKET, token=_token, sleep=lambda s: None, max_attempts=4)
    counter.add("crg:v1:budget:k", 1, ttl_s=60)
    bucket.steal_writes = 100

    with pytest.raises(CacheUnavailable):
        counter.add("crg:v1:budget:k", 1, ttl_s=60)


def test_a_server_error_is_not_retried_as_a_race(bucket):
    counter = GcsCounter(BUCKET, token=_token, sleep=lambda s: pytest.fail("no backoff on 5xx"))
    bucket.fail_with = 500

    with pytest.raises(CacheUnavailable):
        counter.add("crg:v1:budget:k", 1, ttl_s=60)


def test_the_fallback_counter_counts_locally_while_the_bucket_is_down(bucket, caplog):
    clock = Clock()
    counter = FallbackCounter(GcsCounter(BUCKET, token=_token), LocalCounter(clock=clock),
                              retry_after_s=60, clock=clock)
    assert counter.add("crg:v1:budget:k", 3, ttl_s=600) == 3
    bucket.raise_error = True

    with caplog.at_level(logging.WARNING):
        value, served = counter.add_traced("crg:v1:budget:k", 4, ttl_s=600)

    assert (value, served.name) == (4, "local")  # this process's own count meanwhile
    assert "in this process" in caplog.text
    bucket.raise_error = False
    clock.now += 61
    assert counter.add("crg:v1:budget:k", 1, ttl_s=600) == 4  # the bucket again: 3 + 1


def test_the_local_counter_has_the_same_contract():
    clock = Clock()
    counter = LocalCounter(clock=clock)

    assert counter.add("k", 5, ttl_s=60) == 5
    clock.now += 30
    assert counter.add("k", 1, ttl_s=60) == 6
    clock.now += 61
    assert counter.add("k", 1, ttl_s=60) == 1


def test_a_counter_can_be_deleted(bucket):
    counter = GcsCounter(BUCKET, token=_token)
    counter.add("crg:v1:budget:k", 5, ttl_s=60)

    counter.delete("crg:v1:budget:k")

    assert counter.add("crg:v1:budget:k", 1, ttl_s=60) == 1
