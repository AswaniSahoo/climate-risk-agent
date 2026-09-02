"""Cache v2: one pluggable key/value tier, shared across replicas when it can be.

WHY THIS EXISTS. Every cache in this repo before v2 was a directory of files on
the local disk. On Cloud Run that disk is per-replica and ephemeral: with up to
2 replicas and a redeploy on every push, a warm entry helps exactly one
container until the next deploy throws it away. The costs being hidden are
MEASURED: ~42 s for a cold ERA5 archive fetch (tools/climatology.py) and 12.9 s
for the GEV fit + trend test + bootstrap at n_boot=150 (tools/hazard_stats.py,
tools/gev_trend.py). Paying those again per replica per deploy is the whole
problem, so the cache needed a tier that OUTLIVES the container.

WHY UPSTASH REDIS OVER REST, and not a redis client: Upstash's free tier speaks
HTTPS, so `httpx` (already a dependency) is the whole client — no new runtime
dependency, no connection pool to manage across serverless cold starts, and
tests mock it with pytest-httpx like every other HTTP seam here.

WHY A PROTOCOL WITH A DISK FALLBACK: keyless local dev, CI and the test suite
must keep working exactly as before. No credentials -> disk, one INFO line.
Credentials present but the service is unreachable -> ONE warning per process,
then disk for the rest of it. A cache is an optimisation; it is never allowed
to turn a working request into a failing one.

WHY IT LIVES IN tools/: it has no repo-internal imports at module level (only
httpx + pydantic), so both `rag/` and `tools/` can depend on it without a
cycle, and `tools/validation.py` already sets the precedent that tools/ carries
shared infrastructure alongside the agent's data tools. A new top-level package
would also have to be added to the mypy `files` list for no gain.

UPSTASH REST CONTRACT — verified 2026-09-02 against
https://upstash.com/docs/redis/features/restapi (quoted, not remembered):
- Auth: `Authorization: Bearer $TOKEN` header ("You need to add a header to
  your API requests as `Authorization: Bearer $TOKEN`").
- Command form used here — "POST Command in Body": "you can send the whole
  command in the request body as a single JSON array. Array's first element
  must be the command name", e.g. `curl -X POST -d '["SET", "foo", "bar",
  "EX", 100]' https://...upstash.io`. Chosen over the path form
  (`REST_URL/set/foo/bar/EX/100`) because our values are multi-KB JSON
  documents that have no business being URL-escaped into a path.
- Success response: "a single `result` field" whose value is `null`, an
  integer, a string, or an array — `{"result": "OK"}`, `{"result": null}`.
- Failure response: "a single `error` field with a string value", e.g.
  `{"error":"WRONGPASS invalid password"}`.
- HTTP codes: `200 OK` on success; `400 Bad Request` for a syntax error,
  invalid/unsupported command, or failed execution; `401 Unauthorized` when the
  token is missing or invalid; `405 Method Not Allowed` for anything but
  HEAD/GET/POST/PUT.
- `/pipeline` (POST, two-dimensional JSON array, response `[{"result":...},
  {"error":...}]`, explicitly "not atomic") exists and would batch a prewarm
  run; unused for now because every call site here reads or writes exactly one
  key, and a batch API with one item is just a slower single call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Mapping
from functools import lru_cache
from pathlib import Path
from typing import Protocol, TypeVar, runtime_checkable

import httpx
from pydantic import BaseModel, ValidationError

_log = logging.getLogger(__name__)

# Bumping this invalidates every namespace at once — the escape hatch for a
# change in what we store that no per-entry fingerprint would catch.
CACHE_SCHEMA_VERSION = "v1"
_KEY_PREFIX = "crg"

_DEFAULT_ROOT = "data/cache/kv"

_now = time.time  # module attr so tests can freeze the clock

M = TypeVar("M", bound=BaseModel)


class CacheUnavailable(RuntimeError):
    """A backend could not be reached, or answered with an error.

    Raised only by backends; callers see it turned into a plain cache miss.
    """


class CacheBackend(Protocol):
    """The whole contract: a string in, a string out, and a name to report."""

    name: str

    def get(self, key: str) -> str | None: ...

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None: ...


@runtime_checkable
class _TracingBackend(Protocol):
    """Optional extra: a backend that can say WHICH tier served a read."""

    def get_traced(self, key: str) -> tuple[str | None, str]: ...


def get_traced(backend: CacheBackend, key: str) -> tuple[str | None, str]:
    """Read through `backend`, reporting the tier that actually served the value."""
    if isinstance(backend, _TracingBackend):
        return backend.get_traced(key)
    return backend.get(key), backend.name


class NullCache:
    """Stores nothing. What `cache_from_env()` hands back under pytest, so no
    test can read or write the shared cache by accident (the same hermetic
    guard tools/climatology.py has carried since the disk cache landed)."""

    def __init__(self) -> None:
        self.name = "null"

    def get(self, key: str) -> str | None:
        return None

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        return None


class DiskCache:
    """One JSON file per key: `{"value": ..., "expires_at": epoch|null}`.

    The filename is sha256(key), which is both a stable identity for any key
    shape and inherently traversal-safe — a namespaced key carries `:`, which
    is not a legal Windows filename character, so hashing is not optional here.
    """

    def __init__(self, root: Path | str) -> None:
        self.name = "disk"
        self.root = Path(root)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # read-only FS: degrade to a cache that misses
            _log.warning("cache directory %s unusable (%s)", self.root, exc)

    def _path(self, key: str) -> Path:
        return self.root / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"

    def get(self, key: str) -> str | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            value = payload["value"]
            expires_at = payload["expires_at"]
            if not isinstance(value, str):
                raise ValueError("cached value is not a string")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            # Corrupt entry (torn write, schema drift) -> a LOUD miss, never a crash.
            _log.warning("corrupt cache entry %s ignored (%s)", path.name, exc)
            return None
        if expires_at is not None and _now() >= expires_at:
            try:
                path.unlink(missing_ok=True)  # expired: reclaim the space now
            except OSError:
                pass
            return None
        return value

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        payload = {"value": value, "expires_at": None if ttl_s is None else _now() + ttl_s}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            self._path(key).write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            _log.warning("cache write failed (%s) — continuing uncached", exc)


class UpstashRedisCache:
    """GET/SET (with EX) against the Upstash REST API — see the contract at top.

    Every httpx/JSON/protocol failure is caught here and re-raised as the single
    `CacheUnavailable` type, so the layer above has exactly one thing to handle.
    """

    def __init__(self, url: str, token: str, timeout_s: float = 3.0) -> None:
        self.name = "redis"
        self._url = url.rstrip("/")
        self._token = token
        self._timeout_s = timeout_s

    def _command(self, command: list[str | int]) -> object:
        try:
            response = httpx.post(
                self._url,
                json=command,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise CacheUnavailable(f"Upstash request failed: {exc}") from exc
        if response.status_code != 200:
            # 400 syntax/exec error, 401 bad token, 405 wrong method — all "no cache".
            raise CacheUnavailable(f"Upstash returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise CacheUnavailable(f"Upstash returned a non-JSON body: {exc}") from exc
        if not isinstance(payload, dict):
            raise CacheUnavailable("Upstash returned an unexpected body shape")
        if "error" in payload:
            raise CacheUnavailable(f"Upstash error: {payload['error']}")
        return payload.get("result")

    def get(self, key: str) -> str | None:
        result = self._command(["GET", key])
        return result if isinstance(result, str) else None

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        command: list[str | int] = ["SET", key, value]
        if ttl_s is not None:
            command += ["EX", int(ttl_s)]
        self._command(command)


class FallbackCache:
    """Primary tier with a standby: reads prefer the primary, writes go to both.

    Write-through matters as much as read-through — after a Redis outage the
    local disk is still warm, so the degraded process is no slower than the
    pre-v2 one. The outage warning fires ONCE per instance (and
    `cache_from_env()` is memoised, so once per process): a per-call warning on
    a dead cache is just a second outage in the log.
    """

    def __init__(self, primary: CacheBackend, fallback: CacheBackend) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+{fallback.name}"
        self._warned = False

    def _degrade(self, op: str, exc: Exception) -> None:
        if self._warned:
            return
        self._warned = True
        _log.warning(
            "cache backend %r failed on %s (%s) — falling back to %s for this process",
            self.primary.name, op, exc, self.fallback.name,
        )

    def get_traced(self, key: str) -> tuple[str | None, str]:
        try:
            value = self.primary.get(key)
        except CacheUnavailable as exc:
            self._degrade("get", exc)
        else:
            if value is not None:
                return value, self.primary.name
        try:
            return self.fallback.get(key), self.fallback.name
        except CacheUnavailable as exc:
            self._degrade("get", exc)
            return None, self.fallback.name

    def get(self, key: str) -> str | None:
        return self.get_traced(key)[0]

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        for backend in (self.primary, self.fallback):
            try:
                backend.set(key, value, ttl_s)
            except CacheUnavailable as exc:
                self._degrade("set", exc)


def _backend_from_env(env: Mapping[str, str]) -> CacheBackend:
    """Pick a backend from configuration. Pure in `env` so it is directly testable."""
    root = Path(env.get("CRG_CACHE_DIR", _DEFAULT_ROOT))
    disk = DiskCache(root)
    url, token = env.get("UPSTASH_REDIS_REST_URL"), env.get("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        # Deliberately no URL/token in the log line: the token is a secret and
        # the URL identifies the database.
        _log.info("cache backend: Upstash Redis (REST) with disk fallback at %s", root)
        return FallbackCache(UpstashRedisCache(url, token), disk)
    _log.info(
        "cache backend: disk at %s (set UPSTASH_REDIS_REST_URL + "
        "UPSTASH_REDIS_REST_TOKEN to share one cache across replicas)",
        root,
    )
    return disk


@lru_cache(maxsize=1)
def cache_from_env() -> CacheBackend:
    """The process-wide backend (memoised: one instance, one outage warning)."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return NullCache()  # hermetic tests: the shared cache is never touched
    return _backend_from_env(os.environ)


def _record(namespace: str, backend_name: str, hit: bool, latency_ms: float) -> None:
    """Emit one cache event in obs/telemetry.py's existing record shape.

    `op` names the namespace and `model` names the tier that served it, so a
    future UI panel reads "hazard fit: hit (redis)" straight off the rollup,
    and `summarize()` counts hits with no change to its logic. Both token
    counts are 0 because a cache read bills nothing.
    """
    try:
        from obs.telemetry import record

        record(op=f"cache:{namespace}", model=backend_name, latency_ms=latency_ms,
               tokens_in=0, tokens_out=0, retries=0, ok=True, cached=hit)
    except Exception as exc:  # noqa: BLE001 — observability never breaks the observed
        _log.debug("cache telemetry unavailable (%s)", exc)


class JsonCache:
    """Typed façade: Pydantic models in and out, namespaced keys, no exceptions.

    Keys are `crg:<schema version>:<namespace>:<key>` so one Redis database can
    hold every namespace, and a schema bump retires all of them at once.
    """

    def __init__(self, namespace: str, backend: CacheBackend | None = None) -> None:
        self.namespace = namespace
        self._backend = backend

    @property
    def backend(self) -> CacheBackend:
        # Resolved late: import order must not pin the environment.
        return self._backend if self._backend is not None else cache_from_env()

    def key(self, key: str) -> str:
        return f"{_KEY_PREFIX}:{CACHE_SCHEMA_VERSION}:{self.namespace}:{key}"

    def get_model(self, key: str, model: type[M]) -> M | None:
        started = time.perf_counter()
        try:
            raw, served_by = get_traced(self.backend, self.key(key))
        except Exception as exc:  # noqa: BLE001 — a broken cache is a miss, not an error
            _log.warning("cache read failed for %s (%s) — treating as a miss", self.namespace, exc)
            return None
        parsed: M | None = None
        if raw is not None:
            try:
                parsed = model.model_validate_json(raw)
            except ValidationError as exc:
                _log.warning("stale cache entry for %s ignored (%s)", self.namespace, exc)
        _record(self.namespace, served_by, parsed is not None,
                (time.perf_counter() - started) * 1000.0)
        return parsed

    def set_model(self, key: str, value: BaseModel, ttl_s: int | None = None) -> None:
        try:
            self.backend.set(self.key(key), value.model_dump_json(), ttl_s)
        except Exception as exc:  # noqa: BLE001 — a failed write is not a failed request
            _log.warning("cache write failed for %s (%s) — continuing", self.namespace, exc)
