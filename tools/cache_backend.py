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

COUNTERS (rag/spend_guard.py), verified 2026-11-11 against the same REST page
and the Upstash command pages for INCRBY and EXPIRE (quoted, not remembered):
- `/multi-exec`: "Upstash REST API supports transactions to execute multiple
  commands atomically", sent "as a two dimensional JSON array"; a successful
  transaction answers `[{"result":"RESPONSE_A"},{"result":"RESPONSE_B"}, ...]`
  and a discarded one "a single error ... `{ "error": "ERR ..." }`".
- INCRBY: "A missing key counts as `0`", and "the update is atomic, so it is
  safe for counters that several clients update at once".
- EXPIRE: "give a key a lifetime in seconds, after which the key is deleted
  automatically", and commands "that modify a value in place, such as
  [`INCR`] ... leave it untouched".
So one INCRBY + EXPIRE transaction is an atomic counter that cleans itself up.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import logging
import os
import random
import secrets
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable
from urllib.parse import quote

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
        """Write one entry ATOMICALLY: temp file in the same directory, then
        `os.replace`.

        A plain `write_text` is not atomic. A crash, a full disk or a container
        stopped mid-write leaves a truncated JSON file that `get` then reports
        as a corrupt entry, and two processes writing the same key can
        interleave into a file neither of them wrote. `os.replace` is atomic on
        POSIX and on Windows (same filesystem, which a sibling temp file
        guarantees), so a reader sees either the old entry or the new one and
        never half of either.
        """
        payload = {"value": value, "expires_at": None if ttl_s is None else _now() + ttl_s}
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(key)
            handle, tmp_name = tempfile.mkstemp(
                dir=self.root, prefix=f"{path.stem}.", suffix=".tmp"
            )
            tmp = Path(tmp_name)
            try:
                with os.fdopen(handle, "w", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload))
                os.replace(tmp, path)
            except BaseException:
                # Never leave the half-written temp behind for the next reader
                # (or for `*.json` globs) to trip over.
                tmp.unlink(missing_ok=True)
                raise
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

    def _transaction(self, commands: list[list[str | int]]) -> list[object]:
        """Run `commands` atomically (`/multi-exec`); one result per command."""
        try:
            response = httpx.post(
                f"{self._url}/multi-exec",
                json=commands,
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self._timeout_s,
            )
        except httpx.HTTPError as exc:
            raise CacheUnavailable(f"Upstash request failed: {exc}") from exc
        if response.status_code != 200:
            raise CacheUnavailable(f"Upstash returned HTTP {response.status_code}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise CacheUnavailable(f"Upstash returned a non-JSON body: {exc}") from exc
        if isinstance(payload, dict):  # the whole transaction was discarded
            raise CacheUnavailable(f"Upstash discarded the transaction: {payload.get('error')}")
        if not isinstance(payload, list) or len(payload) != len(commands):
            raise CacheUnavailable("Upstash returned an unexpected transaction body")
        results: list[object] = []
        for item in payload:
            if not isinstance(item, dict) or "error" in item:
                raise CacheUnavailable(f"Upstash error in a transaction: {item}")
            results.append(item.get("result"))
        return results

    def incrby_with_ttl(self, key: str, amount: int, ttl_s: int) -> int:
        """Add `amount` to the integer at `key` and (re)set its lifetime, as one
        transaction; returns the value after the increment."""
        value, _ = self._transaction(
            [["INCRBY", key, int(amount)], ["EXPIRE", key, int(ttl_s)]]
        )
        if not isinstance(value, int):
            raise CacheUnavailable(f"Upstash INCRBY returned {type(value).__name__}")
        return value


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


# --- Cloud Storage: the shared tier that outlives every instance --------------
#
# WHY CLOUD STORAGE, and WHY THE JSON API OVER httpx rather than the
# google-cloud-storage client: the service already ships google-auth (through
# google-genai, whose lock entry asks for google-auth[requests]) and httpx, so
# minting a token with google-auth and sending four JSON API requests with httpx
# adds no dependency to the image, and pytest-httpx mocks it like every other
# HTTP seam here. The client library would add google-cloud-storage and its
# google-api-core and google-resumable-media stack to cold start for exactly
# four calls.
#
# CLOUD STORAGE CONTRACT, JSON API, read 2026-11-11 at
# cloud.google.com/storage/docs (quoted, not remembered):
# - objects.get: "When `alt=media` is included as a query parameter in the
#   request, this method retrieves object data." Without it, the object
#   resource: `generation` is "The content generation of this object" and
#   `metadata` is "User-provided metadata, in key-value pairs."
# - objects.insert, POST https://storage.googleapis.com/upload/storage/v1/b/bucket/o
#   with uploadType=multipart: "Upload both the object data and its metadata, in
#   a single request." And: "If the object being inserted has the same name as
#   an existing object, the user must also have the `storage.objects.delete`
#   permission to overwrite the existing object."
# - request preconditions: with ifGenerationMatch, "If the values don't match,
#   the request fails with a `412 Precondition Failed` response", and with the
#   value 0 "the request only proceeds if no object with the specified name
#   exists in the bucket".
# - objects.delete: "If successful, this method returns an empty response body."
# - a request URL must percent-encode `/`, `:` and the other listed characters
#   in an object name.
# - scope devstorage.read_write: "Allows access to read and change data, but not
#   metadata like IAM policies."
# The runtime identity holds storage.objects.get, .create and .delete on the
# bucket and nothing else, so nothing here lists, patches or reads bucket
# settings: an expiry, or a counter's value, is written with the object at
# upload time, and an overwrite relies on the delete permission quoted above.

GCS_SCOPE = "https://www.googleapis.com/auth/devstorage.read_write"
_GCS_API = "https://storage.googleapis.com/storage/v1"
_GCS_UPLOAD_API = "https://storage.googleapis.com/upload/storage/v1"


def _adc_file() -> str | None:
    """The Application Default Credentials file, or None (as on Cloud Run).

    Read directly for the reason rag/gemini_client.py gives: google.auth.default()
    may shell out to gcloud to find a project, which hangs under an MCP stdio
    server. On Cloud Run there is no file and the metadata server answers.
    """
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit and os.path.isfile(explicit):
        return explicit
    try:
        from google.auth import _cloud_sdk

        path = _cloud_sdk.get_application_default_credentials_path()
    except Exception:  # noqa: BLE001 -- a private helper that moved: let default() decide
        return None
    return path if os.path.isfile(path) else None


class GoogleAuthToken:
    """`Authorization` headers from Application Default Credentials, refreshed when stale.

    The token goes into request headers only: it is never logged, and no error
    message here carries it.

    The bucket scope is passed as `default_scopes`, not `scopes`. A user ADC
    file (gcloud auth application-default login) keeps the scopes it was
    granted, and google-auth hands user credentials only `scopes`
    (_get_authorized_user_credentials), so asking it for this one failed with
    invalid_scope in the smoke test. Service accounts, and the metadata server
    on Cloud Run, take the default scope; the bucket's IAM role is the boundary.
    """

    def __init__(self, scopes: tuple[str, ...] = (GCS_SCOPE,)) -> None:
        self._scopes = list(scopes)
        self._credentials: Any = None  # google.auth credentials, imported lazily
        self._lock = threading.Lock()

    def __call__(self) -> dict[str, str]:
        import google.auth
        from google.auth.transport.requests import Request

        with self._lock:
            if self._credentials is None:
                path = _adc_file()
                if path:
                    self._credentials, _ = google.auth.load_credentials_from_file(
                        path, default_scopes=self._scopes)
                else:
                    self._credentials, _ = google.auth.default(default_scopes=self._scopes)
            credentials = self._credentials
            if not credentials.valid:
                credentials.refresh(Request())
            return {"Authorization": f"Bearer {credentials.token}"}


class _GcsObjects:
    """The four JSON API calls the tier needs. Every failure is CacheUnavailable."""

    def __init__(self, bucket: str, *, token: Callable[[], Mapping[str, str]],
                 timeout_s: float) -> None:
        self.bucket = bucket
        self._token = token
        self._timeout_s = timeout_s

    def _url(self, name: str) -> str:
        return f"{_GCS_API}/b/{quote(self.bucket, safe='')}/o/{quote(name, safe='')}"

    def _send(self, method: str, url: str, *, params: Mapping[str, str] | None = None,
              content: bytes | None = None, headers: Mapping[str, str] | None = None
              ) -> httpx.Response:
        try:
            auth = dict(self._token())
        except Exception as exc:  # noqa: BLE001 -- no credentials is a cache that cannot answer
            raise CacheUnavailable(
                f"Cloud Storage credentials unavailable ({type(exc).__name__})") from exc
        try:
            return httpx.request(method, url, params=params, content=content,
                                 headers={**auth, **(headers or {})}, timeout=self._timeout_s)
        except httpx.HTTPError as exc:
            raise CacheUnavailable(f"Cloud Storage request failed ({type(exc).__name__})") from exc

    @staticmethod
    def _refused(response: httpx.Response) -> CacheUnavailable:
        return CacheUnavailable(f"Cloud Storage returned HTTP {response.status_code}")

    def read(self, name: str) -> bytes | None:
        """The object's data, or None when it does not exist."""
        response = self._send("GET", self._url(name), params={"alt": "media"})
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise self._refused(response)
        return response.content

    def stat(self, name: str) -> dict | None:
        """The object resource (`generation`, `metadata`), or None when it does not exist."""
        response = self._send("GET", self._url(name))
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise self._refused(response)
        try:
            resource = response.json()
        except ValueError as exc:
            raise CacheUnavailable("Cloud Storage returned a non-JSON resource") from exc
        if not isinstance(resource, dict):
            raise CacheUnavailable("Cloud Storage returned an unexpected resource shape")
        return resource

    def write(self, name: str, data: bytes, metadata: Mapping[str, str], *,
              if_generation_match: int | None) -> bool:
        """Upload data and custom metadata in one request. False on 412: the
        precondition failed (0: the object exists; n: it is no longer generation n)."""
        boundary = f"crg{secrets.token_hex(12)}"
        resource = {"name": name, "contentType": "application/json", "metadata": dict(metadata)}
        body = b"".join([
            f"--{boundary}\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n".encode(),
            json.dumps(resource).encode(),
            f"\r\n--{boundary}\r\nContent-Type: application/json\r\n\r\n".encode(),
            data,
            f"\r\n--{boundary}--\r\n".encode(),
        ])
        params = {"uploadType": "multipart"}
        if if_generation_match is not None:
            params["ifGenerationMatch"] = str(if_generation_match)
        response = self._send(
            "POST", f"{_GCS_UPLOAD_API}/b/{quote(self.bucket, safe='')}/o", params=params,
            content=body, headers={"Content-Type": f"multipart/related; boundary={boundary}"})
        if response.status_code == 200:
            return True
        if response.status_code == 412:
            return False
        raise self._refused(response)

    def delete(self, name: str) -> None:
        response = self._send("DELETE", self._url(name))
        if response.status_code not in (200, 204, 404):
            raise self._refused(response)


class GcsCache:
    """Key/value entries in one Cloud Storage bucket, each carrying its expiry.

    The expiry is written into the object's data (read back with the value in
    one GET) and into its custom metadata (for a person browsing the bucket),
    and it is enforced on read. The bucket's own lifecycle rule deletes objects
    at 400 days of age, longer than any TTL used here.
    """

    def __init__(self, bucket: str, *, prefix: str = "kv/",
                 token: Callable[[], Mapping[str, str]] | None = None, timeout_s: float = 3.0,
                 clock: Callable[[], float] | None = None) -> None:
        self.name = "gcs"
        self.bucket = bucket
        self._prefix = prefix
        self._objects = _GcsObjects(bucket, token=token or GoogleAuthToken(), timeout_s=timeout_s)
        self._clock = clock

    def _now(self) -> float:
        return self._clock() if self._clock is not None else _now()

    def get_with_expiry(self, key: str) -> tuple[str | None, float | None]:
        data = self._objects.read(self._prefix + key)
        if data is None:
            return None, None
        try:
            payload = json.loads(data)
            value, expires_at = payload["value"], payload["expires_at"]
            if not isinstance(value, str):
                raise ValueError("cached value is not a string")
        except (ValueError, KeyError, TypeError) as exc:
            _log.warning("corrupt Cloud Storage cache entry ignored (%s)", exc)
            return None, None
        if expires_at is not None and self._now() >= expires_at:
            return None, None
        return value, expires_at

    def get(self, key: str) -> str | None:
        return self.get_with_expiry(key)[0]

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        expires_at = None if ttl_s is None else self._now() + ttl_s
        data = json.dumps({"value": value, "expires_at": expires_at}).encode()
        metadata = {} if expires_at is None else {"expires_at": repr(expires_at)}
        self._objects.write(self._prefix + key, data, metadata, if_generation_match=None)

    def delete(self, key: str) -> None:
        self._objects.delete(self._prefix + key)


class TieredCache:
    """The instance's own disk first, then the shared bucket.

    Writes go through to both. A bucket hit fills the disk with the time the
    entry has left, so the next read on this instance costs no request. Any
    bucket failure is logged and the bucket is left alone for `retry_after_s`:
    a dead bucket costs one timeout a minute, not one per report.
    """

    def __init__(self, near: CacheBackend, far: GcsCache, *, retry_after_s: float = 60.0,
                 clock: Callable[[], float] | None = None) -> None:
        self.near = near
        self.far = far
        self.name = f"{near.name}+{far.name}"
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._down_until = 0.0

    def _now(self) -> float:
        return self._clock() if self._clock is not None else _now()

    def _degrade(self, op: str, exc: Exception) -> None:
        self._down_until = self._now() + self._retry_after_s
        _log.warning("Cloud Storage cache tier failed on %s (%s); using this instance's disk "
                     "only for the next %.0f s", op, exc, self._retry_after_s)

    def get_traced(self, key: str) -> tuple[str | None, str]:
        value, served_by = get_traced(self.near, key)
        if value is not None or self._now() < self._down_until:
            return value, served_by
        try:
            value, expires_at = self.far.get_with_expiry(key)
        except CacheUnavailable as exc:
            self._degrade("get", exc)
            return None, self.near.name
        if value is None:
            return None, self.far.name
        ttl = None if expires_at is None else max(1, int(expires_at - self._now()))
        self.near.set(key, value, ttl)
        return value, self.far.name

    def get(self, key: str) -> str | None:
        return self.get_traced(key)[0]

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        self.near.set(key, value, ttl_s)
        if self._now() < self._down_until:
            return
        try:
            self.far.set(key, value, ttl_s)
        except CacheUnavailable as exc:
            self._degrade("set", exc)


# --- counters: the INCRBY plus EXPIRE contract -------------------------------
#
# Shaped on the spend guard's counter (rag/spend_guard.py on the hardening
# branch): `add(key, amount, ttl_s)` adds `amount`, which may be negative,
# renews the key's lifetime to `ttl_s` and returns the value after the change,
# raising CacheUnavailable when the shared store cannot answer.

class Counter(Protocol):
    name: str

    def add(self, key: str, amount: int, ttl_s: int) -> int: ...


class LocalCounter:
    """Integer counters with a lifetime, kept in this process."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.name = "local"
        self._clock = clock
        self._values: dict[str, tuple[int, float]] = {}
        self._lock = threading.Lock()

    def add(self, key: str, amount: int, ttl_s: int) -> int:
        now = self._clock()
        with self._lock:
            for stale in [k for k, (_, until) in self._values.items() if until <= now]:
                del self._values[stale]
            value = self._values.get(key, (0, 0.0))[0] + amount
            self._values[key] = (value, now + ttl_s)
            return value


class GcsCounter:
    """A shared counter in one Cloud Storage object, changed by compare-and-swap.

    Each add reads the object's resource (its `generation`, and the value and
    expiry kept in its custom metadata), then uploads the new value with
    ifGenerationMatch set to that generation, or to 0 when the object does not
    exist. A 412 means another writer got there first: read again and retry,
    with exponential backoff and jitter, up to `max_attempts` times. Anything
    else, or running out of attempts, is CacheUnavailable.
    """

    def __init__(self, bucket: str, *, prefix: str = "counter/",
                 token: Callable[[], Mapping[str, str]] | None = None, timeout_s: float = 3.0,
                 max_attempts: int = 8, base_delay_s: float = 0.05, max_delay_s: float = 1.0,
                 sleep: Callable[[float], None] = time.sleep,
                 jitter: Callable[[], float] = random.random,
                 clock: Callable[[], float] | None = None) -> None:
        self.name = "gcs"
        self._prefix = prefix
        self._objects = _GcsObjects(bucket, token=token or GoogleAuthToken(), timeout_s=timeout_s)
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep
        self._jitter = jitter
        self._clock = clock

    def _now(self) -> float:
        return self._clock() if self._clock is not None else _now()

    def add(self, key: str, amount: int, ttl_s: int) -> int:
        name = self._prefix + key
        for attempt in range(self._max_attempts):
            resource = self._objects.stat(name)
            now = self._now()
            generation, current = 0, 0
            if resource is not None:
                generation = int(resource["generation"])
                metadata = resource.get("metadata") or {}
                try:
                    if float(metadata.get("expires_at", "0")) > now:
                        current = int(metadata.get("value", "0"))
                except ValueError:
                    _log.warning("counter %r held an unreadable value; starting it again", key)
            value = current + amount
            expires_at = now + ttl_s
            data = json.dumps({"value": value, "expires_at": expires_at}).encode()
            if self._objects.write(name, data, {"value": str(value), "expires_at": repr(expires_at)},
                                   if_generation_match=generation):
                return value
            self._sleep(min(self._max_delay_s,
                            self._base_delay_s * (2 ** attempt) * (0.5 + self._jitter())))
        raise CacheUnavailable(
            f"counter {key!r}: lost the compare-and-swap {self._max_attempts} times in a row")

    def delete(self, key: str) -> None:
        self._objects.delete(self._prefix + key)


class FallbackCounter:
    """The shared counter while it answers; this process's own when it does not.

    After a failure the primary is left alone for `retry_after_s`. While it is,
    this instance counts on its own, on top of what the shared counter holds,
    and the log says so.
    """

    def __init__(self, primary: Counter, fallback: Counter, *, retry_after_s: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.primary = primary
        self.fallback = fallback
        self.name = f"{primary.name}+{fallback.name}"
        self._retry_after_s = retry_after_s
        self._clock = clock
        self._down_until = 0.0

    def add_traced(self, key: str, amount: int, ttl_s: int) -> tuple[int, Counter]:
        if self._clock() >= self._down_until:
            try:
                return self.primary.add(key, amount, ttl_s), self.primary
            except CacheUnavailable as exc:
                self._down_until = self._clock() + self._retry_after_s
                _log.warning("counter %r unavailable (%s): counting in this process for the "
                             "next %.0f s", self.primary.name, exc, self._retry_after_s)
        return self.fallback.add(key, amount, ttl_s), self.fallback

    def add(self, key: str, amount: int, ttl_s: int) -> int:
        return self.add_traced(key, amount, ttl_s)[0]


def _counter_for(env: Mapping[str, str]) -> Counter:
    bucket = env.get("CRG_CACHE_GCS_BUCKET")
    if bucket:
        _log.info("counters: Cloud Storage bucket %s, falling back to this process", bucket)
        return FallbackCounter(GcsCounter(bucket), LocalCounter())
    return LocalCounter()


@lru_cache(maxsize=1)
def counter_from_env() -> Counter:
    """The process-wide counter: shared through the bucket when one is configured."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return LocalCounter()  # hermetic tests: the shared counter is never touched
    return _counter_for(os.environ)


# Precomputed entries that ship with the code. The service runs at
# min-instances 0, and without Upstash credentials the live tier is a disk that
# dies with every instance, so a value computed once has to be IN the image to
# help the next cold visitor. Each entry is stored under the exact namespaced
# key JsonCache builds, so it can only answer the request, and the code
# version, folded into that key; anything else falls through to the live tier.
SEED_DIR = Path(__file__).resolve().parent / "cache_seed"


def source_bytes(path: Path | str) -> bytes:
    """A source file's bytes with CRLF line endings read as LF.

    Cache keys hash source code so that an edit retires what the old code
    computed. A Windows checkout (core.autocrlf) holds the same code with CRLF
    endings; hashed raw, it would build other keys than a Linux checkout and
    every shipped entry would miss there. Line endings are not code.
    """
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


class PrecomputedCache:
    """Read-only entries loaded from the seed files. Writes are ignored."""

    def __init__(self, entries: Mapping[str, str]) -> None:
        self.name = "precomputed"
        self._entries = dict(entries)

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, key: str) -> str | None:
        return self._entries.get(key)

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        return None


class SeededCache:
    """Precomputed entries first, then the live backend; writes go to the live one."""

    def __init__(self, seed: PrecomputedCache, backend: CacheBackend) -> None:
        self.seed = seed
        self.backend = backend
        self.name = f"{seed.name}+{backend.name}"

    def get_traced(self, key: str) -> tuple[str | None, str]:
        value = self.seed.get(key)
        if value is not None:
            return value, self.seed.name
        return get_traced(self.backend, key)

    def get(self, key: str) -> str | None:
        return self.get_traced(key)[0]

    def set(self, key: str, value: str, ttl_s: int | None = None) -> None:
        self.backend.set(key, value, ttl_s)


def load_seed_entries(directory: Path | str) -> dict[str, str]:
    """Every well-formed entry in every `*.json` seed file under `directory`.

    A file that does not parse is skipped with a warning naming it, and so is an
    entry without a namespaced key or a value: a bad seed costs its own entries,
    never the process.
    """
    entries: dict[str, str] = {}
    for path in sorted(Path(directory).glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload["entries"]
            if not isinstance(rows, list):
                raise ValueError("`entries` is not a list")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            _log.warning("cache seed %s unreadable (%s), skipped", path.name, exc)
            continue
        malformed = 0
        for row in rows:
            key = row.get("key") if isinstance(row, dict) else None
            value = row.get("value") if isinstance(row, dict) else None
            if not isinstance(key, str) or not key.startswith(f"{_KEY_PREFIX}:") or value is None:
                malformed += 1
                continue
            entries[key] = value if isinstance(value, str) else json.dumps(
                value, ensure_ascii=False, separators=(",", ":")
            )
        if malformed:
            _log.warning("cache seed %s: %d malformed entries skipped", path.name, malformed)
    return entries


def write_seed_file(path: Path | str, *, header: Mapping[str, object], entries: list[dict]) -> None:
    """Write a seed file: the header fields, then one entry per line, sorted by key.

    One line per entry keeps a regenerated file reviewable as a diff. Each entry
    needs `key` (the full JsonCache key) and `value` (the stored model, as JSON);
    any other fields (a city name, a region) are for the reader and ignored on load.
    """
    rows = sorted(entries, key=lambda row: row["key"])
    head = json.dumps(dict(header), ensure_ascii=False, indent=1)[:-2]  # drop the closing "\n}"
    body = ",\n".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows)
    text = f'{head},\n "entries": [\n{body}\n ]\n}}\n' if rows else f'{head},\n "entries": []\n}}\n'
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


def _with_seed(backend: CacheBackend, env: Mapping[str, str]) -> CacheBackend:
    """Put the precomputed entries in front of `backend`, when there are any.

    CRG_CACHE_SEED_DIR overrides the directory; set to an empty string it turns
    the seed off (what a precompute run wants, so it computes instead of reading).
    """
    raw = env.get("CRG_CACHE_SEED_DIR")
    if raw == "":
        return backend
    directory = SEED_DIR if raw is None else Path(raw)
    entries = load_seed_entries(directory)
    if not entries:
        return backend
    _log.info("cache seed: %d precomputed entries from %s", len(entries), directory)
    return SeededCache(PrecomputedCache(entries), backend)


def _backend_from_env(env: Mapping[str, str]) -> CacheBackend:
    """Pick a backend from configuration. Pure in `env` so it is directly testable.

    CRG_CACHE_GCS_BUCKET wins when set: this instance's disk, then the shared
    bucket (the seed goes in front of either, in `cache_from_env`).
    """
    root = Path(env.get("CRG_CACHE_DIR", _DEFAULT_ROOT))
    disk = DiskCache(root)
    bucket = env.get("CRG_CACHE_GCS_BUCKET")
    if bucket:
        _log.info("cache backend: disk at %s, then Cloud Storage bucket %s", root, bucket)
        return TieredCache(disk, GcsCache(bucket))
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
    return _with_seed(_backend_from_env(os.environ), os.environ)


# Disk sampling for cache-read telemetry. Every read used to append a line to
# the daily JSONL: an open + write per cache GET, which on a warm request is the
# bulk of the telemetry volume and none of its information. So MISSES always
# land (they are what a cost or cold-start investigation is about) and hits land
# 1-in-N. Every hit still reaches the IN-MEMORY ring, which is what the UI cache
# badge (`Span.summary()["cache_hits"]`) and scripts/prewarm.py read, so nothing
# user-visible changes.
_HIT_SAMPLE_N = 20
_hit_counter = itertools.count()


def _persist_cache_event(hit: bool) -> bool:
    """Should this cache read reach the durable JSONL sink? (see above)"""
    if not hit:
        return True
    return next(_hit_counter) % _HIT_SAMPLE_N == 0


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
               tokens_in=0, tokens_out=0, retries=0, ok=True, cached=hit,
               persist=_persist_cache_event(hit))
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
