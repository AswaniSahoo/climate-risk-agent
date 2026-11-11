"""A daily spend limit on model calls, and a switch that pauses them per scope.

WHY AT THE SEAM. rag/gemini_client.py is the only module that calls Gemini, so a
limit enforced there holds for every caller at once (the UI, the API, the MCP
server, the evals) and no later caller can forget it. This module holds the
policy and the counters; gemini_client turns a refusal into
ModelBudgetExceeded, a GeminiError, so each caller takes the degraded path it
already has for a failed model call: the report ships without IPCC citations,
and agent/graph.py says why in the report.

WHAT IS COUNTED: money, in micro-US-dollars per UTC day, at the list prices in
obs/telemetry.py. Before a call the guard RESERVES the most the call can cost,
and the call starts only if that fits in what is left of the day:
- generate: the prompt at one token per two characters (Google documents about
  four for Gemini, so this is double), plus 8,192 thinking tokens (on Vertex
  the default thinking budget of gemini-2.5-flash is "Auto (up to 8,192
  tokens)") and 2,048 visible ones, at the output price, which bills reasoning;
- embed: each text at one token per two characters, at the embedding price.
After a generate call the reservation is settled to what the response's usage
metadata says was billed. A call that failed on a rate limit, or before any
request was built, gives its reservation back; any other failure keeps the
prompt part, because the guard cannot see whether a failed call was billed.
Embedding responses carry no usage, so their reservation is their charge.

WHERE THE COUNTER LIVES. In Upstash Redis when UPSTASH_REDIS_REST_URL and
UPSTASH_REDIS_REST_TOKEN are set (one INCRBY + EXPIRE transaction per change,
contract quoted in tools/cache_backend.py), so every replica spends from one
budget. Otherwise, and for 60 s after any Upstash failure, in this process:
each instance then has a budget of its own, and a restart resets it.

SCOPES. `call_scope()` marks a stretch of work (one report in the UI). A scope
counts the calls reserved inside it, and a scope opened with `blocked=` refuses
every call with that sentence, without touching the counter. The scope lives in
a context variable, which LangGraph copies into the nodes it runs.

The per-client request limit for the API (`client_rate_limited`) counts on the
same backends.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import threading
import time
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

from obs.telemetry import price_per_mtok
from tools.cache_backend import CACHE_SCHEMA_VERSION, CacheUnavailable

if TYPE_CHECKING:
    from tools.cache_backend import UpstashRedisCache

_log = logging.getLogger(__name__)

# The safe defaults. DEPLOY.md ("Spend guard") shows the arithmetic: 30 days at
# US$0.15 is US$4.50, about INR 422 at the RBI reference rate of 2026-11-11.
DEFAULT_DAILY_BUDGET_USD = 0.15
DEFAULT_SESSION_FRESH_REPORTS = 5
DEFAULT_API_RATE_LIMIT = 10
DEFAULT_API_RATE_WINDOW_S = 60

# What one generate call reserves (see the module docstring).
RESERVE_CHARS_PER_TOKEN = 2
RESERVE_THINKING_TOKENS = 8_192
RESERVE_VISIBLE_TOKENS = 2_048

_DAY_TTL_S = 2 * 24 * 3600  # a day's key outlives its day, then deletes itself
_UPSTASH_RETRY_AFTER_S = 60.0

_now = time.time  # module attribute so tests can move the clock

_BUDGET_REASON = (
    "The daily model budget for this deployment is used up, so no new model call "
    "runs until it resets at 00:00 UTC. Answers already cached still carry their "
    "citations."
)


class BudgetRefused(RuntimeError):
    """A model call may not start. The message is a sentence for a person;
    `detail` carries the numbers and the setting to change, for the log."""

    def __init__(self, reason: str, *, detail: str = "") -> None:
        super().__init__(reason)
        self.detail = detail or reason


# --- settings ----------------------------------------------------------------

_warned: set[str] = set()
_warn_lock = threading.Lock()


def _warn_once(key: str, message: str, *args: object) -> None:
    with _warn_lock:
        if key in _warned:
            return
        _warned.add(key)
    _log.warning(message, *args)


def _raw(env: Mapping[str, str], name: str) -> str | None:
    """The trimmed value, "off" as None, or "" when unset or blank."""
    raw = (env.get(name) or "").strip()
    return None if raw.lower() == "off" else raw


def _bad(name: str, raw: str, minimum: object, default: object) -> None:
    _warn_once(name, "%s=%r is not a number >= %s or 'off'; using the default %s",
               name, raw, minimum, default)


def _float_setting(env: Mapping[str, str], name: str, default: float) -> float | None:
    raw = _raw(env, name)
    if raw is None:
        return None
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value < 0:
        _bad(name, raw, 0, default)
        return default
    return value


def _int_setting(env: Mapping[str, str], name: str, default: int, minimum: int) -> int | None:
    raw = _raw(env, name)
    if raw is None:
        return None
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        _bad(name, raw, minimum, default)
        return default
    if value < minimum:
        _bad(name, raw, minimum, default)
        return default
    return value


def daily_budget_micro_usd(env: Mapping[str, str] = os.environ) -> int | None:
    """CRG_DAILY_MODEL_BUDGET_USD in micro-dollars; None when it is "off".
    Anything unreadable keeps the default, never "no limit"."""
    usd = _float_setting(env, "CRG_DAILY_MODEL_BUDGET_USD", DEFAULT_DAILY_BUDGET_USD)
    return None if usd is None else int(Decimal(str(usd)) * 1_000_000)


def session_fresh_report_limit(env: Mapping[str, str] = os.environ) -> int | None:
    """CRG_SESSION_FRESH_REPORTS: reports needing new model calls per UI session."""
    return _int_setting(env, "CRG_SESSION_FRESH_REPORTS", DEFAULT_SESSION_FRESH_REPORTS, 0)


def api_rate_limit(env: Mapping[str, str] = os.environ) -> tuple[int, int] | None:
    """(requests, window seconds) per API client; None when CRG_API_RATE_LIMIT is "off"."""
    limit = _int_setting(env, "CRG_API_RATE_LIMIT", DEFAULT_API_RATE_LIMIT, 0)
    if limit is None:
        return None
    window = _int_setting(env, "CRG_API_RATE_WINDOW_S", DEFAULT_API_RATE_WINDOW_S, 1)
    return limit, window if window is not None else DEFAULT_API_RATE_WINDOW_S


def unlimited_unless_set(env: MutableMapping[str, str] = os.environ) -> bool:
    """For the eval runners: no daily budget unless one was set explicitly.

    An eval measures every question, and a refusal halfway through would be
    scored as answers the model got wrong (scripts/eval_gate.py does not read
    `errors`). Its cost is bounded by its fixed question count instead. Returns
    True when it turned the budget off.
    """
    if "CRG_DAILY_MODEL_BUDGET_USD" in env:
        return False
    env["CRG_DAILY_MODEL_BUDGET_USD"] = "off"
    return True


def session_limit_reason(limit: int) -> str:
    return (
        f"This session has reached its limit of {limit} reports that need new model "
        "calls, so new model calls are paused for it. Answers already cached still "
        "carry their citations."
    )


# --- counters ----------------------------------------------------------------

class Counter(Protocol):
    name: str

    def add(self, key: str, amount: int, ttl_s: int) -> int:
        """Add `amount` (which may be negative) and reset the key's lifetime;
        returns the value after the change."""
        ...


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


class UpstashCounter:
    """The shared counter: one INCRBY + EXPIRE transaction on Upstash Redis."""

    def __init__(self, cache: UpstashRedisCache) -> None:
        self.name = "redis"
        self._cache = cache

    def add(self, key: str, amount: int, ttl_s: int) -> int:
        return self._cache.incrby_with_ttl(key, amount, ttl_s)


class FallbackCounter:
    """The shared counter while it answers; this process's own when it does not.

    After a failure the primary is left alone for `retry_after_s`, so a dead
    Upstash costs one timeout a minute rather than one per model call. Each
    fallback is logged: while it lasts, this instance spends from a budget of
    its own on top of what the shared counter already holds.
    """

    def __init__(
        self,
        primary: Counter,
        fallback: Counter,
        *,
        retry_after_s: float = _UPSTASH_RETRY_AFTER_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
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
                _log.warning(
                    "spend counter %r unavailable (%s): counting in this process for "
                    "the next %.0f s, so this instance spends from its own budget meanwhile",
                    self.primary.name, exc, self._retry_after_s,
                )
        return self.fallback.add(key, amount, ttl_s), self.fallback

    def add(self, key: str, amount: int, ttl_s: int) -> int:
        return self.add_traced(key, amount, ttl_s)[0]


_counter_lock = threading.Lock()
_counter: Counter | None = None


def _counter_from_env(env: Mapping[str, str]) -> Counter:
    if env.get("PYTEST_CURRENT_TEST"):
        return LocalCounter()  # hermetic tests: the shared counter is never touched
    url, token = env.get("UPSTASH_REDIS_REST_URL"), env.get("UPSTASH_REDIS_REST_TOKEN")
    if url and token:
        from tools.cache_backend import UpstashRedisCache

        # No URL or token in the log line: the token is a secret.
        _log.info("spend counter: Upstash Redis (REST), falling back to this process")
        return FallbackCounter(
            UpstashCounter(UpstashRedisCache(url, token, timeout_s=2.0)), LocalCounter()
        )
    _log.info("spend counter: this process only (set UPSTASH_REDIS_REST_URL and "
              "UPSTASH_REDIS_REST_TOKEN to share one budget across instances)")
    return LocalCounter()


def _get_counter() -> Counter:
    global _counter
    with _counter_lock:
        if _counter is None:
            _counter = _counter_from_env(os.environ)
        return _counter


def reset(counter: Counter | None = None) -> None:
    """Start over on `counter`, or on one rebuilt from the environment (tests)."""
    global _counter
    with _counter_lock:
        _counter = counter


def _add(key: str, amount: int, ttl_s: int) -> tuple[int, Counter]:
    counter = _get_counter()
    if isinstance(counter, FallbackCounter):
        return counter.add_traced(key, amount, ttl_s)
    return counter.add(key, amount, ttl_s), counter


def _adjust(counter: Counter, key: str, delta: int, ttl_s: int) -> None:
    if not delta:
        return
    try:
        counter.add(key, delta, ttl_s)
    except CacheUnavailable as exc:
        # The reservation stays counted in full: an over-count, never an under-count.
        _log.warning("could not settle a model-call reservation (%s); today's count "
                     "stays higher than what was spent", exc)


def _day_key() -> str:
    day = time.strftime("%Y-%m-%d", time.gmtime(_now()))
    return f"crg:{CACHE_SCHEMA_VERSION}:spend:{day}"


def spent_today_micro_usd() -> int:
    """Today's (UTC) count right now, including reservations still in flight."""
    return _add(_day_key(), 0, _DAY_TTL_S)[0]


# --- scopes ------------------------------------------------------------------

@dataclass
class CallScope:
    """One stretch of work: the calls reserved in it, and whether it may make any."""

    blocked: str | None = None
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def _count(self) -> None:
        with self._lock:
            self.calls += 1


_SCOPE: ContextVar[CallScope | None] = ContextVar("crg_model_call_scope", default=None)


@contextmanager
def call_scope(*, blocked: str | None = None) -> Iterator[CallScope]:
    """Count the model calls made inside the block; refuse them all if `blocked`."""
    scope = CallScope(blocked=blocked)
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        _SCOPE.reset(token)


def current_scope() -> CallScope | None:
    return _SCOPE.get()


# --- reservations ------------------------------------------------------------

@dataclass(frozen=True)
class Reservation:
    """What one call holds against the day's budget until it is settled."""

    model: str
    key: str | None  # None: the budget is off and nothing is held
    micro_usd: int = 0
    prompt_micro_usd: int = 0
    counter: Counter | None = None


def _tokens(chars: int) -> int:
    return math.ceil(chars / RESERVE_CHARS_PER_TOKEN)


def _cost(tokens: int, usd_per_mtok: float) -> int:
    """Micro-dollars for `tokens` at a price per million: tokens x price, rounded up."""
    return math.ceil(Decimal(tokens) * Decimal(str(usd_per_mtok)))


def _usd(micro_usd: int) -> str:
    return f"US${micro_usd / 1_000_000:.4f}"


def _price(model: str) -> tuple[float, float]:
    price = price_per_mtok(model)
    if price is None:
        raise BudgetRefused(
            f"No price is configured for model {model!r}, so the daily model budget "
            "cannot be enforced and the call was not sent. Add the model to "
            "obs/telemetry.py, or set CRG_DAILY_MODEL_BUDGET_USD=off."
        )
    return price


def _reserve(model: str, amounts: Callable[[], tuple[int, int]]) -> Reservation:
    scope = _SCOPE.get()
    if scope is not None and scope.blocked:
        raise BudgetRefused(scope.blocked)
    limit = daily_budget_micro_usd()
    if limit is None:
        if scope is not None:
            scope._count()
        return Reservation(model=model, key=None)
    total, prompt_part = amounts()
    if total > limit:
        raise BudgetRefused(_BUDGET_REASON, detail=(
            f"one call can cost up to {_usd(total)}, more than the whole daily budget "
            f"of {_usd(limit)} (CRG_DAILY_MODEL_BUDGET_USD)"))
    key = _day_key()
    value, counter = _add(key, total, _DAY_TTL_S)
    if value > limit:
        _adjust(counter, key, -total, _DAY_TTL_S)
        raise BudgetRefused(_BUDGET_REASON, detail=(
            f"{_usd(value - total)} of the {_usd(limit)} daily budget is spent or held "
            f"today (UTC) and this call can cost up to {_usd(total)} "
            "(CRG_DAILY_MODEL_BUDGET_USD)"))
    if scope is not None:
        scope._count()
    return Reservation(model=model, key=key, micro_usd=total,
                       prompt_micro_usd=prompt_part, counter=counter)


def reserve_generate(model: str, prompt: str) -> Reservation:
    """Hold the most one generate call with this prompt can cost, or refuse."""
    def amounts() -> tuple[int, int]:
        price_in, price_out = _price(model)
        prompt_cost = _cost(_tokens(len(prompt)), price_in)
        output_cost = _cost(RESERVE_THINKING_TOKENS + RESERVE_VISIBLE_TOKENS, price_out)
        return prompt_cost + output_cost, prompt_cost

    return _reserve(model, amounts)


def reserve_embed(model: str, texts: list[str]) -> Reservation:
    """Hold what embedding `texts` costs, which is also its charge, or refuse."""
    def amounts() -> tuple[int, int]:
        price_in, _ = _price(model)
        cost = _cost(sum(_tokens(len(text)) for text in texts), price_in)
        return cost, cost

    return _reserve(model, amounts)


def settle(reservation: Reservation, *, tokens_in: int, tokens_out: int) -> None:
    """Replace a finished call's reservation with what its usage says was billed.
    No usage at all keeps the reservation: an unknown cost is not a zero cost."""
    if reservation.key is None or reservation.counter is None:
        return
    if tokens_in <= 0 and tokens_out <= 0:
        return
    price_in, price_out = _price(reservation.model)
    billed = _cost(tokens_in, price_in) + _cost(tokens_out, price_out)
    _adjust(reservation.counter, reservation.key, billed - reservation.micro_usd, _DAY_TTL_S)


def release(reservation: Reservation, *, keep_prompt: bool) -> None:
    """A failed call: give its reservation back, or keep only the prompt part."""
    if reservation.key is None or reservation.counter is None:
        return
    kept = reservation.prompt_micro_usd if keep_prompt else 0
    _adjust(reservation.counter, reservation.key, kept - reservation.micro_usd, _DAY_TTL_S)


# --- per-client request limit (the API) --------------------------------------

def client_rate_limited(scope_name: str, client: str,
                        env: Mapping[str, str] = os.environ) -> int | None:
    """Count one request from `client`: the seconds to wait if it is over the
    limit for this window, else None.

    The address is hashed before it becomes a key (unsalted: that keeps it out
    of plain sight in the shared cache, not secret), and keys expire after two
    windows.
    """
    limits = api_rate_limit(env)
    if limits is None:
        return None
    limit, window = limits
    now = _now()
    index = int(now // window)
    digest = hashlib.sha256(client.encode("utf-8")).hexdigest()[:16]
    key = f"crg:{CACHE_SCHEMA_VERSION}:ratelimit:{scope_name}:{digest}:{index}"
    count, _ = _add(key, 1, window * 2)
    if count > limit:
        return max(1, math.ceil((index + 1) * window - now))
    return None
