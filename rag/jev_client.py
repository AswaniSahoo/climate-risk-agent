"""The single seam between this project and TypeSafe's Jev ("System One") API.

Jev answers typed questions about a piece of content with probabilities, not
text: one POST carries a `state` and named questions (`noul` yes/no, `choice`,
`score`), and the reply carries one answer per question plus token usage. This
module is the only place that talks to it, for the same reason
rag/gemini_client.py is the only place that talks to Gemini: a property that
matters (timeouts, retries, telemetry, the key never leaking) is enforced at
one chokepoint instead of by discipline at every call site.

Contract, verified 2026-11-11 against https://api.typesafe.ai/openapi.json
(spec 0.2.0) and https://docs.typesafe.ai/llms-full.txt:

- `POST /v1/systemone` with `Authorization: Bearer <key>`, body
  `{model, state, questions}`, reply `{model, answers, usage}`.
- Request rules (docs "api/errors"): body at most 32 MiB, at most 64 nested
  levels in `state`, `instructions` and `criteria`, finite numbers only.
  Context (docs "models"): 64k tokens per request, 32k per question.
- Errors: 400, 401, 413, 415, 422, 429, 529. The docs say to retry 429 and 529
  "with exponential backoff"; the official SDK's default retry set is
  {408, 429, 500-599} and it honours `Retry-After` and `retry-after-ms`
  (docs "sdk/python/api/retries"). 402 means no credits (TypeSafe help
  center), so it is treated like a bad key: fail fast, do not retry.
- Price: $0.042 per million input tokens, output tokens free (docs "models").

Why not the official `typesafe-sdk`: the surface this product needs is one
POST, httpx is already a runtime dependency, and the SDK would add httpx2,
tenacity and truststore to a slim runtime image. The retry defaults below
mirror the SDK's, with a much tighter time budget, because a triage answer that
arrives after the user has given up is worth nothing.

Failure policy, the part that matters most: Jev is a signal, never a
dependency. Every failure surfaces as a typed `JevError` subclass, callers
catch it and fall back to exactly today's behaviour, and the failure is
recorded in telemetry with a short reason (never the key, never user text).
A bad key or exhausted credits open a circuit for five minutes and three
outages in a row open it for thirty seconds, so a broken dependency costs one
round trip, not one per request.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Union

import httpx

_log = logging.getLogger(__name__)

# The default target. Never caller-supplied: JEV_BASE_URL may only pick one of
# the vetted PROVIDERS below (tests/test_security.py pins the list), so the
# request target is not something a user, or a stray config value, can steer.
JEV_BASE_URL = "https://api.typesafe.ai"
SYSTEMONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"

# A versioned ID, not the `jev-latest` alias. The docs: "An alias moves when a
# new release ships, so its answers can change without any change to your
# code", and "If you have tuned confidence thresholds against one version,
# send that version's ID instead of the alias". Override per deploy.
DEFAULT_MODEL = "jev-1.14.0"


@dataclass(frozen=True)
class Provider:
    """One vetted place to send Jev requests, and what it allows.

    Same request schema at every provider; what differs is checked here, not
    discovered at run time: the model names granted, the context limit, how
    long a round trip takes, and where the request id comes back.
    """

    name: str
    base_url: str
    default_model: str
    max_request_tokens: int
    max_question_tokens: int
    triage_read_timeout_s: float
    triage_budget_s: float
    verify_read_timeout_s: float
    verify_budget_s: float
    request_id_header: str


PROVIDERS: dict[str, Provider] = {
    # TypeSafe direct. Limits and timings from docs.typesafe.ai ("models";
    # "Most calls complete in about 100 ms"), read 2026-11-11.
    "https://api.typesafe.ai": Provider(
        name="typesafe", base_url="https://api.typesafe.ai", default_model=DEFAULT_MODEL,
        max_request_tokens=64_000, max_question_tokens=32_000,
        triage_read_timeout_s=1.5, triage_budget_s=2.5,
        verify_read_timeout_s=3.0, verify_budget_s=6.0,
        request_id_header="x-typesafe-request-id",
    ),
    # Experiential Labs, an AI gateway that relays the same request to TypeSafe
    # (reply field "provider": "typesafe"). Read on 2026-11-11: its model list
    # grants only the `jev-latest` alias (jev-1.14.0 is refused with HTTP 403
    # model_not_granted) with context_window_tokens 32000, and one triage
    # request took 1,015 ms end to end, so its timeouts are wider.
    "https://api.experientiallabs.ai": Provider(
        name="experientiallabs", base_url="https://api.experientiallabs.ai",
        default_model="jev-latest", max_request_tokens=32_000, max_question_tokens=32_000,
        triage_read_timeout_s=3.0, triage_budget_s=4.0,
        verify_read_timeout_s=4.0, verify_budget_s=8.0,
        request_id_header="x-request-id",
    ),
}

USD_PER_MTOK_INPUT = 0.042  # docs.typesafe.ai/models, verified 2026-11-11

MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_NESTING = 64
PER_REQUEST_CONTEXT_TOKENS = 64_000
PER_QUESTION_CONTEXT_TOKENS = 32_000
# No tokenizer is published, so the local size check estimates. Two characters
# per token is deliberately pessimistic for English (an independent test
# measured 2.06 on Norwegian, lindfors.no 2026-09-18), so a request this check
# passes is inside the real limit; the server's own 400 is the backstop.
CHARS_PER_TOKEN_FLOOR = 2.0

RETRYABLE_STATUSES = frozenset({408, 429, *range(500, 600)})
# Every later call would fail the same way, so these open the circuit.
FATAL_STATUSES = frozenset({401, 402, 403})
FATAL_COOLDOWN_S = 300.0
OUTAGE_FAILURES_TO_OPEN = 3
OUTAGE_COOLDOWN_S = 30.0

# A reply to our questions is a few hundred bytes; anything near this is not one.
MAX_REPLY_BYTES = 1024 * 1024
# Printable ASCII, no space: what an Authorization header value can carry.
_KEY_CHARS = re.compile(r"[\x21-\x7e]+")

MODES = ("off", "triage", "verify", "all")


class _OutOfTime(Exception):
    """Internal: the budget ran out while a reply was still arriving."""


class _ReplyTooLarge(Exception):
    """Internal: the reply passed MAX_REPLY_BYTES."""


_warned: set[str] = set()
_warned_lock = threading.Lock()


def _warn_once(message: str) -> None:
    """Loud, but once per process: a misconfiguration must not flood the log."""
    with _warned_lock:
        if message in _warned:
            return
        _warned.add(message)
    _log.warning(message)


# --- the feature flag -------------------------------------------------------


def jev_mode() -> str:
    """Resolve CRG_JEV (off | triage | verify | all). Unknown value -> off, loudly."""
    mode = os.environ.get("CRG_JEV", "off").strip().lower() or "off"
    if mode not in MODES:
        _warn_once(f"CRG_JEV={mode!r} is not one of {MODES}; Jev stays OFF")
        return "off"
    return mode


def triage_enabled() -> bool:
    return jev_mode() in ("triage", "all")


def verify_enabled() -> bool:
    return jev_mode() in ("verify", "all")


# --- errors -----------------------------------------------------------------


class JevError(RuntimeError):
    """Anything that kept Jev from answering. Callers fall back on it.

    `reason` is short and safe to log or persist: a status code, an error type
    or an exception class name, never response text that could echo input.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class JevConfigError(JevError):
    """No API key is configured."""


class JevRequestError(JevError):
    """Our request was refused (a non-retryable 4xx) or broke a documented limit."""

    def __init__(self, reason: str, *, status: int | None = None) -> None:
        super().__init__(reason)
        self.status = status


class JevUnavailable(JevError):
    """Timeouts, overload or 5xx after the retry budget, or an open circuit."""


class JevResponseError(JevError):
    """A 200 whose body breaks the documented response schema."""


def get_provider() -> Provider:
    """The provider JEV_BASE_URL names (default TypeSafe direct). Raises JevConfigError.

    Picks from the vetted list only. A value outside it is refused rather than
    followed: where user questions are sent is a code review decision, not a
    config typo away.
    """
    raw = os.environ.get("JEV_BASE_URL", "").strip().rstrip("/") or JEV_BASE_URL
    provider = PROVIDERS.get(raw)
    if provider is None:
        _warn_once(f"JEV_BASE_URL is not a vetted Jev provider (vetted: {sorted(PROVIDERS)}); "
                   "Jev calls are skipped")
        raise JevConfigError("JEV_BASE_URL is not a vetted Jev provider")
    return provider


# --- typed answers ----------------------------------------------------------


@dataclass(frozen=True)
class NoulAnswer:
    """Probability of yes (a true statement), 0 to 1."""

    noul: float


@dataclass(frozen=True)
class ChoiceAnswer:
    """The top option, its confidence, and the probability of every option."""

    choice: str
    confidence: float
    probabilities: Mapping[str, float]

    def p(self, option: str) -> float:
        return float(self.probabilities.get(option, 0.0))

    @property
    def top_p(self) -> float:
        return self.p(self.choice)


@dataclass(frozen=True)
class ScoreAnswer:
    """Expected level on an ordered rubric, with the per-level probabilities."""

    score: float
    confidence: float
    probabilities: Mapping[str, float]


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]


@dataclass(frozen=True)
class Usage:
    input_tokens: int
    output_tokens: int
    # What the provider says it charged, when it says (the Experiential Labs
    # gateway sends `usage.cost`). None means "not reported", never "free".
    reported_cost_usd: float | None = None


@dataclass(frozen=True)
class JevResult:
    """One parsed reply, plus what it cost to get it."""

    model: str
    answers: Mapping[str, Answer]
    usage: Usage
    latency_ms: float
    retries: int = 0
    cached: bool = False
    request_id: str | None = None
    upstream: str | None = None  # the gateway's "provider" field, when relayed

    def noul(self, name: str) -> float:
        answer = self.answers.get(name)
        if not isinstance(answer, NoulAnswer):
            raise JevResponseError(f"question {name!r} has no noul answer")
        return answer.noul

    def choice(self, name: str) -> ChoiceAnswer:
        answer = self.answers.get(name)
        if not isinstance(answer, ChoiceAnswer):
            raise JevResponseError(f"question {name!r} has no choice answer")
        return answer

    @property
    def cost_usd(self) -> float:
        """Spend for this reply: what the provider reported, else input tokens at
        the list price. A cache hit is free."""
        if self.cached:
            return 0.0
        if self.usage.reported_cost_usd is not None:
            return self.usage.reported_cost_usd
        return self.list_price_usd

    @property
    def list_price_usd(self) -> float:
        """Input tokens at TypeSafe's list price, whatever the provider charged."""
        return 0.0 if self.cached else self.usage.input_tokens * USD_PER_MTOK_INPUT / 1e6


# --- retry policy -----------------------------------------------------------


@dataclass(frozen=True)
class RetryPolicy:
    """How long one logical call may take, all attempts and waits included.

    Defaults mirror the official SDK's shape (2 retries, doubling backoff, 25%
    jitter), with the total budget cut from its 30 s to what each caller can
    afford: triage sits in front of a person waiting for a first answer. The
    connect and read waits of each attempt are carved out of what is left, and
    the reply is read chunk by chunk against the deadline, so a call can only
    outlast `budget_s` by one chunk's wait (JevClient._exchange).
    """

    max_retries: int = 2
    backoff_initial_s: float = 0.25
    backoff_max_s: float = 2.0
    jitter: float = 0.25
    connect_timeout_s: float = 1.0
    read_timeout_s: float = 1.5
    budget_s: float = 2.5


def _env_seconds(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    try:
        value = float(raw) if raw.strip() else default
    except ValueError:
        _warn_once(f"{name}={raw!r} is not a number of seconds; using {default}")
        return default
    return value if value > 0 else default


def _policy_provider() -> Provider:
    """The configured provider's timings, or TypeSafe's when the config is bad
    (the client refuses that config anyway, so these timings are never used)."""
    try:
        return get_provider()
    except JevConfigError:
        return PROVIDERS[JEV_BASE_URL]


def triage_policy() -> RetryPolicy:
    provider = _policy_provider()
    return RetryPolicy(
        read_timeout_s=provider.triage_read_timeout_s,
        budget_s=_env_seconds("CRG_JEV_TRIAGE_BUDGET_S", provider.triage_budget_s),
    )


def verify_policy() -> RetryPolicy:
    provider = _policy_provider()
    return RetryPolicy(
        read_timeout_s=provider.verify_read_timeout_s,
        budget_s=_env_seconds("CRG_JEV_VERIFY_BUDGET_S", provider.verify_budget_s),
    )


# --- request encoding and the documented limits ------------------------------


def _depth(value: Any) -> int:
    """Nesting depth of objects and arrays (a scalar is 0, `{}` is 1)."""
    if isinstance(value, dict):
        return 1 + max((_depth(v) for v in value.values()), default=0)
    if isinstance(value, (list, tuple)):
        return 1 + max((_depth(v) for v in value), default=0)
    return 0


def _estimated_tokens(value: Any) -> float:
    return len(json.dumps(value, ensure_ascii=False)) / CHARS_PER_TOKEN_FLOOR


def encode_request(
    model: str,
    state: Any,
    questions: Mapping[str, Mapping[str, Any]],
    *,
    max_request_tokens: int = PER_REQUEST_CONTEXT_TOKENS,
    max_question_tokens: int = PER_QUESTION_CONTEXT_TOKENS,
) -> bytes:
    """The exact bytes to send, after every documented limit is checked locally.

    Key order is preserved on purpose, never sorted: Jev "leans toward the
    option that comes first" (docs "model-jaggedness"), so the order of a
    Choice's options is part of the question. The context limits are the
    provider's (TypeSafe direct 64k per request; the gateway 32k).
    """
    if not questions:
        raise JevRequestError("a request needs at least one question")
    parts = [("state", state)]
    for name, question in questions.items():
        parts.append((f"{name}.instructions", question.get("instructions")))
        parts.append((f"{name}.criteria", question.get("criteria")))
    for where, value in parts:
        if _depth(value) > MAX_NESTING:
            raise JevRequestError(f"{where} nests deeper than {MAX_NESTING} levels")

    body = {"model": model, "state": state, "questions": dict(questions)}
    try:
        text = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except ValueError as exc:  # NaN or Infinity: the API rejects non-finite numbers
        raise JevRequestError(f"request holds a non-finite number ({exc})") from exc
    data = text.encode("utf-8")
    if len(data) > MAX_BODY_BYTES:
        raise JevRequestError(f"body is {len(data)} bytes, over the 32 MiB limit", status=413)

    state_tokens = _estimated_tokens(state)
    question_tokens = {name: _estimated_tokens(q) for name, q in questions.items()}
    if state_tokens + sum(question_tokens.values()) > max_request_tokens:
        raise JevRequestError(
            f"request may exceed the {max_request_tokens:,}-token per-request context")
    for name, tokens in question_tokens.items():
        if state_tokens + tokens > max_question_tokens:
            raise JevRequestError(f"question {name!r} may exceed the "
                                  f"{max_question_tokens:,}-token per-question context")
    return data


# --- response parsing -------------------------------------------------------


# Every message below is fixed text plus names WE chose (question names, option
# names checked against our own list). No value from the reply is ever copied
# into one: a reason is logged, persisted to the telemetry sink and shown as a
# status, and a reply field can echo the user's question back.


def _probability(value: Any, where: str) -> float:
    """A finite real in [0, 1]. Booleans are ints in Python and are refused here."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise JevResponseError(f"{where} is not a finite number")
    if not -1e-9 <= value <= 1 + 1e-9:
        raise JevResponseError(f"{where} is outside [0, 1]")
    return min(max(float(value), 0.0), 1.0)


def _token_count(value: Any, where: str) -> int:
    """A non-negative count; an integral float such as 120.0 is read as 120."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevResponseError(f"{where} is not a number")
    if not math.isfinite(value) or value < 0 or value != int(value):
        raise JevResponseError(f"{where} is not a non-negative whole number")
    return int(value)


def parse_answers(
    payload: Any, questions: Mapping[str, Mapping[str, Any]]
) -> tuple[str, dict[str, Answer], Usage]:
    """Validate a reply against the questions that were asked. Raises JevResponseError.

    Strict where a decision depends on it: an answer of the wrong type, a
    missing question, or a probability keyed to an option nobody asked about
    means our reading of the contract is wrong, and a decision built on it
    would be a guess. Lenient where the spec is: `model` may be empty (no
    minimum length is set), and a missing `output_tokens` counts as 0, since
    output is free.
    """
    if not isinstance(payload, dict):
        raise JevResponseError("reply is not a JSON object")
    model = payload.get("model")
    answers = payload.get("answers")
    usage = payload.get("usage")
    if not isinstance(model, str):
        raise JevResponseError("reply has no model name")
    if not isinstance(answers, dict):
        raise JevResponseError("reply has no answers object")
    if not isinstance(usage, dict):
        raise JevResponseError("reply has no usage object")

    parsed: dict[str, Answer] = {}
    for name, question in questions.items():
        raw = answers.get(name)
        kind = question.get("type")
        if not isinstance(raw, dict):
            raise JevResponseError(f"no answer for question {name!r}")
        if raw.get("type") != kind:
            raise JevResponseError(f"answer to {name!r} is not of the type asked ({kind})")
        if kind == "noul":
            parsed[name] = NoulAnswer(noul=_probability(raw.get("noul"), f"{name}.noul"))
            continue
        criteria = question.get("criteria")
        if kind == "choice":
            options = list(criteria) if isinstance(criteria, dict) else []
        elif kind == "score":
            options = [str(i) for i in range(len(criteria))] if isinstance(criteria, list) else []
        else:
            raise JevResponseError(f"question {name!r} has an unsupported type")
        raw_probs = raw.get("probabilities")
        if not isinstance(raw_probs, dict) or set(raw_probs) != set(options):
            raise JevResponseError(f"{name}.probabilities do not match the options asked")
        # Safe to name the keys now: they are exactly our own option names.
        probs = {k: _probability(v, f"{name}.probabilities[{k}]") for k, v in raw_probs.items()}
        confidence = _probability(raw.get("confidence"), f"{name}.confidence")
        if kind == "choice":
            choice = raw.get("choice")
            if not isinstance(choice, str) or choice not in options:
                raise JevResponseError(f"{name}.choice is not one of the options asked")
            parsed[name] = ChoiceAnswer(choice=choice, confidence=confidence, probabilities=probs)
        else:
            score = raw.get("score")
            if (isinstance(score, bool) or not isinstance(score, (int, float))
                    or not math.isfinite(score) or not 0 <= score <= len(options) - 1):
                raise JevResponseError(f"{name}.score is not a level of the rubric asked")
            parsed[name] = ScoreAnswer(score=float(score), confidence=confidence, probabilities=probs)

    cost = usage.get("cost")
    reported = (
        float(cost)
        if isinstance(cost, (int, float)) and not isinstance(cost, bool)
        and math.isfinite(cost) and cost >= 0
        else None  # absent, or not a believable amount: fall back to the list price
    )
    return (
        model,
        parsed,
        Usage(
            input_tokens=_token_count(usage.get("input_tokens"), "usage.input_tokens"),
            output_tokens=_token_count(usage.get("output_tokens", 0), "usage.output_tokens"),
            reported_cost_usd=reported,
        ),
    )


def _error_type(status: int, body: bytes) -> str:
    """The API's machine-readable error type, or the status. Never its free text.

    A 422 body carries an `input` preview of the rejected value, and a user's
    question can be in it, so only the type and the field path are read, and
    anything unexpected in the body falls back to the bare status.
    """
    try:
        parsed = json.loads(body)
        error = parsed.get("error")  # the OpenAI-style gateway: {"error": {"type", "code"}}
        if isinstance(error, dict):
            code = error.get("code") or error.get("type")
            if isinstance(code, str):
                return f"HTTP {status} {code[:60]}"
        detail = parsed.get("detail")
        if isinstance(detail, dict) and isinstance(detail.get("error_type"), str):
            return f"HTTP {status} {detail['error_type'][:60]}"
        if isinstance(detail, list) and detail and isinstance(detail[0], dict):
            loc = detail[0].get("loc")
            where = ".".join(str(p) for p in loc)[:80] if isinstance(loc, list) else "?"
            return f"HTTP {status} {str(detail[0].get('type', ''))[:40]} at {where}"
        if isinstance(detail, str) and "max_tokens_exceeded" in detail:
            return f"HTTP {status} max_tokens_exceeded"
    except Exception:  # noqa: BLE001 (any body we cannot read is just its status)
        pass
    return f"HTTP {status}"


def _retry_after_s(headers: httpx.Headers) -> float | None:
    """Server-requested wait: `retry-after-ms` first, then `Retry-After` in seconds."""
    for name, scale in (("retry-after-ms", 1e-3), ("retry-after", 1.0)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw) * scale
        except ValueError:
            continue  # an HTTP-date form: fall back to our own backoff
        if math.isfinite(value) and value >= 0:
            return value
    return None


# --- cache and circuit breaker ----------------------------------------------


class _ResultCache:
    """Bounded, expiring, thread-safe: identical requests within the TTL cost nothing.

    Keyed by the exact request bytes, so a reordered option list is a different
    entry. Like rag/answer_cache.py this pins ONE sample: Jev is not
    deterministic in the mid range (TypeSafe's own consistency cookbook), and a
    user repeating a question should see a stable reading rather than a coin
    toss. Bounded because the keys are user input.
    """

    def __init__(self, maxsize: int, ttl_s: float, clock: Callable[[], float]) -> None:
        self._maxsize, self._ttl_s, self._clock = maxsize, ttl_s, clock
        self._items: OrderedDict[str, tuple[float, JevResult]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> JevResult | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            stored_at, result = item
            if self._clock() - stored_at > self._ttl_s:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return result

    def put(self, key: str, result: JevResult) -> None:
        if self._maxsize <= 0:
            return
        with self._lock:
            self._items[key] = (self._clock(), result)
            self._items.move_to_end(key)
            while len(self._items) > self._maxsize:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class _Breaker:
    """Fail fast while the dependency is known to be down."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self._clock = clock
        self._open_until = 0.0
        self._reason = ""
        self._failures = 0
        self._lock = threading.Lock()

    def check(self) -> None:
        with self._lock:
            if self._clock() < self._open_until:
                raise JevUnavailable(f"circuit open after {self._reason}")

    def success(self) -> None:
        with self._lock:
            self._failures = 0

    def fatal(self, reason: str) -> None:
        with self._lock:
            self._open_until = self._clock() + FATAL_COOLDOWN_S
            self._reason = reason

    def outage(self, reason: str) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= OUTAGE_FAILURES_TO_OPEN:
                self._open_until = self._clock() + OUTAGE_COOLDOWN_S
                self._reason = reason
                self._failures = 0


# --- the client -------------------------------------------------------------


def _record(op: str, model: str, *, latency_ms: float, usage: Usage | None, retries: int,
            ok: bool, cached: bool = False, detail: str = "", persist: bool = True) -> None:
    from obs.telemetry import record

    record(
        op=f"jev:{op}", model=model, latency_ms=latency_ms,
        tokens_in=usage.input_tokens if usage else 0,
        tokens_out=usage.output_tokens if usage else 0,
        retries=retries, ok=ok, cached=cached, detail=detail, persist=persist,
        cost_usd=usage.reported_cost_usd if usage is not None and not cached else None,
    )


class JevClient:
    """Typed, retrying, cached access to `POST /v1/systemone`.

    One pooled httpx client per instance (httpx's docs recommend a single
    client for connection pooling); the verify fan-out shares it across a few
    worker threads. The key lives in one private attribute and is sent only in
    the Authorization header; `repr` never shows it.
    """

    def __init__(
        self,
        api_key: str,
        *,
        provider: Provider | None = None,
        model: str | None = None,
        cache_size: int = 2048,
        cache_ttl_s: float = 3600.0,
        min_interval_s: float = 0.0,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        rng: Callable[[], float] = random.random,
    ) -> None:
        if not api_key or not api_key.strip():
            raise JevConfigError("no Jev API key is set (JEV_API_KEY or TYPESAFE_API_KEY)")
        # A key with a stray CR, LF or space (a mangled .env line) would be
        # refused by the HTTP layer with the header, key included, in its error,
        # and every request would fail the same way. Refuse it here instead,
        # with a message that does not repeat it.
        if not _KEY_CHARS.fullmatch(api_key.strip()):
            raise JevConfigError("the Jev API key holds characters an HTTP header cannot carry")
        self.__key = api_key.strip()
        self.provider = provider or PROVIDERS[JEV_BASE_URL]
        self.model = (model or os.environ.get("CRG_JEV_MODEL", "").strip()
                      or self.provider.default_model)
        self._sleep, self._clock, self._rng = sleep, clock, rng
        self._cache = _ResultCache(cache_size, cache_ttl_s, clock)
        self._breaker = _Breaker(clock)
        # Pacing for plans that cap requests per minute (the gateway's free plan
        # allows 10). Off by default: in a request path, waiting for a slot is
        # latency, so production leaves this at 0 and relies on 429 handling.
        self.min_interval_s = max(0.0, min_interval_s)
        self._next_slot = 0.0
        self._pace_lock = threading.Lock()
        self._http = httpx.Client(
            base_url=self.provider.base_url,
            headers={"Content-Type": "application/json", "Accept": "application/json",
                     "User-Agent": "climate-risk-agent"},
            # sized for the verify fan-out (rag/jev_verify.py MAX_WORKERS)
            limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        )

    def __repr__(self) -> str:
        return (f"JevClient(provider={self.provider.name!r}, model={self.model!r}, "
                "key=<redacted>)")

    def key_matches(self, api_key: str) -> bool:
        return self.__key == api_key.strip()

    def _pace(self) -> float:
        """Wait for this request's slot, if pacing is on. Returns the seconds waited."""
        if self.min_interval_s <= 0:
            return 0.0
        with self._pace_lock:
            wait = self._next_slot - self._clock()
            if wait > 0:
                self._sleep(wait)
            self._next_slot = self._clock() + self.min_interval_s
        return max(wait, 0.0)

    def close(self) -> None:
        self._http.close()

    def clear_cache(self) -> None:
        self._cache.clear()

    def _auth(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.__key}"}

    def ask(
        self,
        state: Any,
        questions: Mapping[str, Mapping[str, Any]],
        *,
        op: str,
        policy: RetryPolicy | None = None,
        use_cache: bool = True,
    ) -> JevResult:
        """Ask `questions` about `state`. Raises a JevError subclass on any failure."""
        payload = encode_request(
            self.model, state, questions,
            max_request_tokens=self.provider.max_request_tokens,
            max_question_tokens=self.provider.max_question_tokens,
        )
        key = hashlib.sha256(payload).hexdigest()
        if use_cache:
            hit = self._cache.get(key)
            if hit is not None:
                _record(op, hit.model, latency_ms=0.0, usage=hit.usage, retries=0, ok=True,
                        cached=True, persist=False)
                return replace(hit, cached=True, latency_ms=0.0, retries=0)
        try:
            self._breaker.check()
        except JevUnavailable as exc:
            _record(op, self.model, latency_ms=0.0, usage=None, retries=0, ok=False,
                    detail=exc.reason, persist=False)
            raise
        try:
            result = self._post(payload, questions, op=op, policy=policy or triage_policy())
        except JevError:
            raise
        except Exception as exc:  # noqa: BLE001 (callers are promised a JevError, nothing else)
            why = f"unexpected {type(exc).__name__}"
            _record(op, self.model, latency_ms=0.0, usage=None, retries=0, ok=False, detail=why)
            raise JevUnavailable(why) from exc
        if use_cache:
            self._cache.put(key, result)
        return result

    def _backoff(self, policy: RetryPolicy, attempt: int) -> float:
        delay = min(policy.backoff_max_s, policy.backoff_initial_s * (2 ** attempt))
        return max(0.0, delay - delay * policy.jitter * self._rng())

    def _exchange(self, payload: bytes, *, deadline: float, remaining: float,
                  policy: RetryPolicy) -> tuple[int, httpx.Headers, bytes]:
        """One attempt: send, then read the reply in chunks, stopping at the deadline.

        httpx's timeouts are per phase and the read timeout restarts on every
        chunk, so on their own they cannot hold a total budget: a reply that
        trickles in, each chunk inside the read timeout, never times out. So the
        connect and read waits are carved out of what is left of the budget, and
        the clock is checked after every chunk. An attempt can then outlast the
        budget by at most one chunk's wait, which is itself bounded by what was
        left when the attempt began.
        """
        connect_s = min(policy.connect_timeout_s, remaining / 2)
        read_s = max(min(policy.read_timeout_s, remaining - connect_s), 0.001)
        timeout = httpx.Timeout(read_s, connect=connect_s, pool=connect_s)
        with self._http.stream("POST", SYSTEMONE_PATH, content=payload,
                               headers=self._auth(), timeout=timeout) as response:
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_bytes():
                size += len(chunk)
                if size > MAX_REPLY_BYTES:
                    raise _ReplyTooLarge()
                chunks.append(chunk)
                if self._clock() >= deadline:
                    raise _OutOfTime()
            return response.status_code, response.headers, b"".join(chunks)

    def _post(self, payload: bytes, questions: Mapping[str, Mapping[str, Any]], *,
              op: str, policy: RetryPolicy) -> JevResult:
        # Time spent waiting for a paced slot is neither Jev's latency nor part
        # of the call's budget: the clock starts once the request may go.
        self._pace()
        start = self._clock()
        deadline = start + policy.budget_s
        paced = 0.0
        attempt = 0
        reason = "no attempt made"

        def elapsed_ms() -> float:
            return (self._clock() - start - paced) * 1000.0

        def give_up(why: str) -> JevUnavailable:
            _record(op, self.model, latency_ms=elapsed_ms(), usage=None, retries=attempt,
                    ok=False, detail=why)
            self._breaker.outage(why)
            return JevUnavailable(why)

        while True:
            remaining = deadline - self._clock()
            if remaining <= 0:
                raise give_up(f"{reason}; time budget of {policy.budget_s:g} s spent")
            server_wait: float | None = None
            try:
                status, headers, body = self._exchange(payload, deadline=deadline,
                                                       remaining=remaining, policy=policy)
            except _OutOfTime:
                raise give_up(f"reply still arriving when the {policy.budget_s:g} s budget "
                              "ran out") from None
            except _ReplyTooLarge:
                self._note_bad_reply(op, elapsed_ms(), attempt, "reply over 1 MiB")
                raise JevResponseError("reply over 1 MiB") from None
            except httpx.TimeoutException as exc:
                reason = f"timeout ({type(exc).__name__})"
            except httpx.TransportError as exc:
                reason = f"network error ({type(exc).__name__})"
            except httpx.HTTPError as exc:  # decoding and stream errors: not worth a retry
                raise give_up(f"http error ({type(exc).__name__})") from exc
            else:
                request_id = headers.get(self.provider.request_id_header)
                if status == 200:
                    try:
                        reply = json.loads(body)
                    except ValueError as exc:  # json.JSONDecodeError and UnicodeDecodeError
                        self._note_bad_reply(op, elapsed_ms(), attempt, "reply is not JSON")
                        raise JevResponseError("reply is not JSON") from exc
                    try:
                        model, answers, usage = parse_answers(reply, questions)
                    except JevResponseError as exc:
                        self._note_bad_reply(op, elapsed_ms(), attempt, exc.reason)
                        raise
                    model = model or self.model  # the spec allows an empty name
                    # The gateway also names the decision and the upstream it used.
                    decision_id = reply.get("id")
                    upstream = reply.get("provider")
                    latency = elapsed_ms()
                    _record(op, model, latency_ms=latency, usage=usage, retries=attempt, ok=True)
                    self._breaker.success()
                    return JevResult(
                        model=model, answers=answers, usage=usage, latency_ms=latency,
                        retries=attempt,
                        request_id=request_id or (str(decision_id)[:80] if decision_id else None),
                        upstream=str(upstream)[:40] if isinstance(upstream, str) else None,
                    )
                why = _error_type(status, body)
                if status in FATAL_STATUSES:
                    _record(op, self.model, latency_ms=elapsed_ms(), usage=None,
                            retries=attempt, ok=False, detail=why)
                    self._breaker.fatal(why)
                    _warn_once(f"Jev refused the API key or account ({why}); calls pause for "
                               f"{FATAL_COOLDOWN_S:.0f} s at a time and every caller falls back")
                    raise JevRequestError(why, status=status)
                if status not in RETRYABLE_STATUSES:
                    _record(op, self.model, latency_ms=elapsed_ms(), usage=None,
                            retries=attempt, ok=False, detail=why)
                    raise JevRequestError(why, status=status)
                reason = why
                server_wait = _retry_after_s(headers)
                if status == 429 and server_wait is None and self.min_interval_s > 0:
                    # A per-minute cap with no Retry-After: a sub-second backoff
                    # would only spend another request of the same minute.
                    server_wait = self.min_interval_s

            if attempt >= policy.max_retries:
                raise give_up(f"{reason} after {attempt + 1} attempts")
            delay = server_wait if server_wait is not None else self._backoff(policy, attempt)
            if self._clock() + delay >= deadline:
                raise give_up(f"{reason}; the next retry would overrun the "
                              f"{policy.budget_s:g} s budget")
            self._sleep(delay)
            waited = self._pace()  # a retry is a request too, for a paced plan
            paced += waited
            deadline += waited
            attempt += 1

    def _note_bad_reply(self, op: str, latency_ms: float, attempt: int, reason: str) -> None:
        """A malformed 200 is a contract problem: recorded, not retried, circuit untouched."""
        _record(op, self.model, latency_ms=latency_ms, usage=None, retries=attempt, ok=False,
                detail=f"bad reply: {reason}")

    def list_models(self) -> list[dict[str, Any]]:
        """`GET /v1/models`: the names this account may send. Raises JevError only."""
        try:
            response = self._http.get(MODELS_PATH, headers=self._auth(),
                                      timeout=httpx.Timeout(5.0, connect=2.0))
        except httpx.HTTPError as exc:
            raise JevUnavailable(f"network error ({type(exc).__name__})") from exc
        if response.status_code != 200:
            raise JevRequestError(_error_type(response.status_code, response.content),
                                  status=response.status_code)
        try:
            payload = response.json()
            # TypeSafe sends {"models": [{"name": ...}]}; the OpenAI-style
            # gateway sends {"object": "list", "data": [{"id": ...}]}.
            models = payload.get("models", payload.get("data"))
        except (ValueError, AttributeError) as exc:
            raise JevResponseError("models reply is not a JSON object") from exc
        if not isinstance(models, list):
            raise JevResponseError("models reply has no models list")
        return [{**m, "name": m.get("name") or m.get("id")} for m in models if isinstance(m, dict)]


# --- the process-wide client ------------------------------------------------

_client: JevClient | None = None
_client_lock = threading.Lock()


def get_client() -> JevClient:
    """The shared client for the configured provider. Raises JevConfigError.

    The key is JEV_API_KEY, else TYPESAFE_API_KEY; the provider is the vetted
    one JEV_BASE_URL names (default TypeSafe direct); CRG_JEV_MIN_INTERVAL_S
    paces requests (default 0, off). Rebuilt whenever any of the three changes,
    so a rotated key or a switched provider needs no restart. The circuit and
    cache belong to the instance and go with it. The old instance is dropped,
    not closed: requests still in flight on it (the verify fan-out) finish
    normally rather than failing on a closed pool.
    """
    global _client
    key = (os.environ.get("JEV_API_KEY", "").strip()
           or os.environ.get("TYPESAFE_API_KEY", "").strip())
    if not key:
        _warn_once("CRG_JEV is on but neither JEV_API_KEY nor TYPESAFE_API_KEY is set: "
                   "Jev calls are skipped and every caller keeps its non-Jev behaviour")
        raise JevConfigError("no Jev API key is set (JEV_API_KEY or TYPESAFE_API_KEY)")
    provider = get_provider()
    pace = _env_seconds("CRG_JEV_MIN_INTERVAL_S", 0.0) if os.environ.get(
        "CRG_JEV_MIN_INTERVAL_S", "").strip() else 0.0
    with _client_lock:
        if (_client is None or not _client.key_matches(key)
                or _client.provider != provider or _client.min_interval_s != pace):
            _client = JevClient(key, provider=provider, min_interval_s=pace)
        return _client


def reset_client() -> None:
    """Drop the shared client, its cache and its circuit (tests; key changes)."""
    global _client
    with _client_lock:
        if _client is not None:
            _client.close()
        _client = None
    with _warned_lock:
        _warned.clear()
