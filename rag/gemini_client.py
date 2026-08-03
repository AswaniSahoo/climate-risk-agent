"""The single seam between this project and the Gemini API (google-genai SDK).

Auth is env-driven, explicit, and fails loudly:
- GOOGLE_GENAI_USE_VERTEXAI=true  -> Vertex AI (ADC via `gcloud auth application-default
  login`), project = GOOGLE_CLOUD_PROJECT, location = GOOGLE_CLOUD_LOCATION ("global"
  default). This is the paid path.
- else GEMINI_API_KEY             -> AI Studio key (free tier works; conservative
  pacing defaults elsewhere assume this).
- else                            -> typed error, never a silent fallback.

Everything above this seam (caching, batching, schema validation, scope guard)
is SDK-agnostic; tests monkeypatch these two functions instead of mocking HTTP.
"""
from __future__ import annotations

import os
import threading
import time

# Model IDs are env-overridable so a deploy can pin/upgrade without a code change
# (a real production knob); the defaults are the deliberately-chosen GA models.
EMBED_MODEL = os.environ.get("CRG_EMBED_MODEL", "gemini-embedding-2")
GENERATE_MODEL = os.environ.get("CRG_GENERATE_MODEL", "gemini-2.5-flash")

_RETRY_MAX = 6
_sleep = time.sleep  # module-level so tests can stub the waiting out

_local = threading.local()  # one SDK client per thread (shared clients get closed
# under concurrent use — measured: "Cannot send a request, as the client has been closed")


_auth_lock = threading.Lock()
_vertex_creds = None  # resolved ONCE per process, shared by every thread's client


def _vertex_credentials():
    """Resolve ADC once per process, WITHOUT letting google-auth shell out.

    google.auth.default() discovers the project by running the `gcloud` CLI
    whenever the ADC file carries no project of its own. That child process
    never returns once an MCP stdio server owns stdio, so the tool call hangs
    forever (measured: search_ipcc stuck >300s in _run_subprocess_ignore_stderr,
    before any HTTP request, so the 120s client timeout never fires). Reading
    the ADC file directly yields the same credentials with no subprocess at all;
    the project comes from GOOGLE_CLOUD_PROJECT, which we hand to the client
    anyway. Shared across threads so per-thread clients never re-resolve.
    """
    global _vertex_creds
    with _auth_lock:
        if _vertex_creds is None:
            import google.auth

            scopes = ["https://www.googleapis.com/auth/cloud-platform"]
            adc_file = _adc_file()
            if adc_file:
                _vertex_creds, _ = google.auth.load_credentials_from_file(adc_file, scopes=scopes)
            else:
                _vertex_creds, _ = google.auth.default(scopes=scopes)
    return _vertex_creds


def _adc_file() -> str | None:
    """Path to the Application Default Credentials JSON, or None if unreadable."""
    explicit = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if explicit and os.path.isfile(explicit):
        return explicit
    try:
        from google.auth import _cloud_sdk

        path = _cloud_sdk.get_application_default_credentials_path()
    except Exception:  # private helper moved/renamed -> fall back to default()
        return None
    return path if os.path.isfile(path) else None


class GeminiError(RuntimeError):
    """Raised when a Gemini call fails (auth, quota exhausted after retries, API)."""


def _new_client():
    from google import genai
    from google.genai import types

    # Explicit timeout: a hung call must FAIL LOUDLY, never leave the user
    # staring at a stuck progress bar wondering (observable-CLI rule).
    http_options = types.HttpOptions(timeout=120_000)  # ms

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise GeminiError("GOOGLE_GENAI_USE_VERTEXAI is set but GOOGLE_CLOUD_PROJECT is not")
        return genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
            # Explicit, or the SDK resolves ADC per client -> gcloud subprocess
            # per thread -> hangs forever under an MCP stdio server.
            credentials=_vertex_credentials(),
            http_options=http_options,
        )
    if os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=http_options)
    raise GeminiError(
        "no Gemini auth configured: set GOOGLE_GENAI_USE_VERTEXAI=true (+ project) or GEMINI_API_KEY"
    )


def _client():
    client = getattr(_local, "client", None)
    if client is None:
        client = _local.client = _new_client()
    return client


def _reset_clients() -> None:
    """Drop this thread's cached client and the shared credentials (tests; env changes)."""
    global _vertex_creds
    _local.__dict__.clear()
    with _auth_lock:
        _vertex_creds = None


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def _with_retry(call, *, op: str, model: str, extract_tokens=None):
    """Run `call` with rate-limit retries AND record telemetry at this chokepoint.

    Every Gemini call in the project passes through here, so latency, retries,
    token counts, and failures are measured structurally — no call can opt out.
    `extract_tokens(result) -> (tokens_in, tokens_out)`; embeds pass estimates.
    """
    from obs.telemetry import record

    start = time.perf_counter()
    for attempt in range(_RETRY_MAX):
        try:
            result = call()
        except Exception as exc:  # SDK raises its own hierarchy; classify, don't guess
            if _is_rate_limit(exc) and attempt < _RETRY_MAX - 1:
                _sleep(min(30.0 * (attempt + 1), 120.0))
                continue
            record(op=op, model=model, latency_ms=(time.perf_counter() - start) * 1000,
                   tokens_in=0, tokens_out=0, retries=attempt, ok=False)
            raise GeminiError(f"Gemini call failed: {exc}") from exc
        tokens_in, tokens_out = extract_tokens(result) if extract_tokens else (0, 0)
        record(op=op, model=model, latency_ms=(time.perf_counter() - start) * 1000,
               tokens_in=tokens_in, tokens_out=tokens_out, retries=attempt, ok=True)
        return result
    raise GeminiError("rate-limit retries exhausted")  # pragma: no cover


_EMBED_WORKERS = 8


def embed_batch(texts: list[str], *, task_type: str, dims: int) -> list[list[float]]:
    """Embed texts, ONE content per API call (the SDK enforces this on Vertex:
    "embedContent ... only supports one content at a time"; a plain string list
    gets silently JOINED into one content — measured, and it poisoned a cache).
    Per-text calls with retry, threaded for speed, order preserved.

    task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY.
    """
    from concurrent.futures import ThreadPoolExecutor

    from google.genai import types

    config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=dims)

    def embed_one(text: str) -> list[float]:
        def call():
            response = _client().models.embed_content(
                model=EMBED_MODEL, contents=text, config=config
            )
            if not response.embeddings:
                raise GeminiError("API returned no embedding for a text")
            return list(response.embeddings[0].values)

        # embed responses carry no usage metadata -> chars/4 token ESTIMATE
        return _with_retry(
            call, op="embed", model=EMBED_MODEL,
            extract_tokens=lambda _result: (len(text) // 4, 0),
        )

    with ThreadPoolExecutor(max_workers=_EMBED_WORKERS) as pool:
        return list(pool.map(embed_one, texts))


def embedding_available(*, dims: int = 768) -> tuple[bool, str]:
    """Probe the embedding endpoint with a 1-token query at startup.

    The whole point of this project's "loud fallback" rule: a wrong embedding
    region/model (e.g. requesting gemini-embedding-2 from a single region that
    only serves it on global/us/eu) must FAIL LOUD here, not hide behind a
    citation-less report while every live query silently 404s to BM25-only.

    Returns (ok, human-readable detail); never raises.
    """
    try:
        embed_batch(["healthcheck"], task_type="RETRIEVAL_QUERY", dims=dims)
    except Exception as exc:  # noqa: BLE001 — a probe must classify, never crash the app
        return False, f"dense embeddings unavailable [{EMBED_MODEL}]: {exc}"
    return True, f"dense OK [{EMBED_MODEL}]"


def generate_json(prompt: str, *, schema: dict) -> str:
    """One generation call, forced to the JSON response schema, temperature 0."""
    from google.genai import types

    def call():
        return _client().models.generate_content(
            model=GENERATE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )

    def usage(response) -> tuple[int, int]:
        """(input, output) tokens. Reasoning models report the tokens they spent
        THINKING separately from the visible answer, and Google bills them as
        output — counting only `candidates_token_count` undercounted a measured
        3.6-flash call ~20x (50 visible vs 987 thinking). Cost telemetry must
        charge for what is actually billed, so thoughts are added to output."""
        meta = getattr(response, "usage_metadata", None)
        visible_out = getattr(meta, "candidates_token_count", 0) or 0
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        return (
            getattr(meta, "prompt_token_count", 0) or 0,
            visible_out + thoughts,
        )

    response = _with_retry(call, op="generate", model=GENERATE_MODEL, extract_tokens=usage)
    return response.text
