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
import time
from functools import lru_cache

EMBED_MODEL = "gemini-embedding-2"
GENERATE_MODEL = "gemini-2.5-flash"

_RETRY_MAX = 6
_sleep = time.sleep  # module-level so tests can stub the waiting out


class GeminiError(RuntimeError):
    """Raised when a Gemini call fails (auth, quota exhausted after retries, API)."""


@lru_cache(maxsize=1)
def _client():
    from google import genai

    if os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "").lower() == "true":
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        if not project:
            raise GeminiError("GOOGLE_GENAI_USE_VERTEXAI is set but GOOGLE_CLOUD_PROJECT is not")
        return genai.Client(
            vertexai=True,
            project=project,
            location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
        )
    if os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    raise GeminiError(
        "no Gemini auth configured: set GOOGLE_GENAI_USE_VERTEXAI=true (+ project) or GEMINI_API_KEY"
    )


def _is_rate_limit(exc: Exception) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def _with_retry(call):
    for attempt in range(_RETRY_MAX):
        try:
            return call()
        except Exception as exc:  # SDK raises its own hierarchy; classify, don't guess
            if _is_rate_limit(exc) and attempt < _RETRY_MAX - 1:
                _sleep(min(30.0 * (attempt + 1), 120.0))
                continue
            raise GeminiError(f"Gemini call failed: {exc}") from exc
    raise GeminiError("rate-limit retries exhausted")  # pragma: no cover


def embed_batch(texts: list[str], *, task_type: str, dims: int) -> list[list[float]]:
    """Embed up to ~100 texts. task_type: RETRIEVAL_DOCUMENT | RETRIEVAL_QUERY."""
    from google.genai import types

    def call():
        response = _client().models.embed_content(
            model=EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type, output_dimensionality=dims),
        )
        return [list(e.values) for e in response.embeddings]

    return _with_retry(call)


def generate_json(prompt: str, *, schema: dict) -> str:
    """One generation call, forced to the JSON response schema, temperature 0."""
    from google.genai import types

    def call():
        response = _client().models.generate_content(
            model=GENERATE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=schema,
                temperature=0.0,
            ),
        )
        return response.text

    return _with_retry(call)
