"""Security Tier 1 invariants, pinned as tests.

Each test is a control from the threat model: if a refactor weakens it, the
suite goes red — security by regression pin, not by policy document.
"""
import json
import re
import subprocess

import pytest

from agent.contracts import Hazard


# --- SSRF: outbound targets are hardcoded constants, never caller-supplied ---

def test_outbound_targets_are_pinned():
    from rag.gemini_client import EMBED_MODEL, GENERATE_MODEL
    from tools.climatology import ARCHIVE_URL
    from tools.forecast import OPEN_METEO_URL

    assert OPEN_METEO_URL.startswith("https://api.open-meteo.com/")
    assert ARCHIVE_URL.startswith("https://archive-api.open-meteo.com/")
    # Gemini goes through the official SDK (Google-controlled endpoints); what we
    # pin is the MODEL choice — stable GA versions, upgraded deliberately.
    assert EMBED_MODEL == "gemini-embedding-2"
    assert GENERATE_MODEL == "gemini-2.5-flash"


def test_gemini_auth_is_explicit_never_a_silent_fallback(monkeypatch):
    import rag.gemini_client as gc

    gc._client.cache_clear()
    for var in ("GOOGLE_GENAI_USE_VERTEXAI", "GOOGLE_CLOUD_PROJECT", "GEMINI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(gc.GeminiError):
        gc._client()
    gc._client.cache_clear()


# --- input validation: coordinates are range-checked at the tool boundary ---

@pytest.mark.parametrize("lat,lon", [(91.0, 0.0), (-91.0, 0.0), (0.0, 181.0), (0.0, -181.0)])
def test_forecast_rejects_out_of_range_coordinates(lat, lon):
    from tools.forecast import get_forecast

    with pytest.raises(ValueError, match="range"):
        get_forecast(lat, lon, horizon_days=3)


@pytest.mark.parametrize("lat,lon", [(90.5, 0.0), (0.0, -180.5)])
def test_climatology_rejects_out_of_range_coordinates(lat, lon):
    from tools.climatology import climatology_hazard_stat

    with pytest.raises(ValueError, match="range"):
        climatology_hazard_stat(lat, lon, Hazard.HEATWAVE)


def test_forecast_rejects_absurd_horizon():
    from tools.forecast import get_forecast

    with pytest.raises(ValueError, match="horizon"):
        get_forecast(0.0, 0.0, horizon_days=10_000)  # denial-of-wallet guard


# --- secrets: the API key can never appear in serialized output ---

def test_api_key_never_serializes_into_answers(monkeypatch):
    from rag.answer import CitedAnswer

    sentinel = "TEST-SENTINEL-KEY-a1b2c3"
    monkeypatch.setenv("GEMINI_API_KEY", sentinel)
    answer = CitedAnswer(
        answer="x", citations=["c#p1#0"], abstain=False, allowed_ids=["c#p1#0"]
    )
    assert sentinel not in answer.model_dump_json()
    assert sentinel not in json.dumps(CitedAnswer.model_json_schema())


# --- repo hygiene: no Google API key pattern in any tracked file ---

def test_no_api_key_committed_to_repo():
    tracked = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout.splitlines()
    key_pattern = re.compile(r"AIza[0-9A-Za-z_\-]{35}")
    offenders = []
    for path in tracked:
        try:
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        if key_pattern.search(text):
            offenders.append(path)
    assert not offenders, f"Google API key pattern found in: {offenders}"
