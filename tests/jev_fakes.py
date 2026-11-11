"""Shared fakes for the Jev tests: replies shaped like the documented API, and a clock.

Replies follow the `SystemOneResponse` schema in https://api.typesafe.ai/openapi.json
(read 2026-11-11): `{model, answers, usage}`, one answer per question name, a Choice
carrying `choice`, `confidence` and a probability for every option. Nothing here
talks to the network; pytest-httpx answers every request.
"""
from __future__ import annotations

FAKE_KEY = "fake-typesafe-key-for-tests-0000000000"
JEV_URL = "https://api.typesafe.ai/v1/systemone"

TOPICS = ("assessed_hazard", "climate_science", "other_hazard", "policy_or_economics", "unrelated")
HAZARDS = ("heatwave", "extreme_precip", "wind")
OTHER_TOPICS = (
    "drought", "tropical_cyclone", "coastal_flooding", "wildfire", "marine_heatwave",
    "convective_storm", "landslide", "earthquake", "air_quality", "policy", "unrelated",
)


def noul(p: float) -> dict:
    return {"type": "noul", "noul": p}


def choice(probabilities: dict[str, float], options: tuple[str, ...] | None = None) -> dict:
    """A Choice answer; options missing from `probabilities` get 0 so the keys match."""
    probs = {name: 0.0 for name in (options or tuple(probabilities))}
    probs.update(probabilities)
    top = max(probs, key=lambda k: probs[k])
    n = len(probs)
    confidence = (probs[top] - 1 / n) / (1 - 1 / n) if n > 1 else 1.0
    return {"type": "choice", "choice": top, "confidence": round(max(confidence, 0.0), 4),
            "probabilities": probs}


def reply(answers: dict, *, model: str = "jev-1.14.0", input_tokens: int = 420,
          output_tokens: int = 30) -> dict:
    return {"model": model, "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def triage_reply(*, topic: dict[str, float], about: float, hazard: dict[str, float],
                 other: dict[str, float] | None = None, **kw) -> dict:
    """A reply to rag.jev_triage.build_questions(), with every option filled in."""
    return reply({
        "topic": choice(topic, TOPICS),
        "about_assessed_hazard": noul(about),
        "hazard": choice(hazard, HAZARDS),
        "other_topic": choice(other or {"unrelated": 1.0}, OTHER_TOPICS),
    }, **kw)


def relation_reply(supports: float, contradicts: float = 0.0, **kw) -> dict:
    """A reply to rag.jev_verify.RELATION_QUESTION."""
    says_nothing = max(0.0, round(1.0 - supports - contradicts, 4))
    return reply({"relation": choice(
        {"supports": supports, "contradicts": contradicts, "says_nothing": says_nothing}
    )}, **kw)


class FakeClock:
    """A monotonic clock that only moves when something sleeps on it."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def enable_jev(monkeypatch, mode: str = "triage") -> None:
    monkeypatch.setenv("CRG_JEV", mode)
    monkeypatch.setenv("TYPESAFE_API_KEY", FAKE_KEY)


GATEWAY = "https://api.experientiallabs.ai"
GATEWAY_URL = f"{GATEWAY}/v1/systemone"


def gateway_reply(answers: dict, *, cost: float | None = 0.0, input_tokens: int = 1011) -> dict:
    """A reply as the Experiential Labs gateway sends it (probed 2026-11-11): the
    TypeSafe shape plus a decision `id`, `usage.cost`, `usage.is_byok` and the
    upstream `provider`, with the alias, not a version, as the model."""
    usage: dict = {"input_tokens": input_tokens, "output_tokens": 252, "is_byok": False}
    if cost is not None:
        usage["cost"] = cost
    return {"id": "decision_0123", "model": "jev-latest", "answers": answers,
            "usage": usage, "provider": "typesafe"}


def use_gateway(monkeypatch, mode: str = "triage") -> None:
    monkeypatch.setenv("CRG_JEV", mode)
    monkeypatch.setenv("JEV_BASE_URL", GATEWAY)
    monkeypatch.setenv("JEV_API_KEY", FAKE_KEY)
