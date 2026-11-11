"""Instant question triage with Jev: is it about heat, rain or wind, and if not, what.

Where this sits, which is the safety argument:

1. Stage 1 (`rag.scope`, lexical) runs first and its verdict is final. Jev is
   never asked about a question stage 1 has already refused, and it can never
   un-refuse one.
2. Jev reads only what stage 1 cannot see: a question with no hazard
   vocabulary at all ("How hot will it get in Delhi next week?"), plus, in the
   free-text front door, a question stage 1 ruled in scope whose hazard the
   parser cannot name ("temperature extremes"). Everything else never leaves
   this process, which is both cheaper and the privacy-minimal choice.
3. Jev's answer is a SIGNAL. The decision is made here, in code, by fixed
   thresholds, and on any doubt the answer is "defer": behaviour is then
   exactly what it is with Jev off. Any failure is also "defer" in effect
   (decision `unavailable`), reported in telemetry by the client.

One request, four questions (TypeSafe's "speculative fan-out" pattern: every
question is answered in parallel against the same state, so asking what we
might need costs input tokens, not latency):

- `topic` (Choice): which of five kinds of question this is.
- `about_assessed_hazard` (Noul): the companion yes/no for "is it about one of
  our three hazards at all", because a Choice always picks something ("A
  Choice question always picks an option, even when none of them applies",
  docs "patterns/companion-noul").
- `hazard` (Choice): heatwave, extreme precipitation or wind.
- `other_topic` (Choice): what to name in a refusal.

The user's text is untrusted input to Jev. TypeSafe says so plainly: content
written to steer the model "can move the answer" (docs "model-jaggedness").
That is why the worst a manipulated answer can do here is bounded: refuse the
manipulator's own question, or route a question that has no hazard words to a
report for one of our three hazards, which the report then names. It cannot
bypass stage 1, reach the answering prompt, or touch a citation.

Thresholds are fixed starting points from TypeSafe's docs (0.7 for yes and 0.3
for no, docs "primitives/noul"; 0.8 before acting without a human, the
citation-check cookbook), set before any measurement, and env-overridable like
stage 2's.
"""
from __future__ import annotations

import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from rag.jev_client import JevClient, JevError, JevResult, get_client, triage_policy

_log = logging.getLogger(__name__)

# The same budget rag/prompt_safety.py gives a question in a Gemini prompt.
QUESTION_CHARS = 1000

def env_probability(name: str, default: float) -> float:
    """A threshold from the environment, or `default`, loudly, if it is not one.

    Read at import, and this module is imported lazily inside request paths
    (rag/scope.py, agent/nl.py), so a typo such as "0,8" must not raise: that
    would turn a misconfigured knob into a 500 on every question it touches.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 1.0:  # also false for NaN
        _log.warning("%s=%r is not a probability in [0, 1]; using %s", name, raw, default)
        return default
    return value


REFUSE_P = env_probability("CRG_JEV_REFUSE_P", 0.8)
HINT_P = env_probability("CRG_JEV_HINT_P", 0.7)
NOUL_YES = env_probability("CRG_JEV_NOUL_YES", 0.7)
NOUL_NO = env_probability("CRG_JEV_NOUL_NO", 0.3)

# Option order matters: Jev "leans toward the option that comes first" (docs
# "model-jaggedness"), so the in-scope options come first and the lean, if any,
# is toward deferring rather than refusing. evals/run_jev_eval.py --order-check
# measures how often reversing this order flips a decision.
TOPIC_OPTIONS: dict[str, str] = {
    "assessed_hazard": (
        "Extreme heat, heavy rain or strong wind, in any wording: heatwaves, hot days, "
        "how hot it will get, downpours, cloudbursts, rainfall extremes, gales, gusts, "
        "or storms getting windier"
    ),
    "climate_science": (
        "General climate science with no hazard named: greenhouse gas levels, global "
        "warming, climate sensitivity, glaciers, sea ice, ocean circulation, or average "
        "rainfall and monsoons"
    ),
    "other_hazard": (
        "A hazard other than heat, rain and wind: drought, tropical cyclones, hurricanes, "
        "sea level rise, coastal flooding, wildfire or smoke, marine heatwaves, hail, "
        "tornadoes, landslides, avalanches, earthquakes, or air pollution"
    ),
    "policy_or_economics": (
        "Climate policy or money: emissions pledges and targets, climate negotiations, "
        "adaptation costs, insurance, or economic losses"
    ),
    "unrelated": "Something unrelated to weather and climate",
}
REFUSAL_TOPICS = ("other_hazard", "policy_or_economics", "unrelated")

HAZARD_OPTIONS: dict[str, str] = {
    "heatwave": "Heat: hot days, heatwaves, high temperatures",
    "extreme_precip": "Rain: heavy rainfall, downpours, cloudbursts, extreme precipitation",
    "wind": "Wind: strong winds, gusts, gales, windstorms",
}

# Option -> the name a refusal reports. The names reuse the vocabulary stage 1
# and stage 2 already refuse with, so one hazard is never called two things.
OTHER_TOPIC_OPTIONS: dict[str, tuple[str, str]] = {
    "drought": ("Drought, dry spells or aridity", "drought"),
    "tropical_cyclone": ("Tropical cyclones, hurricanes or typhoons", "tropical cyclone"),
    "coastal_flooding": (
        "Sea level rise, storm surge or coastal flooding", "coastal flooding / sea level"
    ),
    "wildfire": ("Wildfires, fire weather or wildfire smoke", "wildfire / fire weather"),
    "marine_heatwave": ("Marine or ocean heatwaves", "marine heatwave"),
    "convective_storm": ("Hail or tornadoes", "severe convective storm"),
    "landslide": ("Landslides or avalanches", "landslide / avalanche"),
    "earthquake": ("Earthquakes or tremors", "earthquake / seismic hazard"),
    "air_quality": ("Air pollution or air quality", "air quality"),
    "policy": ("Climate policy, pledges, costs or economic losses", "climate policy and economics"),
    "unrelated": ("Something unrelated to weather and climate", "unrelated to climate risk"),
}
_GENERIC_REFUSAL_NAME = {
    "other_hazard": "an unsupported hazard",
    "policy_or_economics": "climate policy and economics",
    "unrelated": "unrelated to climate risk",
}

# Jev's hazard option -> the policy name stage 2 uses for a hazard hint
# (rag.scope_semantic.SUPPORTED_HAZARDS), which agent/nl.py maps to the enum.
HAZARD_HINT_NAME = {
    "heatwave": "heatwave",
    "extreme_precip": "extreme precipitation",
    "wind": "wind",
}


def build_questions(*, topic_order: tuple[str, ...] | None = None) -> dict[str, dict[str, Any]]:
    """The four questions, as data. `topic_order` reorders `topic` (the order check)."""
    order = topic_order or tuple(TOPIC_OPTIONS)
    if sorted(order) != sorted(TOPIC_OPTIONS):
        raise ValueError(f"topic_order must list each topic once: {order}")
    return {
        "topic": {
            "type": "choice",
            "instructions": "What is `question` asking about?",
            "criteria": {name: TOPIC_OPTIONS[name] for name in order},
        },
        "about_assessed_hazard": {
            "type": "noul",
            "instructions": "`question` asks about extreme heat, heavy rain, or strong wind.",
            "criteria": {
                "true": "It asks about hot weather, heatwaves, heavy rain, downpours, strong "
                        "winds, gusts or gales, in any wording.",
                "false": "It asks about another hazard such as drought, cyclones, sea level, "
                         "wildfire or earthquakes, or about general climate science, policy, "
                         "money, or something else.",
            },
        },
        "hazard": {
            "type": "choice",
            "instructions": "Which hazard does `question` ask about?",
            "criteria": dict(HAZARD_OPTIONS),
        },
        "other_topic": {
            "type": "choice",
            "instructions": "Which hazard or topic does `question` ask about?",
            "criteria": {name: text for name, (text, _) in OTHER_TOPIC_OPTIONS.items()},
        },
    }


def clean_question(question: str) -> str:
    """Collapse whitespace and cap the length. The text is sent as data, never as a prompt."""
    return re.sub(r"\s+", " ", question or "").strip()[:QUESTION_CHARS]


Decision = Literal["refuse", "hint", "defer", "unavailable"]


@dataclass(frozen=True)
class JevTriage:
    """What Jev's reading of one question decided, and the numbers it decided on.

    `decision`: `refuse` (with `out_of_scope` naming the topic), `hint` (with
    `hazard_hint` naming one of our three hazards), `defer` (not confident
    enough to act: behaviour stays as with Jev off), or `unavailable` (Jev
    could not be reached or answered badly: the same fallback, plus a reason).
    """

    decision: Decision
    out_of_scope: str | None = None
    hazard_hint: str | None = None
    hazard: str | None = None  # Jev's top hazard option, whatever was decided
    hazard_probability: float | None = None
    about_assessed_hazard: float | None = None
    topic: str | None = None
    topic_probabilities: dict[str, float] = field(default_factory=dict)
    refuse_probability: float | None = None
    model: str | None = None
    latency_ms: float = 0.0
    input_tokens: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    status: str = "ok"

    @property
    def detail(self) -> str:
        """One compact line for logs and ScopeDecision.detail."""
        if self.decision == "unavailable":
            return f"jev {self.status}"
        return (
            f"jev {self.decision} topic={self.topic}:{self.topic_probabilities.get(self.topic or '', 0):.2f} "
            f"refuse={self.refuse_probability or 0:.2f} noul={self.about_assessed_hazard or 0:.2f} "
            f"hazard={self.hazard}:{self.hazard_probability or 0:.2f} [{self.model}]"
        )


def decide(result: JevResult, *, may_refuse: bool = True) -> JevTriage:
    """Fixed thresholds over Jev's answers -> refuse, hint or defer. Pure; no I/O.

    `may_refuse=False` is the front door's second use: stage 1 has already ruled
    the question in scope by its vocabulary, so only the hazard is open, and the
    topic question has nothing left to decide.
    """
    topic = result.choice("topic")
    hazard = result.choice("hazard")
    other = result.choice("other_topic")
    noul = result.noul("about_assessed_hazard")
    p_refuse = sum(topic.p(name) for name in REFUSAL_TOPICS)

    out_of_scope = hint = None
    if may_refuse and p_refuse >= REFUSE_P and noul <= NOUL_NO:
        decision: Decision = "refuse"
        refused_topic = max(REFUSAL_TOPICS, key=topic.p)
        out_of_scope = (
            OTHER_TOPIC_OPTIONS[other.choice][1]
            if other.top_p >= 0.5
            else _GENERIC_REFUSAL_NAME[refused_topic]
        )
    elif (
        noul >= NOUL_YES
        and hazard.top_p >= HINT_P
        and (not may_refuse or topic.p("assessed_hazard") >= HINT_P)
    ):
        decision = "hint"
        hint = HAZARD_HINT_NAME[hazard.choice]
    else:
        decision = "defer"

    return JevTriage(
        decision=decision,
        out_of_scope=out_of_scope,
        hazard_hint=hint,
        hazard=hazard.choice,
        hazard_probability=round(hazard.top_p, 4),
        about_assessed_hazard=round(noul, 4),
        topic=topic.choice,
        topic_probabilities={k: round(v, 4) for k, v in topic.probabilities.items()},
        refuse_probability=round(p_refuse, 4),
        model=result.model,
        latency_ms=round(result.latency_ms, 1),
        input_tokens=result.usage.input_tokens,
        cost_usd=result.cost_usd,
        cached=result.cached,
        status="cached" if result.cached else "ok",
    )


def triage(
    question: str,
    *,
    may_refuse: bool = True,
    client: JevClient | None = None,
    topic_order: tuple[str, ...] | None = None,
    use_cache: bool = True,
) -> JevTriage:
    """One Jev request for one question. Never raises: a failure is `unavailable`."""
    text = clean_question(question)
    if not text:
        return JevTriage(decision="defer", status="skipped: empty question")
    try:
        jev = client or get_client()
        result = jev.ask(
            {"question": text},
            build_questions(topic_order=topic_order),
            op="triage",
            policy=triage_policy(),
            use_cache=use_cache,
        )
        return decide(result, may_refuse=may_refuse)
    except JevError as exc:
        _log.warning("Jev triage unavailable (%s); the lexical scope behaviour applies", exc.reason)
        return JevTriage(decision="unavailable", status=f"unavailable: {exc.reason}")
    except Exception as exc:  # noqa: BLE001 (a signal must degrade, never take a request down)
        _log.exception("Jev triage failed unexpectedly; the lexical scope behaviour applies")
        return JevTriage(decision="unavailable", status=f"unavailable: {type(exc).__name__}")
