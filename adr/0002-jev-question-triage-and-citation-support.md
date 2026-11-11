# ADR 0002: Where Jev helps this product, and on what evidence

- **Status:** Proposed. Built behind `CRG_JEV`, default `off`, and measured
  on the dev set on 2026-11-11 through the Experiential Labs gateway (see
  "Measured" below): triage meets every quality criterion but not the latency
  one, and the citation check misses one of its criteria, so both stay off.
  The key supplied for this work is a gateway key: TypeSafe direct refused it
  with HTTP 401 four times between 07:01 and 08:16 UTC, so TypeSafe direct
  itself is still unmeasured.
- **Date:** 2026-11-11
- **Decision:** Build (a) question triage and (b) a citation support check,
  each behind the flag and each off by default. Do not build (c) an
  answerability score yet. Turn (a) or (b) on only when
  `evals/run_jev_eval.py` meets the criteria in this record, which were fixed
  before any Jev answer was seen.

## Context

Jev is TypeSafe AI's "System One" model. One `POST /v1/systemone` carries a
`state` and named typed questions (`noul` for yes or no, `choice`, `score`),
and the reply is one probability-carrying answer per question, not text. The
price is $0.042 per million input tokens with output tokens free, and the docs
say "Most calls complete in about 100 ms." The contract was read on 2026-11-11
from https://api.typesafe.ai/openapi.json (spec 0.2.0) and
https://docs.typesafe.ai/llms-full.txt; the earlier notes in
`tools/jev/RESEARCH-2026-10-31.md` (in the `lfx-prep` workspace) were checked
against both, and the parts this code relies on are unchanged.

TypeSafe documents its own weak spots, and each one shaped a decision here:
literal reading; "Jev reads dates as text, not as ordered quantities"; content
written to steer it "can move the answer"; a Choice "leans toward the option
that comes first"; and "A Choice question always picks an option, even when
none of them applies" (docs "model-jaggedness", "patterns/companion-noul").

What the code does today, measured offline on 2026-11-11 with every model off
(`uv run python -m evals.run_jev_eval --dry-run`, report in
`evals/results/jev-dev-2026-11-11.md`):

- **Stage 1 scope guard on the 60-question dev set:** 58/60 decided right at
  scope, no false refusals. The two misses, OOC-01 (India's NDC pledge) and
  OOC-03 (GDP lost at 2 degrees), carry no hazard word, so they reach
  retrieval and Gemini, which refuses them (dev e2e matrix 48/11/1/0,
  2026-09-02). 16 of the 60 questions are in the bucket stage 1 cannot read at
  all: no supported and no unsupported hazard term.
- **On the 14 paraphrase probes** written for stage 2 in 2026-09: 7/14. All
  seven out-of-scope paraphrases ("Is the smoke season getting longer where I
  live?") pass the guard.
- **The free-text front door names the right hazard** on 39/42 dev questions
  that have one, and on 1/7 in-scope paraphrases. A missed hazard is a refusal
  today: "How hot will it get in Delhi next week?" is answered with "could not
  identify a supported hazard in the question". The misses on the dev set are
  SP-03 and MP-10 ("temperature extremes", which stage 1 accepts but the front
  door's patterns cannot map to a hazard) and RT-10 ("hot-extreme (TXx/TNn)").
- **Citations are validated at page level only.** Dev citation validity is
  41/48 and held-out 96% (README); LIMITATIONS.md says claim-level entailment
  "is not yet automated".

## Options evaluated

**(a) Instant question triage.** Jev reads a question before the pipeline and
answers: is it about one of our three hazards, background climate science,
another hazard, policy or money, or something unrelated; which of our hazards;
and what to name in a refusal. Expected user value, ranked:

1. Fewer wrong refusals at the front door. A paraphrase the regex misses
   becomes a report instead of "could not identify a supported hazard". This is
   the failure a person meets first.
2. Refusals that name the topic ("climate policy and economics is not a
   supported hazard") instead of a generic one, and off-topic questions on the
   IPCC path (MCP `answer_ipcc`, `answer_with_guard`) refused before retrieval
   and Gemini rather than after them.
3. Pre-filled controls. A `/triage` call returns hazard, horizon and place in
   one round trip, which is what the sidebar needs to follow a typed question.

Two corrections to the brief, from reading the code: refusals for known
out-of-scope words are already instant on the front door (it refuses before
geocoding), so the latency win is on the IPCC path only; and the sidebar bug
needs the parse wired to the widgets, which the lexical parser already supplies
for 39 of 42 dev questions. Jev widens that to paraphrases; it is not what
fixes the bug.

Rejected parts of (a): a **horizon bucket** (dates and numbers are a documented
weakness, the dev set has no horizon labels to measure against, and the regex
handles "next N days"); an **injection class** (no labelled injection items to
measure on, TypeSafe's terms bar probing their system, and TypeSafe itself says
adversarial content moves the answer, so it could never be a security layer).
On the front door path the user's text never reaches Gemini anyway: the IPCC
question is built from the hazard enum and the geocoded place label.

**(b) Citation support check.** For each sentence of the report's IPCC
statement and each cited passage, Jev answers TypeSafe's own citation-check
question (supports, contradicts, says nothing; docs
"cookbooks/citation_check", numbers there from jev-1.14.0 on 2026-11-03). The
reader sees an evidence strength per citation and the sentences no cited page
supports. User value: it is the first check of whether a real, retrieved page
says what the sentence claims, the gap LIMITATIONS.md names. It is second here
because its in-product quality can only be judged on real Gemini answers, and
no Gemini call may run under the current cost rule.

**(c) An answerability or refusal-confidence score.** Rejected for now. The dev
set has three unanswerable items, which can neither set nor check a threshold,
and the answering model already abstains with grounding. A gate before Gemini
would trade a grounded refusal for an ungrounded one.

## Decision

Build (a) and (b), behind `CRG_JEV=off|triage|verify|all`, default `off`.

### The safety argument for triage (a)

1. Stage 1 runs first and its verdict is final, flag on or off.
2. Jev reads only what stage 1 cannot see: the silent bucket in the guard
   (`rag/scope.py`, 16 of 60 dev questions), and, at the front door only, a
   question stage 1 let in whose hazard the parser cannot name. In the second
   case Jev may name one of our three hazards and may not refuse.
3. Jev's answer is a signal and code decides, with thresholds set before any
   measurement from TypeSafe's documented starting points: refuse when the
   three refusal topics sum to at least 0.8 and the companion noul is at most
   0.3; name a hazard when the topic, the companion noul and the hazard are
   each at least 0.7. Anything in between defers, which is today's behaviour.
4. The in-scope options come first, so the documented first-option lean, if it
   shows, leans toward deferring rather than refusing. The eval measures it by
   reversing the order (`--order-check`).
5. The worst a manipulated answer can do is refuse the manipulator's own
   question, or route a question with no hazard words to a report for one of
   our three hazards, which the report then names with Jev's probability. It
   cannot bypass stage 1, reach the answering prompt, or touch a citation.

### The rule for the citation check (b)

It only annotates `RiskReport.evidence`. It never adds or removes a citation
and never moves the risk level or the confidence, and the structural citation
validation in `rag/answer.py` is untouched. Labels: strong at P(supports) of
0.8 or more, moderate from 0.5, weak below; contradicted at P(contradicts) of
0.8 or more. One request per (sentence, passage) pair, at most 4 sentences by
the 8 passages retrieval can return, all sent at once, with the whole check held
to one call's budget. Anything a cap or the deadline leaves out is marked
`unchecked` and the check `partial`; a sentence is only called unsupported when
every cited page was read against it.

### Failure policy

Jev being down never breaks a request. `rag/jev_client.py` raises only typed
errors; every caller catches them and keeps today's behaviour; telemetry
records the failure with a short reason and never the key or user text.
Triage has a 2.5 s total budget (`CRG_JEV_TRIAGE_BUDGET_S`), verify 6 s.
Retries follow the official SDK's set (408, 429, 5xx, honouring `Retry-After`
and `retry-after-ms`) within that budget. Because httpx's read timeout restarts
on every chunk, each attempt's waits are carved out of what is left and the
reply is read chunk by chunk against the deadline, so a call can outlast its
budget by at most one chunk's wait. 401, 402 and 403 pause calls for five
minutes; three outages in a row pause them for thirty seconds. Identical
requests are served from a bounded in-process cache for an hour.

## Criteria for turning each part on

Fixed on 2026-11-11, before any Jev answer existed. All on the dev set and the
probes, from `uv run python -m evals.run_jev_eval --task all --order-check --repeats 2`.

Triage (`CRG_JEV=triage`) only if all of these hold:

1. **Hard gate:** no new false refusals at scope on the dev set. The
   background-science items and PI-03 in the silent bucket must still reach the
   answering model.
2. **Hard gate:** no wrong hazard named by the front door on the dev set or the
   probes. A missing hazard is allowed: it falls back to today's refusal.
3. **Usefulness:** OOC-01 or OOC-03 refused at the guard, or at least 4 of the 7
   in-scope probes given the right hazard.
4. **Stability:** at least 90% of silent-bucket rows keep their decision under
   reversed option order and across two uncached runs, and every refusal that
   flips is listed.
5. **Latency:** live-call p95 at or under 1,000 ms with no failed call in the run.
6. Then one held-out run at the next release gate, counted as an exposure,
   before the default changes, the same rule stage 2's `embed` arm is under.

Citation check (`CRG_JEV=verify`) only if:

1. At least 80% of the `cid_flipped` and `premise` pairs come back contradicted.
2. At least 80% of `cid_true` pairs come back strong, at least 80% of `cross`
   pairs come back weak, and the AUC of P(supports), supporting pairs against
   cross pairs, is at least 0.9.
3. Then a comparison on real Gemini answers against the claim judge
   (`evals/claim_judge.py`), which needs Gemini calls and therefore waits for
   that rule to change. Until it passes, the UI should not show the labels.

## Measured, 2026-11-11, through the Experiential Labs gateway

`JEV_BASE_URL=https://api.experientiallabs.ai uv run python -m evals.run_jev_eval
--task all --order-check --repeats 2`, dev set and the 14 probes only, report in
`evals/results/jev-dev-2026-11-11.md`. The gateway relays to TypeSafe (replies
carry `"provider": "typesafe"`) but grants only the alias `jev-latest` and does
not say which version answered. 244 live calls, paced 7.5 s apart, 0 failures,
4 retries; 216,786 input tokens (255,729 with output); $0.009105 at the $0.042
per million list price, $0.00 reported by the gateway (free plan).

| measure | keyword guard today | with Jev (`CRG_JEV=triage`) |
| --- | --- | --- |
| scope decided right, dev set | 58/60 | 60/60 |
| scope decided right, 14 probes | 7/14 | 14/14 |
| hazard named, dev questions with one | 39/42 | 41/42 |
| hazard named, 7 in-scope probes | 1/7 | 6/7 |

With Jev there were no false refusals and no wrong hazard anywhere. OOC-01
(the NDC pledge) and OOC-03 (GDP lost) are now refused at the guard, before
retrieval and Gemini, and all seven out-of-scope paraphrases are refused with a
name. PI-03 and the eight background-science questions in the silent bucket
were deferred, so they still reach the answering model. The two misses fell
back to today's refusal: SP-03 (Jev's companion noul 0.69, just under 0.7) and
PIN-01, "Are storms getting stronger?" (noul 0.59). Jev on its own, without
stage 1, also scored 60/60 and 14/14 on scope, and its top hazard pick was
right on 42/42 and 7/7.

Latency per call, pacing waits excluded: triage p50 640 ms, p95 2,055 ms (30
of 134 over 1 s, slowest 4,047 ms); citation check p50 606 ms, p95 1,748 ms.

Citation check, 110 pairs:

| pair kind | n | labelled as expected |
| --- | --- | --- |
| `cid_true`, expected strong | 15 | 11 strong (2 moderate, 2 weak) |
| `quote`, expected strong | 38 | 36 strong (2 moderate) |
| `cid_flipped`, expected contradicted | 15 | 13 contradicted |
| `premise`, expected contradicted | 4 | 4 contradicted |
| `cross`, expected weak | 38 | 37 weak |

AUC of P(supports), supporting pairs against cross pairs: 0.9955.

Against the criteria fixed above:

- Triage: 1 met, 2 met, 3 met, 4 met (0 of 30 decisions changed with the topic
  options reversed, and 0 of 30 across two uncached runs). **5 not met**:
  triage p95 is 2,055 ms against the 1,000 ms gate, though no call failed. 6,
  the held-out run, has not been done.
- Citation check: 1 met (17 of 19, 89%, contradicted). **2 not met**: 11 of 15
  `cid_true` pairs are strong, 73% against 80%; the `cross` (37 of 38) and AUC
  parts pass. 3, the comparison on real Gemini answers, has not been done.

Decision after measurement: both stay off by default.

- **Triage.** It passed every quality gate on this data. The latency gate was
  written against TypeSafe's "about 100 ms" direct calls, which this run could
  not test. Through this gateway, turning triage on would add a median 0.64 s
  and a p95 about 2 s to questions that name no hazard word, which is the only
  traffic Jev sees. Whether that is worth the fixes above is a product call
  for Aswani; it is not a reason to re-read the criterion.
- **Citation check.** It misses criterion 2, but two of its four non-strong
  `cid_true` pairs look like label errors rather than Jev errors. CID-11's
  recorded passage says "a decrease in frequency of cyclones landing in SEAF,
  ESAF and MDG", which is not mean wind speed, and CID-06's says "Annual mean
  precipitation is projected to increase", which is not heavy precipitation.
  Both claims come from the dev set's recorded Chapter 12 parse, so Jev's
  "weak" reads the passage correctly, and the parse in `rag/cid.py` should be
  checked. CID-11 is also the dev set's one e2e false refusal. Fix the labels,
  then re-measure; until then this counts as not met.

## Consequences

- With the flag off: zero Jev calls (`tests/test_jev_wiring.py`), and the parse
  and report carry no new fields (`triage` and `evidence` stay `None`).
- New surface: `POST /triage`; `RiskReport.triage` and `RiskReport.evidence`;
  `ScopeDecision.stage == "jev"`; `jev:triage` and `jev:verify` telemetry ops,
  priced from the response's versioned model name.
- `evals/run_e2e_eval.py` now refuses a held-out run (`EVAL_SET=test`) while
  `CRG_JEV` is on, so held-out questions never reach TypeSafe and gate numbers
  never mix in an unmeasured reader; a dev run records the mode in its
  artifact and filename, like the stage-2 arm.
- A second agent reviewed the code before hand-off. Its findings on the time
  budget, the verify caps, threshold parsing, reply text in error reasons,
  header-unsafe keys and eval scoring were fixed and pinned by tests. One
  finding predates this work and is left open: the front door's hazard patterns
  are wider than stage 1's ("extreme heat", "rainstorms"), so "drought and
  extreme heat" is answered as heat where stage 1 alone would refuse it as
  drought. That follows the compound-question policy in `rag/scope.py`, but it
  deserves a deliberate decision.
- A triage request for a 100-character question is 2,477 bytes on the wire,
  and the measured average was 1,020 input tokens per triage request and 728
  per citation pair: $0.000043 and $0.000031 each at the list price.
- Privacy: with triage on, the text of a question that has no hazard word is
  sent out; with verify on, the Gemini statement, which can name the queried
  place, and public IPCC passages are sent. On the gateway route they reach
  Experiential Labs first, which records prompts and responses, and then
  TypeSafe. Both providers' wording is quoted in LIMITATIONS.md.
- A second vetted provider, the Experiential Labs gateway, is selected with
  `JEV_BASE_URL`; the target can only be one of the two pinned hosts. It uses
  `jev-latest`, a 32,000-token request limit and wider timeouts (triage 4 s,
  verify 8 s), and its reported `usage.cost` is used as the cost in telemetry.
- Found while wiring the front door and fixed in the same change, independent
  of the flag: "marine heatwave" contains "heatwave", so the front door used to
  parse it as our heat hazard and build a land heat report. It now refuses, as
  stage 1 already does.

## What was not tested

- TypeSafe direct: the key is a gateway key, and TypeSafe refused it, so the
  latency and the pinned `jev-1.14.0` behaviour there are unmeasured.
- The production triage budget through the gateway (4 s): the eval widened the
  budget to 40 s so a rate-limited call could back off, and 4 of the 244 calls
  needed a retry; under the 4 s budget some of those would have fallen back.
- Robustness to injected text: not sent, by TypeSafe's terms and for lack of
  labelled items.
- Non-English questions. The Jev 1.14 changelog says "Better multilingual
  support."; the docs read on 2026-11-11 give no accuracy figure per language.
- The citation check on real Gemini answers.
