# ADR 0001: Which model answers, and on what evidence

- **Status:** Accepted
- **Date:** 2026-07-23
- **Decision:** Keep `gemini-2.5-flash` as the answering model. Ship the
  regional-grounding prompt rule **disabled**. Reject `gemini-3.5-flash` and
  `gemini-3.6-flash` for now.

## Context

The answering model had been switched from `gemini-2.5-flash` to
`gemini-3.6-flash` to get "faster, cheaper" responses. Nothing was re-measured.

The live demo then started returning short reports with **no IPCC citations**.
The logs showed no failure at all: embeddings 200 OK, `generateContent` 200 OK,
`finish_reason: STOP`, zero warnings. The system was working exactly as designed
and honestly reporting that the answerer had abstained.

Two facts made this worth a full investigation rather than a quick revert:

1. Every published number for this project (`false_answer = 0`, 94% citation
   validity, 87% held-out R@3) was measured on `gemini-2.5-flash`. The deployed
   model was different, so the README and UI were advertising numbers that did
   not describe what was running.
2. A model swap is invisible to the test suite. 240 tests passed, CI was green,
   nothing raised. Only an eval can catch this class of change.

## Decision criteria (fixed before looking at results)

1. **Hard gate:** `false_answer = 0`. A single confabulation blocks release.
2. **Stability:** identical inputs should produce comparable results, or the
   benchmark is not reproducible.
3. **Usefulness:** maximise correct answers, minimise false refusals.
4. **Citation validity** and **numeric provenance** must not regress.
5. **Latency and cost** — the original motivation for the change.

## Evidence

All runs on the frozen 45-question **dev** set (the held-out test set is
reserved for release gates; a config change is diagnosed on dev by protocol).

### Model comparison (dev set, one run each, regional rule enabled)

| Model | CA | CR | FA | FR | Citation | Numeric | p50 | p95 | Failures |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gemini-2.5-flash | 33 | 11 | 0 | 1 | 94% | 85% | 3.9 s | 40.3 s | 0 |
| gemini-3.5-flash | 30 | 11 | 0 | 3 | 93% | 70% | 8.9 s | 112.0 s | 1 |
| gemini-3.6-flash | 31 | 11 | 0 | 3 | 94% | 77% | 6.1 s | 16.1 s | 0 |

### Stability (repeat runs, identical inputs)

| Model | False refusals across runs | Spread |
| --- | --- | --- |
| gemini-2.5-flash | 1, 1, 1, 1 | none |
| gemini-3.6-flash | 3, 2, 5, 2 | 2 to 5 |

`gemini-3.6-flash` varies by three questions on the same 45 inputs. That alone
disqualifies it: a benchmark whose result moves that much cannot support a
reproducibility claim, and users see it as the same question sometimes being
answered and sometimes refused.

### Prompt ablation (gemini-2.5-flash)

| Regional-grounding rule | CA | FA | FR |
| --- | --- | --- | --- |
| Enabled (n=4) | 33 | 0 | 1 (RT-07, every run) |
| Disabled (n=2) | **34** | **0** | **0** |

The rule was written to stop 3.x abstaining on city-level questions. For 2.5 it
costs a correct answer, because 2.5 already infers that regional evidence
grounds a location question. Hence: keep the rule, default it off.

### Cost

Reported cost was `$0.052` (2.5) vs `$0.052` (3.6) — misleading on two counts,
both since fixed:

- Thinking tokens were not counted. One measured 3.6 call used 50 visible
  output tokens and **987 thinking tokens**, all billed. Roughly a 20x
  undercount on the output side, making 3.6 far more expensive than reported.
- Unpriced models silently reported `$0.00` instead of "unknown"
  (`gemini-3.5-flash` had no price entry).

### Release gate on the shipped config (held-out set, 2nd exposure)

`gemini-2.5-flash`, regional rule disabled, 105 held-out questions:

| Metric | Result |
| --- | --- |
| correct answer / correct refuse | 49 / 35 |
| **false answer** | **0** (gate met) |
| false refuse | 21 |
| citation validity | 96% (47/49) |
| numeric provenance | 88% (43/49) |
| p50 / p95 latency | 3.9 s / 15.5 s |
| cost | $0.285 for 105 questions (~$0.0027 each) |
| errors | 0 |

This is the second exposure of the held-out set, spent deliberately: the
published numbers must describe the configuration that is actually deployed.

## Consequences

- Production, code default and the `test_security` pin are all
  `gemini-2.5-flash`.
- `CRG_GENERATE_MODEL` and `CRG_REGIONAL_GROUNDING` remain env-overridable, so
  a future model change is a config change plus a re-run of this ADR's
  experiments, not a code rewrite.
- Telemetry now counts thinking tokens and refuses to present an unknown cost
  as zero.
- The determinism claim in `rag/answer_cache.py` was corrected: temperature 0
  does **not** make generation deterministic (measured for both families), so
  the cache pins one valid answer rather than guaranteeing the answer.

## What was not tested

- `gemini-3.6-flash` with `thinking_level` set to MINIMAL or LOW. This could cut
  its latency and cost and might reduce the erratic abstention. It is the
  obvious first experiment if 3.x is revisited.
- Repeat counts are small (n=2 to n=4). They are sufficient to separate a
  zero-variance model from one swinging across three questions, not to
  estimate a precise distribution.

## Rule for next time

Changing the answering model is a **release-gated change**: re-run the dev
evals, compare against this table, update the published numbers, and only then
deploy. The model that produced a number is now recorded inside every eval
artifact (`generate_model`) and in its filename, so this cannot silently drift
again.
