# Jev triage, held-out check, 2026-11-11

Run 2026-11-11T14:21:33+00:00. Held-out set (`evals/gold_set_v2.json`, 105 questions, frozen hash in `evals/gold_set_v2.sha256`: True). Triage only: no Gemini or Vertex call, no citation check, scope stage 2 off.

- Exposure: one exposure of the held-out set, the first for Jev; earlier dated runs on it: 2026-07-18 (retrieval, e2e), 2026-07-22 (e2e), 2026-09-07 (retrieval)
- Sent to Jev: 37 of 105 questions, the ones `CRG_JEV=triage` itself sends (36 in the silent bucket, 1 where the front door cannot name the hazard stage 1 found). The other questions never left this machine.
- Privacy: The Experiential Labs privacy notice (last updated September 16, 2026, read 2026-11-11 at https://www.experientiallabs.ai/privacy) says: "The standard hosted service records prompts, responses and related activity." and "Stored prompts and responses in the primary capture table are automatically removed after 30 days. That period does not describe every trace, file, backup or provider-held copy."
- Jev status: **ok**, through provider `experientiallabs`, model requested `jev-latest`, models that answered: jev-latest
- Live Jev calls: 111, failures: 0, retries: 4, input tokens: 114747
- Spend: $0.004819 at TypeSafe's list price (cap $1.00); reported by the provider: $0.000000
- Latency per live call: p50 762.0 ms, p95 2854.6 ms (36 of 111 calls over 1,000 ms)
- Requests paced 7.5 s apart; pacing waits are not counted as latency. The per-call budget was widened for the run so a rate-limited call could back off; production keeps the provider's own budget.
- Decision thresholds (unchanged from the dev run): refuse at summed refusal probability >= 0.8 with the companion noul <= 0.3; hint at >= 0.7 with the noul >= 0.7

## Pass conditions (fixed before the run)

| condition | result | detail |
| --- | --- | --- |
| 1. no new false refusal at scope | NOT met | new false refusals: SP2-08 |
| 2. no wrong hazard named by the front door | met | wrong: none |
| 3. stability, at least 90% kept | met | kept with the options reversed: 0.973; kept across two uncached runs: 1.0 |

## Scope: refuse at the guard or not

| arm | correct | false refusals | false accepts |
| --- | --- | --- | --- |
| stage 1 today | 97/105 (92%) | none | OOC2-01, OOC2-02, OOC2-03, OOC2-04, OOC2-05, OOC2-06, OOC2-07, OOC2-08 |
| stage 1 + Jev (`CRG_JEV=triage`) | 98/105 (93%) | SP2-08 | OOC2-01, OOC2-02, OOC2-03, OOC2-04, OOC2-06, OOC2-08 |

A false accept is a question that should be refused at scope reaching the answering model, which refuses it itself today (held-out e2e, 2026-07-22: all 35 questions built to be refused were refused). It is a missed instant refusal, not a false answer.

## Hazard named by the front door

| arm | right | missed | wrong |
| --- | --- | --- | --- |
| lexical parser today | 57/77 | RT2-01, RT2-04, RT2-06, RT2-09, RT2-14, RT2-17, RT2-19, DR2-03, DR2-08, SP2-06, SP2-11, SP2-12, SP2-14, SP2-23, MP2-05, MP2-11, MP2-12, PI2-06, PI2-08, PI2-13 | none |
| front door with Jev | 73/77 | SP2-11, SP2-23, MP2-12, PI2-06 | none |

## Usefulness (reported, not gated)

- Refused at the guard with Jev that stage 1 lets through: OOC2-05, OOC2-07
- Hazards the front door names with Jev that the lexical parser misses: DR2-03, DR2-08, MP2-05, MP2-11, PI2-08, PI2-13, RT2-01, RT2-04, RT2-06, RT2-09, RT2-14, RT2-17, RT2-19, SP2-06, SP2-12, SP2-14

## Stability

- Topic options reversed: 1 of 37 decisions changed (OOC2-02); refusals among them: none; left out, Jev unavailable: none.
- 2 uncached runs per question: 0 of 37 decisions were not the same every time (none); refusals among them: none; left out, Jev unavailable: none.

## Questions sent to Jev, as it read them

| id | slice | should refuse | asked as | decision | refuse p | noul | hazard (p) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RT2-01 | regional_table | False | silent bucket | hint | 0.0 | 0.98 | heatwave (1.0) |
| RT2-04 | regional_table | False | silent bucket | hint | 0.0 | 0.89 | heatwave (1.0) |
| RT2-06 | regional_table | False | silent bucket | hint | 0.0 | 0.93 | heatwave (1.0) |
| RT2-09 | regional_table | False | silent bucket | hint | 0.0 | 0.91 | heatwave (1.0) |
| RT2-14 | regional_table | False | silent bucket | hint | 0.0 | 0.82 | heatwave (1.0) |
| RT2-17 | regional_table | False | silent bucket | hint | 0.0 | 0.89 | heatwave (1.0) |
| RT2-19 | regional_table | False | silent bucket | hint | 0.0 | 0.92 | heatwave (1.0) |
| DR2-03 | duplicate_region | False | silent bucket | hint | 0.0 | 0.92 | heatwave (1.0) |
| DR2-08 | duplicate_region | False | silent bucket | hint | 0.0 | 0.94 | heatwave (1.0) |
| SP2-01 | single_page | False | silent bucket | defer | 0.0 | 0.08 | heatwave (1.0) |
| SP2-02 | single_page | False | silent bucket | defer | 0.15 | 0.01 | heatwave (0.86) |
| SP2-03 | single_page | False | silent bucket | defer | 0.0 | 0.02 | heatwave (0.99) |
| SP2-04 | single_page | False | silent bucket | defer | 0.0 | 0.02 | heatwave (0.99) |
| SP2-05 | single_page | False | silent bucket | defer | 0.01 | 0.09 | heatwave (1.0) |
| SP2-06 | single_page | False | silent bucket | hint | 0.0 | 0.97 | heatwave (1.0) |
| SP2-07 | single_page | False | silent bucket | defer | 0.01 | 0.01 | heatwave (0.74) |
| SP2-08 | single_page | False | silent bucket | refuse | 0.85 | 0.01 | heatwave (0.97) |
| SP2-11 | single_page | False | silent bucket | defer | 0.01 | 0.09 | heatwave (1.0) |
| SP2-12 | single_page | False | silent bucket | hint | 0.0 | 0.98 | heatwave (1.0) |
| SP2-14 | single_page | False | silent bucket | hint | 0.0 | 0.89 | heatwave (1.0) |
| SP2-19 | single_page | False | silent bucket | hint | 0.0 | 0.89 | heatwave (1.0) |
| SP2-21 | single_page | False | silent bucket | defer | 0.0 | 0.03 | heatwave (1.0) |
| SP2-22 | single_page | False | silent bucket | hint | 0.0 | 0.98 | heatwave (1.0) |
| SP2-23 | single_page | False | front-door gap | defer | 0.0 | 0.52 | heatwave (1.0) |
| OOC2-02 | out_of_corpus | True | silent bucket | hint | 0.11 | 0.87 | extreme_precip (1.0) |
| OOC2-03 | out_of_corpus | True | silent bucket | defer | 0.15 | 0.01 | heatwave (0.94) |
| OOC2-05 | out_of_corpus | True | silent bucket | refuse | 0.96 | 0.02 | heatwave (1.0) |
| OOC2-07 | out_of_corpus | True | silent bucket | refuse | 1.0 | 0.2 | extreme_precip (0.99) |
| OOC2-08 | out_of_corpus | True | silent bucket | defer | 0.38 | 0.09 | heatwave (1.0) |
| MP2-05 | multi_page | False | silent bucket | hint | 0.0 | 0.79 | heatwave (1.0) |
| MP2-06 | multi_page | False | silent bucket | defer | 0.0 | 0.18 | heatwave (0.85) |
| MP2-11 | multi_page | False | silent bucket | hint | 0.0 | 0.96 | heatwave (1.0) |
| MP2-12 | multi_page | False | silent bucket | defer | 0.13 | 0.08 | heatwave (0.99) |
| PI2-06 | premise_injection | False | silent bucket | defer | 0.12 | 0.06 | heatwave (0.98) |
| PI2-08 | premise_injection | False | silent bucket | hint | 0.0 | 0.97 | heatwave (1.0) |
| PI2-13 | premise_injection | False | silent bucket | hint | 0.0 | 0.84 | heatwave (1.0) |
| PI2-15 | premise_injection | False | silent bucket | defer | 0.04 | 0.18 | heatwave (0.53) |

## Caveats

- The thresholds and question wording are the dev run's, unchanged; nothing was tuned on this set, and a failure here is diagnosed on the dev set, never by iterating against this one.
- Counts are small: few held-out questions lack every hazard word, so most of the set never reaches Jev, and a difference of one item is not evidence of a difference.
- The questions sent are now held by the gateway and by TypeSafe under their terms (LIMITATIONS.md), so a later held-out run cannot assume those providers never saw them.
- No prompt-injection probes were sent: TypeSafe's terms bar testing their systems.
