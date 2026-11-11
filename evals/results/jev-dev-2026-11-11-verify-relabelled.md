# Jev dev-set eval, 2026-11-11

Run 2026-11-11T14:36:35+00:00. Dev set only (`evals/gold_set.json`, 60 questions, frozen hash in `evals/gold_set.sha256`: True). The held-out set was not read. No Gemini call was made.

- Jev status: **ok**, through provider `experientiallabs`
- Model requested: `jev-latest`; models that answered: jev-latest
- Live Jev calls: 108, failures: 0, retries: 1, input tokens: 78629
- Spend: $0.003302 at TypeSafe's list price (cap $1.00); reported by the provider: $0.000000
- Latency per live call: p50 841.2 ms, p95 2282.8 ms (first try only: p50 837.2 ms, p95 2166.0 ms; 32 calls over 1,000 ms)
- Requests paced 7.5 s apart; pacing waits are not counted as latency
- Decision thresholds (fixed before the run): refuse at summed refusal probability >= 0.8 with the companion noul <= 0.3; hint at >= 0.7 with the noul >= 0.7

## Scope: refuse at the guard or not

| set | n | stage 1 today | stage 1 + Jev (shipped with CRG_JEV=triage) | Jev alone |
| --- | --- | --- | --- | --- |
| dev | 60 | 58/60 (97%) | not measured | not measured |
| probe | 14 | 7/14 (50%) | not measured | not measured |

False refusals and false accepts, by item:

- dev, stage1: false refusals none; false accepts OOC-01, OOC-03
- probe, stage1: false refusals none; false accepts POUT-01, POUT-02, POUT-03, POUT-04, POUT-05, POUT-06, POUT-07

A false accept is a question that should have been refused at scope reaching the answering model, which refuses it itself today. It is a missed instant refusal, not a false answer.

## Hazard named by the front door

| set | items with a hazard | lexical parser today | front door with Jev | Jev alone |
| --- | --- | --- | --- | --- |
| dev | 42 | 39/42 (93%) | not measured | not measured |
| probe | 7 | 1/7 (14%) | not measured | not measured |

Missed or wrong, by item:

- dev, lexical: missed SP-03, MP-10, RT-10; wrong none
- probe, lexical: missed PIN-01, PIN-02, PIN-03, PIN-04, PIN-05, PIN-07; wrong none

## Citation support check

| pair kind | n | expected | Jev agrees | mean P(supports) | mean P(contradicts) | strong / moderate / weak / contradicted |
| --- | --- | --- | --- | --- | --- | --- |
| cid_true | 14 | supports | 13/14 | 0.9414 | 0.0429 | 12 / 1 / 1 / 0 |
| quote | 38 | supports | 37/38 | 0.9724 | 0.0258 | 36 / 1 / 1 / 0 |
| cid_flipped | 14 | contradicts | 14/14 | 0.0014 | 0.9921 | 0 / 0 / 0 / 14 |
| premise | 4 | contradicts | 4/4 | 0.015 | 0.9825 | 0 / 0 / 0 / 4 |
| cross | 38 | says_nothing | 37/38 | 0.0408 | 0.0121 | 1 / 0 / 37 / 0 |

- AUC of P(supports), supporting pairs against cross pairs: 0.998
- Pairs skipped while building: {'cid_chunk_missing': 0, 'cid_no_statement': 1, 'quote_not_found': 0, 'premise_unparsed': 0}; failed calls: none

## Silent-bucket items, as Jev read them

The questions stage 1 cannot see, the only ones the shipped path sends to Jev.

| id | slice | should refuse | Jev decision | refuse p | noul | hazard (p) |
| --- | --- | --- | --- | --- | --- | --- |
| SP-05 | single_page | False | not run | | | |
| SP-06 | single_page | False | not run | | | |
| SP-09 | single_page | False | not run | | | |
| SP-10 | single_page | False | not run | | | |
| SP-11 | single_page | False | not run | | | |
| SP-12 | single_page | False | not run | | | |
| MP-07 | multi_page | False | not run | | | |
| MP-08 | multi_page | False | not run | | | |
| RT-10 | regional_table | False | not run | | | |
| OOC-01 | out_of_corpus | True | not run | | | |
| OOC-03 | out_of_corpus | True | not run | | | |
| PI-03 | premise_injection | False | not run | | | |
| DR-02 | duplicate_region | False | not run | | | |
| CID-13 | cid-table | False | not run | | | |
| CID-14 | cid-table | False | not run | | | |
| CID-15 | cid-table | False | not run | | | |
| PIN-01 | paraphrase_in | False | not run | | | |
| PIN-02 | paraphrase_in | False | not run | | | |
| PIN-03 | paraphrase_in | False | not run | | | |
| PIN-04 | paraphrase_in | False | not run | | | |
| PIN-05 | paraphrase_in | False | not run | | | |
| PIN-06 | paraphrase_in | False | not run | | | |
| PIN-07 | paraphrase_in | False | not run | | | |
| POUT-01 | paraphrase_out | True | not run | | | |
| POUT-02 | paraphrase_out | True | not run | | | |
| POUT-03 | paraphrase_out | True | not run | | | |
| POUT-04 | paraphrase_out | True | not run | | | |
| POUT-05 | paraphrase_out | True | not run | | | |
| POUT-06 | paraphrase_out | True | not run | | | |
| POUT-07 | paraphrase_out | True | not run | | | |

## Caveats

- Thresholds and question wording were fixed before this run, from TypeSafe's documented starting points; the dev set steers development by protocol, so any later retune on these numbers makes them optimistic.
- The dev set is IPCC-knowledge questions, not front-door questions with a place and a horizon; the 14 paraphrase probes are closer to front-door wording but were authored, not sampled from users.
- Counts are small: 2 dev items should be refused inside the silent bucket, and 7 probes. A difference of one item is not evidence of a difference.
- No prompt-injection probes were sent to Jev: TypeSafe's terms bar testing their systems, and the dev set has no labelled injection items.
- Dev labels corrected after the first Jev run, each with its evidence in its notes in evals/gold_set.json: CID-06, CID-11. Jev's disagreement prompted the check, so the figures above carry a selection effect; adr/0002 says how the other cid-table labels were re-read.
