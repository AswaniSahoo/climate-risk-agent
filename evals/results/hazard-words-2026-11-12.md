# Everyday hazard words in the front door, 2026-11-12

On the live app (2026-11-11, `CRG_JEV=off`), "How windy will it get in Berlin over the next 7
days?" and "Will it be windy in Mumbai this week?" were refused with "could not identify a
supported hazard in the question", while "How strong will wind gusts be in Berlin over the next 7
days?" worked. With Jev off, the lexical parser in `agent/nl.py` is the whole front door.

Run 2026-11-11T20:01Z (dev) and 20:02Z (held-out), 2026-11-12 in India, local and
deterministic: `CRG_JEV=off`, `CRG_SCOPE_STAGE2=off`, no Gemini, Vertex or Jev credentials, no
network. Before is eecde01 (deployed revision 00031 runs 4d826ff, the same parser). After is
0b9990c: the words landed in fe3a492, and 0b9990c makes the evals score the reading the front
door uses.

## The change

- New words, whole words only (`_EVERYDAY_PATTERNS`): for heat, hot, hotter, hottest,
  scorching, scorcher(s), swelter, sweltering; for extreme precipitation, deluge(s), deluged,
  cloudburst(s), cloud burst; for wind, windy, windier, windiest, gusty, gustier, gustiest,
  gusting, gale(s), windstorm(s). "Rainy" and "downpour" were already read (`rain\w*`,
  `downpours?`).
- They name a hazard only when the older words name none and the scope guard (`rag/scope.py`,
  unchanged) names no unsupported hazard. AR6 Chapter 11 calls fire weather "compound hot, dry
  and windy events" (chunk `IPCC_AR6_WGI_Chapter11.pdf#p7#5`), so these words cannot let in a
  hazard the guard refuses. Every question the old parser named, or the guard refused by
  name, keeps the reading it had.
- Left out: "storm" (storm surge is refused as coastal flooding, hail and tornadoes as severe
  convective storms, and tropical, snow and dust storms are none of our three); "warm" and
  "warming" (dev SP-06, SP-11 and MP-08 use them for the background trend and name no hazard);
  "temperatures" (hot or cold, so SP-03, "temperature extremes", stays a miss); "hot spot"
  (where climate changes most, not heat).

## Dev set and probes, where the words were chosen

| measure | before | after |
| --- | --- | --- |
| hazard named, 42 dev questions with one | 39/42 | 41/42 |
| hazard named, 7 in-scope paraphrase probes | 1/7 | 6/7 |
| wrong hazard named, both | 0 | 0 |
| questions to refuse that the front door refuses, 7 dev and 7 probes | 14/14 | 14/14 |

Newly named: MP-10 and RT-10 ("hot temperature extremes", "hot-extreme"), PIN-02, PIN-03,
PIN-04, PIN-05 and PIN-07. Still missed: SP-03 and PIN-01 ("Are storms getting stronger?").
No other question changed its reading. These are the items the front door named with Jev on
2026-11-11 ([report](jev-dev-2026-11-11.md)).

## Held-out set, one confirmation run

The words were fixed at fe3a492 before this ran, and nothing changed after it.

| measure | before | after |
| --- | --- | --- |
| hazard named, 77 questions with one | 57/77 | 63/77 |
| wrong hazard named | 0 | 0 |
| scope stage 1 (code unchanged) | 97/105, no false refusal | 97/105, no false refusal |
| the other 87: refused by the front door for want of a hazard | 30 | 24, all among the 30 |
| `out_of_scope_hazard` questions refused, under the same name | 10/10 | 10/10 |
| all 18 questions to refuse (with `out_of_corpus`) that the front door refuses | 15/18 | 15/18 |

Newly named, all heat: RT2-01, SP2-06, MP2-05, MP2-11, PI2-08, PI2-13. Still missed: RT2-04,
RT2-06, RT2-09, RT2-14, RT2-17, RT2-19, DR2-03, DR2-08, SP2-11, SP2-12, SP2-14, SP2-23, MP2-12,
PI2-06. Jev named 10 of those 14 on 2026-11-11, for 73/77 ([report](jev-triage-test-2026-11-11.md)).
The three questions to refuse that the front door does not refuse (OOC2-01, OOC2-04, OOC2-06)
got the same hazard before, from the older words.

## Caveats

- Exposure: one more use of the held-out set, offline; no question left this machine. Its
  `out_of_scope_hazard` slice was read before the change, to know what must stay refused. Its
  fire-weather item, OSH2-06, is written in "hot" and "windy", and the guard-first rule was
  settled with it in view, so that refusal is not a blind result. The 77 questions with a
  hazard were not read; only ids and scores were printed.
- The dev set and the probes chose the words, so their after numbers are optimistic. The
  probes were written to dodge the lexical stage.
- All three sets are IPCC questions, not front-door questions with a place and a horizon. The
  three live questions are pinned in `tests/test_nl.py`.
- Words, not meaning: "in hot water" names heat, and "Are storms getting stronger?" is still
  refused with the flag off.

Reproduce: `uv run python -m evals.run_jev_eval --dry-run` (dev and probes) and
`uv run python -m evals.run_jev_heldout` (held-out, offline without `--spend-exposure`) score
`agent.nl.lexical_hazard` since 0b9990c; at eecde01 they give the before column. The front-door
refusal rows come from `parse_query` with `CRG_JEV=off` on every row, at both commits.
