# Limitations (read before trusting a number)

This project's core bet is that stated limitations build more trust than
polished silence. Everything below is also encoded in the output schema where
possible, so a downstream consumer can check it programmatically.

## Hazard statistics (ERA5 return levels)

- **Point-interpolated reanalysis, not station observations.** Return levels
  come from ERA5 (~25 km) interpolated to the requested coordinates. Every
  `HazardStat` carries `representativeness = point_interpolated_reanalysis`,
  its native resolution, and `is_bias_corrected = false`. In heterogeneous
  terrain the true local extreme can exceed the reanalysis value.
- **Wind is a lower bound.** ERA5 underestimates sharp convective gusts; the
  wind statistic says so in its `interpretation` field.
- **`record_max` ships beside the fitted return levels** so a degenerate
  extreme-value fit (100-year level ≈ record max) is visible at a glance.
- **GEV fit is stationary by default; a non-stationary check runs beside it.**
  Baseline return levels are fitted on 1960–2022 as a stationary record. A
  drifting-location GEV (`tools/gev_trend.py`) also fits a linear warming trend,
  and a likelihood-ratio test decides whether it is signal (p < 0.05); when it
  is, return levels are reported "effective" at the latest year instead of
  averaged across the record. Residual limitation: the trend model assumes the
  drift is *linear* in year and still fits a GEV, so it will not capture abrupt
  or non-linear regime shifts.

## Retrieval and answers

- **Corpus = IPCC AR6 WG1 SPM + Chapter 11 + Chapter 12 only.** Questions whose
  evidence lives elsewhere (WG2 adaptation, WG3 mitigation, the Atlas) are out
  of corpus and should be refused; the eval's out-of-corpus slice measures this.
- **Retrieval quality is published, not perfect.** On the held-out test set
  (105 questions, first exposure) the hybrid BM25+dense retriever gets headline
  recall@3 87% / @5 91% / @10 96% on answerable questions. On the full
  60-question dev set (49 answerable, measured 2026-09-02) it gets recall@3 82%
  / @5 90% / @10 94%, against 76% / 82% / 88% for BM25-only, which is what a
  keyless deployment falls back to. That dev-set headline was 91% on the set's
  first 45 questions; adding the 15 `cid-table` questions moved it to 82%,
  because that slice is the hardest in the set. Per-slice numbers, including the
  ones we are not proud of (test-set premise-injection R@3 = 59% and test-set
  regional-table 77%; dev-set `cid-table` at hybrid R@3 67% / @5 80% / @10 93%,
  n=15, MRR 0.57), are in the eval output. Rerun it with
  `uv run python -m evals.run_retrieval_eval`.
- **Dev and test sets are split (eval v2).** The dev set
  (`evals/gold_set.json`, 60 questions: the original 45 plus 15 `cid-table` ones
  added with the Chapter 12 projection layer) steered retrieval and context-size choices
  (chunking, top_k), so its numbers carry optimistic bias and it is kept for
  diagnosis only. A second hash-pinned set of 105 questions (`evals/gold_set_v2.json`),
  authored after the dev set and never used to tune anything, is the held-out
  test set; it runs at release gates only and every published test number
  carries its exposure count. Failures are diagnosed on the dev set, never by
  iterating against the test set.
- **The scope guard is lexical by default; a semantic second stage exists but
  ships OFF.** Stage 1 (`rag/scope.py`) matches hazard vocabulary, runs before
  any model call, and cannot be prompt-injected, so a paraphrase that avoids
  all known terms still slips past it to the LLM layer, whose prompt-level rules
  are best-effort. In the arm comparison below that blind spot is 13 of the 45
  questions it ran on.
  Stage 2 (`rag/scope_semantic.py`, flag `CRG_SCOPE_STAGE2=off|embed|llm`)
  reads *only* that blind spot: it scores the question against 41 labelled
  anchors (`rag/scope_anchors.json`) in embedding space and refuses only when
  the nearest out-of-scope anchor clears a floor of 0.60 and beats the best
  in-scope anchor by 0.04, otherwise deferring to today's behaviour. Stage 1's
  verdict is never overridden, and stage 2 can only add a refusal, never remove
  one. Thresholds are calibrated, not guessed
  (`evals/results/scope-stage2-calibration.json`: 0 false refusals at every grid
  point tested).
  Measured on the dev set, `uv run python -m evals.run_e2e_eval --scope-stage2 <arm>`
  (gemini-2.5-flash, top_k 8, 2026-09-02; correct answer / correct refusal /
  false refusal / false answer): the shipped `off` arm is **48/11/1/0** on all
  60 questions. The three-arm comparison below was run before the 15 `cid-table`
  questions landed and is still at n=45: `off` 34/11/0/0 · `embed` 33/11/1/0 ·
  `llm` 34/11/0/0. Zero false answers in every arm. The embed arm's single
  false refusal (RT-07) carries lexical hazard
  vocabulary and therefore never reached stage 2: it is known run-to-run LLM
  variance on that item, not a guard regression. `embed` added 24 ms and **zero**
  API calls per question, because the question vector is already in the
  retriever's cache; it recovered a supported hazard from two questions the
  regex misses ("hot-extreme (TXx/TNn)") and refused two out-of-corpus policy
  questions before the LLM. `llm` matched the baseline matrix but cost 2.8 s and
  $0.00025 per question, failed 1 call in 13, and refused the AMOC
  premise-injection item as a scope violation (right cell, wrong reason),
  dropping grounded refusals from 3/4 to 2/3. **The default stays `off`:** the
  dev set shows no gain from either arm, and the only evidence stage 2 closes the
  paraphrase hole is an authored probe set (7/7 out-of-scope paraphrases refused,
  7/7 in-scope paraphrases correctly routed), not held-out data. `embed` is the
  arm to promote, and only after a held-out test-set run confirms the matrix
  holds.
- **Citation validity is checked at page level.** A citation is scored valid
  when it lands on a page that answers the question; claim-level entailment
  (does this sentence support this exact claim) is not yet automated in the
  release gate.
- **Figures and maps are not read.** The corpus is the PDFs' text layer. Values
  are never extracted from charts (a model reading a chart is unverifiable);
  where a figure is the only source, the honest behavior is to point to it.
- **"Projected change" is a Ch.12 sentence, not a table cell, and never a
  number from a figure.** The `projected_change` section quotes one verbatim
  sentence from AR6 WG1 Chapter 12's regional climatic impact-driver
  assessment (Sections 12.4.x) and reports the direction, the calibrated
  confidence phrase, and any warming level or period stated *in that sentence*,
  parsed by regex, with no model writing any of it. It cannot read the Ch.12 CID
  summary tables (12.3–12.10) themselves: those encode each region/driver cell
  as a coloured glyph, so the PDF text layer yields only the region label and
  footnote markers (measured: 61 Ch.12 table-row chunks carry a `Table 12.N`
  caption, none carries a direction or a confidence for its own row, and the
  confidence wording on those pages belongs to the shared legend). Granularity
  is therefore the AR6 reference region containing the point, never the city,
  and the section is absent, with the report saying regional projections were
  not found in the corpus, whenever no qualifying sentence is retrieved, which
  is the case for roughly a quarter of region/hazard pairs and for every ocean
  point. Where one sentence carries two calibrated phrases and no clause
  boundary separates them, `confidence_language` is reported as null rather
  than guessed; the quoted sentence is always shown so a reader can check the
  attribution. Ch.12 also mixes observed trends with projections in a single
  sentence, so a "projected change" statement can be quoting an observed one:
  read the sentence, not just the direction badge.

## Location coverage

- **Any coordinate works; regional IPCC context does not follow everywhere.**
  The forecast and the ERA5 hazard statistics are global. The AR6 assessment
  tables are keyed to *land* reference regions, so a point in open ocean (or in
  a gap between regions) maps to no region: the report still runs, and says the
  regional IPCC context is unavailable rather than borrowing a neighbour's.
  The polygons are the IPCC WGI Atlas reference regions v4, bundled in
  `tools/ar6/` under CC BY 4.0. Required citation: Iturbide, M., Fernández, J., Gutiérrez, J.M.
  et al. Implementation of FAIR principles in the IPCC: the WGI AR6 Atlas
  repository. *Scientific Data* 9, 629 (2022).
  <https://doi.org/10.1038/s41597-022-01739-y>. They are v4 and frozen, so a
  later Atlas revision would need a manual refresh of that file.
- **The geocoder returns one best match, not a disambiguation list.** Open-Meteo
  ranks by population and relevance and we take the top hit, so an ambiguous
  name ("Springfield") can resolve somewhere you did not mean. The resolved
  name, country and coordinates are always displayed so a wrong pick is visible;
  entering latitude/longitude directly bypasses the guess entirely.
- **The map is display-only.** It marks the selected point; it cannot be clicked
  to choose one. Streamlit's chart selection reports which rendered *objects* a
  click picked, not the coordinates of the click.

## Model dependence

- Answers are generated by a pinned LLM (`gemini-2.5-flash`, temperature 0),
  the code default in `rag/gemini_client.py` and the model every published
  number was measured on. `CRG_GENERATE_MODEL` overrides it per deploy.
  Structured-output constraints and citation validation bound what it can emit,
  but generation is not fully deterministic across model updates; the frozen
  benchmark exists to detect regressions when the pin changes.

## Jev (TypeSafe), optional and off by default

- **Measured on the dev set only, through one route, and kept off.** Question
  triage and the citation support check (`CRG_JEV=triage|verify|all`,
  [adr/0002](adr/0002-jev-question-triage-and-citation-support.md)) were
  measured on 2026-11-11 on the 60 dev questions and 14 authored paraphrase
  probes, through the Experiential Labs gateway. Triage met every quality gate
  (60/60 and 14/14 on scope, no false refusal, no wrong hazard) but not the
  latency gate (p95 2,055 ms against 1,000 ms). The citation check missed one
  accuracy gate (11 of 15 true Chapter 12 claims read as strong, against 80%).
  These are small sets that steered development, the held-out set has not been
  run, and TypeSafe direct is unmeasured. Leave the flag off until the ADR's
  criteria are met.
- **What leaves this system.** With triage on, the first 1,000 characters of a
  question that names no hazard word are sent out; questions the lexical stage
  settles are not. With verify on, the IPCC statement Gemini wrote, which can
  name the queried place, is sent with the public IPCC passages it cites.
- **Where it goes depends on `JEV_BASE_URL`.** Unset, requests go straight to
  TypeSafe and only TypeSafe's terms apply. Set to
  `https://api.experientiallabs.ai`, they go to the Experiential Labs gateway,
  which relays them to TypeSafe (its replies carry `"provider": "typesafe"`),
  so the gateway's terms apply first and TypeSafe's then apply to what it
  receives. A deployment that cannot send user text to a US processor must
  keep `CRG_JEV=off` on either route.
- **TypeSafe's wording** (both routes). Its privacy policy (last updated Nov
  19, 2025, read 2026-11-11 at https://typesafe.ai/legal/privacy-policy) says:
  "We will not train or fine tune any artificial intelligence or machine
  learning models on your prompts or other Input." and "The Services are hosted
  in the United States (“U.S.”). If you choose to use the Services from the
  EEA, the UK or other regions of the world with laws governing data collection
  and use that may differ from U.S. law, then please note that you are
  transferring your personal data outside of those regions to the U.S. for
  storage and processing." The docs say "Jev is not trained on customer
  requests or responses." (https://docs.typesafe.ai/models). The customer
  agreement (last updated Sep 23, 2026, https://typesafe.ai/legal/mca) also
  says: "“Telemetry” means information generated in connection with the
  Services, such as technical logs, hashes, summary statistics and
  classifications, metrics, and learnings related to Customer’s use of the
  Services. TypeSafe may Process Telemetry without restriction, including to
  improve the Services or TypeSafe’s other products and services." Zero data
  retention is offered to enterprise customers only
  (https://typesafe.ai/enterprise).
- **The gateway's wording** (gateway route only). Its privacy notice (last
  updated September 16, 2026, read 2026-11-11 at
  https://www.experientiallabs.ai/privacy) says: "The standard hosted service
  records prompts, responses and related activity." and "Stored prompts and
  responses in the primary capture table are automatically removed after 30
  days. That period does not describe every trace, file, backup or
  provider-held copy." It adds: "A provider's zero-retention or no-training
  setting covers that provider's specified handling; it does not describe all
  processing by Experiential.", "Our Terms do not authorize using Customer
  Content or datasets derived from it to train shared models or other
  customers' models, or selling those datasets." and "Experiential operates in
  the United States." Its model list (`GET /v1/models`, 2026-11-11) reports
  `jev-latest` with `zdr: false` and `no_training: false`, for the gateway and
  for its `typesafe` upstream alike; the meaning of those flags was not checked
  beyond their names.
- **A signal, never a gate the system relies on.** The lexical scope stage
  stays first and final, and the citation check only annotates. TypeSafe
  documents that content written to steer the model "can move the answer", so
  a question crafted against Jev can at worst get its own question refused, or
  get a question with no hazard words routed to a report for one of the three
  supported hazards, which then says Jev chose it and with what probability.
- **Readings are pinned, not guaranteed.** An identical request is not
  promised the same answer, so the client keeps one reading per identical
  request for an hour, as the answer cache does for answers; `--repeats` in the
  eval measures how often a decision changes. Dates, numbers and horizons are
  never asked of Jev (a documented weakness). TypeSafe direct is pinned to the
  version `jev-1.14.0`; the gateway grants only the alias `jev-latest` (it
  refuses `jev-1.14.0` with HTTP 403 `model_not_granted`) and its replies name
  the alias, not the version that answered, so a TypeSafe release can move its
  readings, and the thresholds measured against it, without any change here.

## Risk bands

- **The band is local rarity, and the report says which numbers produced it.**
  `agent/risk_bands.py` places the forecast peak on the location's own fitted
  return-level curve: below the 2-year level is low, 2 to 10 years moderate,
  10 to 50 high, at or above 50 severe. Between two fitted levels the return
  period is interpolated linearly in log(T), which is exact only for a Gumbel
  tail and approximate for the general GEV, so the reported "1-in-N-year"
  figure is an estimate within its bracket, not a fitted quantile. The bracket
  itself, which decides the band, is read straight off the fit.
- **The fitted levels carry sampling noise the band does not show.** Each level
  ships a 90% bootstrap interval (`ci_low`/`ci_high`), and a peak sitting near
  a band edge could fall either side of it under a different 60-ish years of
  record. The band is a point verdict on a curve that has width.
- **Absolute cutoffs survive only as the fallback.** With no fitted climatology
  the band comes from fixed Day-1 numbers (heat 35/40/45 °C, precipitation
  20/50/100 mm, sustained wind 40/62/88 km/h). These ignore the location
  entirely, which is exactly the flaw the return-level path exists to fix, so
  the report's `severity_basis` driver states in words that the band is
  absolute and why the climatology path was unavailable.
- **The forecast skill that scales confidence is a lower bound, measured on 13
  cities.** The `forecast_skill` driver and the lead-day weight in the
  confidence come from `tools/forecast_skill_table.json`, which scores each lead
  time against the model's own day-0 run rather than station observations (so
  these are run-to-run consistency errors, not true forecast errors) at 13
  cities over 2024-2025 with GFS Global, while the live forecast is served by
  Open-Meteo's `best_match` blend: the weight is an approximation of the skill
  the agent actually delivers, and it is global, never local to the queried
  point.
- **Wind is graded on the gust, and only against a gust fit.** The forecast now
  fetches `wind_gusts_10m_max` alongside `wind_speed_10m_max` because the ERA5
  wind climatology is a gust fit; the sustained speed is displayed but never
  compared to a gust return level. The absolute fallback still grades the
  sustained speed, which is what its Beaufort-derived cutoffs describe.
