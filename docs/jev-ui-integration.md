# Jev in the UI: integration note for after the redesign

Status: the backend is built and off by default (`CRG_JEV`, see
[adr/0002](../adr/0002-jev-question-triage-and-citation-support.md)). Nothing in
`ui/` uses it yet. This note says what to show, where it comes from, and what
not to show, so the redesign can wire it without reading the backend.

## 1. Read the question before running it

Call the front door's reader as the person submits (or after a short pause in
typing), before starting the report:

- In-process, as `ui/app.py` already does with `parse_query`: `agent.nl.triage_query(text)`
  returns a `TriageOutcome`.
- Over HTTP: `POST /triage` with `{"query": "..."}` returns
  `{"triage": TriageOutcome, "telemetry": {...}}`. Same `x-api-key` rule as `/query`.

It never geocodes, fetches a forecast or calls Gemini. With the flag on it makes
at most one Jev request, and none when the words already name heat, rain or
wind. When Jev is asked, expect a median of about 0.6 s and a p95 of about 2 s
through the Experiential Labs gateway (measured 2026-11-11, adr/0002). If Jev
does not answer in time, the call gives up after the provider's budget (4 s on
the gateway, 2.5 s on TypeSafe direct) and the reply is the lexical reading,
marked unavailable. So show a pending state on the hazard control rather than
blocking the Ask button.

`TriageOutcome` fields:

| field | meaning | show it as |
| --- | --- | --- |
| `hazard` | `heatwave`, `extreme_precip`, `wind`, or null | pre-select the sidebar hazard |
| `horizon_days` | from the words ("next 10 days"), default 7 | pre-set the horizon slider |
| `place` | the place string the words name, or null | pre-fill the place box (still geocode it) |
| `refusal` | the exact sentence `/query` would refuse with, or null | see section 2 |
| `triage` | how it was read; null when the flag is off | see the table below |

`triage` (a `QueryTriage`) says who named the hazard:

| `hazard_source` | what to show next to the hazard |
| --- | --- |
| `lexical` | nothing extra: the person's own word named it |
| `jev` | "Read as heat (Jev, 93%)" using `hazard_probability`, with the control left editable |
| `embed` or `llm` | "Read as heat (semantic match)": scope stage 2, no probability |
| null | no hazard was named |

When `hazard_source` is `jev`, the person is about to get a report on an
interpretation, so say so where the hazard control is, and make changing it one
click. Do not show the probability as "confidence" in the report sense: it is
Jev's probability that the question is about that hazard, nothing more.

`triage.jev_status` is `not called`, `ok`, `cached`, or `unavailable: <reason>`.
Show nothing for the first three. For `unavailable`, show nothing to the reader
either: behaviour is exactly the non-Jev behaviour. Log it.

## 2. The instant refusal

When `refusal` is not null and the reason is the hazard or the topic (not a
missing place), show the refusal at once, without starting the progress panel,
using the same refusal styling the report uses today. Suggested copy:

> Out of scope: {refusal}. This agent assesses heat, extreme rain and wind.

If the only problem is the place ("could not identify a location in the
question"), do not refuse when the sidebar already holds a resolved place:
run with the parsed hazard and horizon and the sidebar place.

The full run returns the same refusal in `report.refusal`, so a UI that skips
`/triage` still behaves correctly, only later.

## 3. Evidence strength per citation

With `CRG_JEV=verify` or `all`, a report with IPCC citations carries
`report.evidence` (an `EvidenceCheck`):

- `status`: `ok`, `partial` (some pairs could not be checked) or `unavailable`.
- `citations`: one `CitationEvidence` per cited page, matching
  `report.citations` by `source` and `locator`, each with `supports` and
  `contradicts` (probabilities) and `strength`:

| `strength` | badge on the citation |
| --- | --- |
| `strong` | "Supports the statement" |
| `moderate` | "Partly supports" |
| `weak` | "Does not clearly support this statement" |
| `contradicted` | "May contradict the statement", styled as a warning |
| `unchecked` | no badge |

- `unsupported_statements`: sentences of the IPCC statement that no cited page
  supports at 0.5 or more. Underline them, with the hover text "No cited page
  clearly supports this sentence".

Hold this section back until ADR 0002's citation criteria pass. They need a
measurement on real Gemini answers that has not run yet; until then the labels
are unvalidated and should not be shown to readers.

## What not to do

- Do not hide or reorder citations by strength. The check annotates; the
  structural citation validation decides what is cited.
- Do not change the risk level, the confidence or the band copy from any Jev
  number.
- Do not show Jev's raw topic probabilities to readers; they are for logs.
- Do not send anything to Jev from the UI directly. The key stays server-side.
