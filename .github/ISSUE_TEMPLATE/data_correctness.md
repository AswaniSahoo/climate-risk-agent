---
name: Data correctness
about: A wrong hazard number, a wrong page citation, or a wrong region mapping
title: ''
labels: data-correctness
assignees: ''
---

A wrong number or a wrong citation is a first-class bug in this project, not a
documentation nit. Fill in as much as you can. Partial reports are still useful.

## What kind of error

- [ ] Hazard number (return level, confidence interval, trend, risk band)
- [ ] Citation (the cited page does not support the claim, or points at the
      wrong page)
- [ ] Region mapping (coordinates mapped to the wrong IPCC AR6 region)
- [ ] Refusal error (answered something it should refuse, or refused something
      it supports)

## Location

Place name, and latitude / longitude if you have them.

## Hazard and horizon

Heat, extreme precipitation, or wind. Forecast horizon or return period used.

## Expected

What the correct value or citation is.

## Observed

What the agent returned. Paste the relevant part of the report, including the
`HazardStat` fields or the citation as printed.

## Source you checked

Where the correct value comes from: the Open-Meteo archive endpoint, ERA5
directly, a station record, or the IPCC AR6 page number. A link or the exact
page is what makes this actionable.

## Eval slice, if known

One of: `single_page`, `multi_page`, `regional_table`, `premise_injection`,
`duplicate_region`, `out_of_scope_hazard`, `out_of_corpus`.

## Reproduction

The question or API call that produced it, and the commit
(`git rev-parse --short HEAD`).

## Already covered?

Check [LIMITATIONS.md](../../LIMITATIONS.md) first. Some gaps are known and
documented, for example ERA5 underestimating convective wind gusts, or figures
and maps not being read. If it is listed there but the wording is wrong or too
weak, say so here and that file gets fixed.
