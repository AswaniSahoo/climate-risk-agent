"""Graph-path eval: run the REAL agent (run_agent) over live scenarios.

Why (evaluator gap #2, in part): the e2e eval exercises answer_with_guard over
gold questions — the RiskReport path itself (plan → call → research →
synthesize) had no numbers. This runner drives run_agent end-to-end (live
Open-Meteo + ERA5 + Vertex) and checks the CONTRACT-LEVEL invariants every
report must satisfy. It is an integration eval: cheap (~6 scenarios), run
before release next to the two question evals.

Checks per report (beyond what Pydantic already enforces):
- verdict basis: GEV return-level basis used whenever climatology is attached
  and variables match (heat/precip); wind must say absolute-thresholds
- citations: page-level, deduped, non-empty unless the answerer abstained
- confidence within the composed range [0.3, 0.75]
- provenance present

Run:  uv run python -m evals.run_graph_eval   (needs network + Gemini auth)
"""
from __future__ import annotations

from agent.contracts import Hazard
from agent.graph import run_agent
from tools.climatology import ClimatologyError, climatology_hazard_stat

_SCENARIOS = [
    ("Rourkela, South Asia", 22.26, 84.85, Hazard.HEATWAVE),
    ("Rourkela, South Asia", 22.26, 84.85, Hazard.EXTREME_PRECIP),
    ("Rourkela, South Asia", 22.26, 84.85, Hazard.WIND),
    ("Berlin, Western and Central Europe", 52.52, 13.40, Hazard.HEATWAVE),
    ("Berlin, Western and Central Europe", 52.52, 13.40, Hazard.EXTREME_PRECIP),
]


def main() -> None:
    from obs.log import configure

    configure()  # runner owns logging config
    from tqdm import tqdm

    failures: list[str] = []
    passed = 0

    progress = tqdm(_SCENARIOS, desc="graph", unit="scenario")
    for location, lat, lon, hazard in progress:
        label = f"{hazard.value}@{location.split(',')[0]}"
        try:
            stat = climatology_hazard_stat(lat, lon, hazard)
        except ClimatologyError as exc:
            progress.write(f"  {label}: climatology unavailable ({exc}); running without")
            stat = None

        report = run_agent(
            location=location, latitude=lat, longitude=lon,
            hazard=hazard, horizon_days=7, hazard_stat=stat,
        )

        checks: list[tuple[str, bool]] = []
        basis = next((d.detail for d in report.drivers if d.factor == "severity_basis"), "")
        if stat is not None and hazard is not Hazard.WIND:
            checks.append(("gev_basis_used", "GEV return levels" in basis))
            checks.append(("ci_present", all(
                r.ci_low is not None for r in report.hazard_stats[0].return_levels
            )))
        if stat is not None and hazard is Hazard.WIND:
            checks.append(("wind_declares_absolute_basis", "absolute thresholds" in basis))
        checks.append(("confidence_in_composed_range", 0.3 <= report.confidence <= 0.75))
        checks.append(("provenance_present", len(report.provenance) >= 1))
        checks.append(("citations_deduped", len(report.citations)
                       == len({(c.source, c.locator) for c in report.citations})))
        if report.citations:
            checks.append(("citations_page_level", all(
                c.locator.startswith("p") for c in report.citations
            )))

        bad = [name for name, ok in checks if not ok]
        if bad:
            failures.append(f"{label}: FAILED {bad}")
            progress.write(f"  {label}: FAIL {bad}")
        else:
            passed += 1
            cited = len(report.citations)
            level = report.risk_level.value if report.risk_level else "refused"
            progress.write(
                f"  {label}: ok — {level}, "
                f"confidence {report.confidence}, {cited} citations"
            )

    print(f"\ngraph-path eval: {passed}/{len(_SCENARIOS)} scenarios pass all checks")
    if failures:
        print("FAILURES:", *failures, sep="\n  ")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
