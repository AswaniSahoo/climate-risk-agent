"""Live end-to-end demo — run the agent against real data.

Unlike the tests (which mock HTTP), this makes real network calls: a live
Open-Meteo forecast AND the ERA5 climatology (return levels) that grounds the
report. Prints a genuine, provenance-rich RiskReport.

Run:  uv run python -m scripts.demo
"""
from agent.contracts import Hazard
from agent.graph import run_agent
from tools.climatology import ClimatologyError, climatology_hazard_stat

LOCATION, LAT, LON = "Rourkela", 22.26, 84.85
HAZARD = Hazard.EXTREME_PRECIP


def main() -> None:
    try:
        hazard_stat = climatology_hazard_stat(LAT, LON, HAZARD)
    except ClimatologyError as exc:  # stay useful if the archive is unreachable
        print(f"(ERA5 climatology unavailable, report will be forecast-only: {exc})")
        hazard_stat = None

    report = run_agent(
        location=LOCATION,
        latitude=LAT,
        longitude=LON,
        hazard=HAZARD,
        horizon_days=7,
        hazard_stat=hazard_stat,
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
