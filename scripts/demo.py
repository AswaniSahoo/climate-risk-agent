"""Live end-to-end demo — run the agent against the REAL Open-Meteo API.

Unlike the tests (which mock HTTP for determinism), this makes a real network
call so you can see a genuine RiskReport from today's forecast.

Run:  uv run python -m scripts.demo
"""
from agent.contracts import Hazard
from agent.graph import run_agent


def main() -> None:
    report = run_agent(
        location="Rourkela",
        latitude=22.26,
        longitude=84.85,
        hazard=Hazard.EXTREME_PRECIP,
        horizon_days=7,
    )
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
