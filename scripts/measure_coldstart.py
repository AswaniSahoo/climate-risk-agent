"""Measure local cold start of a built image: time until Streamlit health endpoint answers.

Usage: uv run python scripts/measure_coldstart.py <image-tag> [port]
Prints image size, seconds to port open, seconds to first "/" 200, then removes the container.
"""
from __future__ import annotations

import subprocess
import sys
import time
from collections.abc import Sequence

import httpx

CONTAINER = "crg-coldstart"


def main(argv: Sequence[str] | None = None) -> int:
    """Run one cold-start measurement. Returns a process exit code.

    Wrapped in a function behind a `__main__` guard so importing this module
    (a test, a docs build, or anything that walks scripts/) does not start
    Docker containers as an import side effect.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(__doc__)
        return 2
    image = args[0]
    port = int(args[1]) if len(args) > 1 else 7861

    size = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    ).stdout.strip()
    print(
        f"image={image} size_bytes={size} size_gb={int(size) / 1e9:.2f}"
        if size.isdigit()
        else f"size unknown: {size}"
    )

    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    t0 = time.perf_counter()
    run = subprocess.run(
        ["docker", "run", "-d", "--name", CONTAINER, "-p", f"{port}:7860",
         "--cpus", "1", "--memory", "2g", image],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if run.returncode != 0:
        print("docker run failed:", run.stderr[:300])
        return 1

    health = f"http://127.0.0.1:{port}/_stcore/health"
    opened = None
    deadline = t0 + 600
    while time.perf_counter() < deadline:
        try:
            response = httpx.get(health, timeout=2)
            if response.status_code == 200:
                opened = time.perf_counter() - t0
                break
        except Exception:
            pass
        time.sleep(0.5)
    print(f"port_open_s={opened:.1f}" if opened else "port never opened within 600 s")

    if opened:
        t1 = time.perf_counter()
        try:
            response = httpx.get(f"http://127.0.0.1:{port}/", timeout=120)
            print(
                f"first_page_status={response.status_code} "
                f"first_page_s={time.perf_counter() - t1:.1f}"
            )
        except Exception as exc:
            print("first page failed:", type(exc).__name__)

    logs = subprocess.run(
        ["docker", "logs", CONTAINER],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    tail = (logs.stdout + logs.stderr).strip().splitlines()[-8:]
    print("--- container log tail ---")
    print("\n".join(tail))
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    return 0 if opened else 1


if __name__ == "__main__":
    raise SystemExit(main())
