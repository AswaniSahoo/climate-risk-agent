"""Measure local cold start of a built image: time until Streamlit health endpoint answers.

Usage: uv run python coldstart.py <image-tag> [port]
Prints image size, seconds to port open, seconds to first "/" 200, then removes the container.
"""
import subprocess
import sys
import time

import httpx

image = sys.argv[1]
port = int(sys.argv[2]) if len(sys.argv) > 2 else 7861

size = subprocess.run(
    ["docker", "image", "inspect", image, "--format", "{{.Size}}"],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
).stdout.strip()
print(f"image={image} size_bytes={size} size_gb={int(size) / 1e9:.2f}" if size.isdigit() else f"size unknown: {size}")

subprocess.run(["docker", "rm", "-f", "crg-coldstart"], capture_output=True)
t0 = time.perf_counter()
run = subprocess.run(
    ["docker", "run", "-d", "--name", "crg-coldstart", "-p", f"{port}:7860", "--cpus", "1", "--memory", "2g", image],
    capture_output=True, text=True, encoding="utf-8", errors="replace",
)
if run.returncode != 0:
    print("docker run failed:", run.stderr[:300])
    sys.exit(1)

health = f"http://127.0.0.1:{port}/_stcore/health"
opened = None
deadline = t0 + 600
while time.perf_counter() < deadline:
    try:
        r = httpx.get(health, timeout=2)
        if r.status_code == 200:
            opened = time.perf_counter() - t0
            break
    except Exception:
        pass
    time.sleep(0.5)
print(f"port_open_s={opened:.1f}" if opened else "port never opened within 600 s")

if opened:
    t1 = time.perf_counter()
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/", timeout=120)
        print(f"first_page_status={r.status_code} first_page_s={time.perf_counter() - t1:.1f}")
    except Exception as e:
        print("first page failed:", type(e).__name__)

logs = subprocess.run(["docker", "logs", "crg-coldstart"], capture_output=True, text=True, encoding="utf-8", errors="replace")
tail = (logs.stdout + logs.stderr).strip().splitlines()[-8:]
print("--- container log tail ---")
print("\n".join(tail))
subprocess.run(["docker", "rm", "-f", "crg-coldstart"], capture_output=True)
