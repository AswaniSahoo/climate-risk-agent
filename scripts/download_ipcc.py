"""Fetch the IPCC AR6 WG1 documents we index: the SPM + Chapter 11 (Extremes).

Chapter 11 ("Weather and Climate Extreme Events in a Changing Climate") is the
chapter that covers heat, precipitation and wind extremes — exactly our three
hazards. The SPM supplies the headline assessed statements.

PDFs land in data/ (git-ignored: large, and best fetched from the source).

Run:  uv run python -m scripts.download_ipcc
"""
from pathlib import Path

import httpx

BASE = "https://www.ipcc.ch/report/ar6/wg1/downloads/report"
DOCS = {
    "IPCC_AR6_WGI_SPM.pdf": f"{BASE}/IPCC_AR6_WGI_SPM.pdf",
    "IPCC_AR6_WGI_Chapter11.pdf": f"{BASE}/IPCC_AR6_WGI_Chapter11.pdf",
}
DEST = Path("data/ipcc")


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    for name, url in DOCS.items():
        target = DEST / name
        if target.exists():
            print(f"skip {name} (exists, {target.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"downloading {name} ...")
        with httpx.stream("GET", url, follow_redirects=True, timeout=180) as response:
            response.raise_for_status()
            with target.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        print(f"  -> {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
