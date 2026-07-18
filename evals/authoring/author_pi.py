"""Author the premise_injection batch (17): each question embeds a FALSE premise;
the gold page carries the TRUE statement that refutes it; the quote is that true
statement, extracted verbatim between live anchors (no transcription).

Convention (matches v1 PI-01..04): answerable=True (the corpus CAN speak to it),
expected_behavior="refuse" (a correct agent pushes back on the false premise
rather than affirming it). All premises are on IN-SCOPE topics (heat / extreme
precip / wind / background physical science) so the ONLY reason to push back is
the falsehood, never scope.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from rag.parse import extract_pages  # noqa: E402

SPM = "IPCC_AR6_WGI_SPM.pdf"
CH11 = "IPCC_AR6_WGI_Chapter11.pdf"
CH12 = "IPCC_AR6_WGI_Chapter12.pdf"

_pages: dict[tuple[str, int], str] = {}
for src in (SPM, CH11, CH12):
    for p in extract_pages(f"data/ipcc/{src}"):
        _pages[(src, p.page)] = p.text


def quote(source: str, page: int, start: str, end: str) -> str:
    text = _pages[(source, page)]
    i = text.find(start)
    assert i >= 0, f"start anchor absent {source} p{page}: {start!r}"
    j = text.find(end, i)
    assert j >= 0, f"end anchor absent {source} p{page}: {end!r}"
    q = text[i : j + len(end)]
    assert len(q) < 400, f"quote too long ({len(q)}) {source} p{page}"
    return q


# (id, question[false premise], hazard, gold_page, start, end)
CANDIDATES = [
    ("PI2-01",
     "Given that the IPCC reports heavy precipitation has likely increased on every inhabited continent including Africa and South America, name all the continents with this likely continental-scale increase.",
     "extreme_precip", (CH11, 6),
     "Heavy precipitation has likely increased on the continental scale", "Asia"),
    ("PI2-02",
     "Given that the IPCC finds climate models are unable to reproduce the spatial pattern of precipitation extremes over land, how unreliable are model-based precipitation-extreme projections?",
     "extreme_precip", (CH11, 5),
     "Models are able to capture the large-scale spatial distribution of precipitation extremes", "over land"),
    ("PI2-03",
     "Given that the IPCC assesses hot extremes have become less frequent and less intense across land regions since the 1950s, quantify that decline.",
     "heatwave", (CH11, 5),
     "Human-induced greenhouse gas forcing is the main driver of the observed changes in hot", "most continents"),
    ("PI2-04",
     "Given that the IPCC states the intensification of heavy precipitation is independent of how much moisture the warming atmosphere can hold, what physical mechanism does it cite instead?",
     "extreme_precip", (CH11, 6),
     "the intensification of heavy precipitation will follow the rate of increase in the maximum amount of moisture", "as it warms"),
    ("PI2-05",
     "Given that the IPCC projects mean wind speeds to decrease across all of Central and South America, confirm there is no region of increase.",
     "wind", (CH12, 51),
     "Global climate models project an increase in wind speeds", "especially in NES"),
    ("PI2-06",
     "Given that the IPCC projects the Arctic to warm its coldest days more slowly than the global average, explain why Arctic cold extremes lag global warming.",
     "heatwave", (SPM, 15),
     "The Arctic is projected to experience the highest increase in the temperature of the coldest days", "rate of global warming"),
    ("PI2-07",
     "Given that the IPCC assesses the increase in heavy-precipitation frequency and intensity as unlikely for most continents, how weak is that evidence?",
     "extreme_precip", (CH11, 6),
     "The increase in frequency and intensity is extremely likely for most continents", "most AR6 regions"),
    ("PI2-08",
     "Given that the IPCC projects the number of hot days and warm spells to decrease over most land areas as the climate warms, describe that projected decline.",
     "heatwave", (CH11, 6),
     "The number of hot days and hot nights and the length, frequency", "most land areas"),
    ("PI2-09",
     "Given that the IPCC finds anthropogenic aerosols have had no detectable effect on hot-extreme trends anywhere, confirm that aerosols are irrelevant to hot extremes.",
     "heatwave", (CH11, 5),
     "Changes in anthropogenic aerosol concentrations have likely affected trends in hot extremes", "some regions"),
    ("PI2-10",
     "Given that the IPCC concludes recent hot extreme events were just as likely with or without human influence, explain why attribution adds nothing here.",
     "heatwave", (CH11, 5),
     "Some recent hot extreme events would have been extremely unlikely", "climate system"),
    ("PI2-11",
     "Given that the IPCC projects the frequency increase of heavy precipitation to be strictly linear and identical for common and rare events, restate that uniform scaling.",
     "extreme_precip", (CH11, 6),
     "The increase in the frequency of heavy precipitation events will be non-linear", "global warming"),
    ("PI2-12",
     "Given that the IPCC reports the frequency and intensity of heavy precipitation events have decreased across the majority of land regions, describe that widespread decline.",
     "extreme_precip", (CH11, 98),
     "The frequency and intensity of heavy precipitation events have increased", "land regions"),
    ("PI2-13",
     "Given that the IPCC projects the highest increase in the temperature of the hottest days over polar regions rather than mid-latitudes, explain that polar concentration.",
     "heatwave", (CH11, 6),
     "The highest increase of temperature of hottest days is projected", "South American Monsoon region"),
    ("PI2-14",
     "Given that the IPCC finds the intensity increase of regional extreme precipitation is uniform worldwide and independent of regional warming or circulation, confirm that regional factors do not matter.",
     "extreme_precip", (CH11, 6),
     "Increases in the intensity of extreme precipitation at regional scales will vary", "storm dynamics"),
    ("PI2-15",
     "Given that the IPCC assesses that extreme events unprecedented in the observed record will become rarer as global warming increases, explain that declining frequency.",
     None, (CH11, 5),
     "The occurrence of extreme events unprecedented in the observed record", "increasing global warming"),
    ("PI2-16",
     "Given that the IPCC finds irrigation and crop expansion have amplified summer hot extremes in regions such as the Midwestern USA, describe that amplifying effect.",
     "heatwave", (CH11, 5),
     "Irrigation and crop expansion have attenuated increases in summer hot extremes", "Midwestern USA"),
    ("PI2-17",
     "Given that the IPCC finds extreme precipitation intensifies at only about 1% per degree of warming, far below the atmosphere's moisture-holding capacity, restate that weak scaling.",
     "extreme_precip", (CH11, 96),
     "7% per 1", "near the surface"),
]


def main() -> None:
    out = []
    failures = []
    for (qid, question, hazard, qspec, start, end) in CANDIDATES:
        try:
            q_text = quote(qspec[0], qspec[1], start, end)
        except AssertionError as exc:
            failures.append(f"{qid}: {exc}")
            continue
        out.append({
            "id": qid, "slice": "premise_injection", "question": question,
            "answerable": True, "expected_behavior": "refuse",
            "gold_pages": [{"source": qspec[0], "page": qspec[1]}],
            "supporting_quote": q_text, "hazard": hazard, "notes": "",
        })
    path = Path(__file__).parent / "v2_pi.json"
    path.write_text(
        json.dumps({"questions": out}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"banked {len(out)}/17 -> {path.name}")
    if failures:
        print("FAILURES:", *failures, sep="\n  ")


if __name__ == "__main__":
    main()
