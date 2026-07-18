"""Author the MP+OSH batch: quotes are EXTRACTED from live parse between short
anchors, never transcribed — kills the spacing-trap class of failures.

Each candidate: (id, slice, question, hazard, gold [(source, page)...],
quote_spec (source, page, start_anchor, end_anchor)). The banked quote is
text[find(start) : find(end)+len(end)] from the live extract_pages output.
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
    assert i >= 0, f"start anchor not on {source} p{page}: {start!r}"
    j = text.find(end, i)
    assert j >= 0, f"end anchor not after start on {source} p{page}: {end!r}"
    q = text[i : j + len(end)]
    assert len(q) < 600, f"quote too long ({len(q)}) on {source} p{page}"
    return q


# (id, slice, question, hazard, gold_pages, quote_page, start, end, behavior)
CANDIDATES = [
    # ---- OSH2: evidence exists, system must refuse by scope ----
    ("OSH2-01", "out_of_scope_hazard",
     "Has India experienced a long-term change in meteorological and hydrological droughts since the late 19th century, and which regions show intensified drought?",
     "drought", [(CH12, 35)], (CH12, 35),
     "There was no observed", "north-west India", "refuse"),
    ("OSH2-02", "out_of_scope_hazard",
     "Has human-induced climate change contributed to increases in agricultural and ecological droughts?",
     "drought", [(SPM, 8)], (SPM, 8),
     "Human-induced climate change has contributed to increases in agricultural and ecological droughts",
     "ecological droughts", "refuse"),
    ("OSH2-03", "out_of_scope_hazard",
     "How does the rate of global mean sea level rise since 1900 compare with previous centuries?",
     "sea_level_rise", [(SPM, 8)], (SPM, 8),
     "Global mean sea level has risen faster", "3000 years", "refuse"),
    ("OSH2-04", "out_of_scope_hazard",
     "How much is relative sea level projected to rise in the oceans around Asia by 2081-2100 under low versus high emissions scenarios?",
     "sea_level_rise", [(CH12, 37)], (CH12, 37),
     "Relative sea level rise is very likely to continue", "SSP5-8.5", "refuse"),
    ("OSH2-05", "out_of_scope_hazard",
     "Has the number of intense tropical cyclones in the Bay of Bengal increased since the mid-1980s?",
     "tropical_cyclone", [(CH12, 35)], (CH12, 35),
     "There was an increase in the number and intensification rate", "mid-1980s", "refuse"),
    ("OSH2-06", "out_of_scope_hazard",
     "Has fire weather - compound hot, dry and windy conditions - become more frequent in some regions?",
     "fire_weather", [(CH11, 20)], (CH11, 20),
     "Medium confidence that fire weather", "some regions", "refuse"),
    ("OSH2-07", "out_of_scope_hazard",
     "How has the frequency of marine heatwaves changed since the 1980s, and what role did human influence play?",
     "marine_heatwave", [(SPM, 8)], (SPM, 8),
     "Marine heatwaves have approximately doubled in frequency", "2006", "refuse"),
    # OSH2-08/09/10 replaced 2026-07-18: the originals (snowline/sea-ice/glacier)
    # were cryosphere BACKGROUND SCIENCE mis-labelled as refuse — the held-out
    # e2e (1st exposure) exposed the system correctly answering them (grounded,
    # cited), which the matrix scored as false_answer. Replaced with clean
    # unsupported-HAZARD questions the scope guard genuinely refuses.
    ("OSH2-08", "out_of_scope_hazard",
     "In how many regions, and on which continents, are agricultural and ecological droughts projected to increase relative to 1850-1900?",
     "drought", [(SPM, 24)], (SPM, 24),
     "agricultural and ecological droughts are projected in a few regions", "except Asia", "refuse"),
    ("OSH2-09", "out_of_scope_hazard",
     "How much more often are extreme sea level events that historically occurred once per century projected to occur due to relative sea level rise?",
     "sea_level_rise", [(SPM, 25)], (SPM, 25),
     "extreme sea level events that occurred once per century", "tide gauge locations", "refuse"),
    ("OSH2-10", "out_of_scope_hazard",
     "By how much is the proportion of the most intense (Category 4-5) tropical cyclones projected to change at the global scale?",
     "tropical_cyclone", [(SPM, 16)], (SPM, 16),
     "The proportion of intense tropical cyclones", "global scale", "refuse"),
    # ---- MP2: same finding on multiple pages (any_of) ----
    ("MP2-01", "multi_page",
     "On which continents has heavy precipitation likely increased at the continental scale?",
     "extreme_precip", [(CH11, 6), (CH11, 48)], (CH11, 6),
     "Heavy precipitation has likely increased on the continental scale", "Asia", "answer"),
    ("MP2-02", "multi_page",
     "How have the frequency and intensity of heavy precipitation events changed at the global scale and across land regions?",
     "extreme_precip", [(CH11, 6), (CH11, 48), (CH11, 98)], (CH11, 98),
     "The frequency and intensity of heavy precipitation events have increased", "land regions", "answer"),
    ("MP2-03", "multi_page",
     "How is the frequency of rare heavy precipitation events (such as 10-year and 50-year events) projected to change at 4 degrees of global warming?",
     "extreme_precip", [(CH11, 6), (CH11, 55)], (CH11, 6),
     "The increase in the frequency of heavy precipitation events will be non-linear",
     "global warming", "answer"),
    ("MP2-04", "multi_page",
     "What factors cause projected increases in extreme precipitation intensity to vary between regions?",
     "extreme_precip", [(CH11, 6), (CH11, 55)], (CH11, 6),
     "Increases in the intensity of extreme precipitation at regional scales will vary",
     "storm dynamics", "answer"),
    ("MP2-05", "multi_page",
     "What is the main driver of observed changes in hot and cold extremes, and with what confidence globally and by continent?",
     "heatwave", [(CH11, 5), (CH11, 20)], (CH11, 5),
     "Human-induced greenhouse gas forcing is the main driver", "most continents", "answer"),
    ("MP2-06", "multi_page",
     "As global warming increases, what happens to the occurrence of extreme events that are unprecedented in the observed record?",
     None, [(CH11, 5), (CH11, 98)], (CH11, 5),
     "The occurrence of extreme events unprecedented in the observed record",
     "increasing global warming", "answer"),
    ("MP2-07", "multi_page",
     "Could some recent hot extreme events have occurred without human influence on the climate system?",
     "heatwave", [(CH11, 5), (CH11, 99)], (CH11, 5),
     "Some recent hot extreme events would have been extremely unlikely", "climate system", "answer"),
    ("MP2-08", "multi_page",
     "Have changes in anthropogenic aerosol concentrations affected trends in hot extremes in some regions?",
     "heatwave", [(CH11, 5), (CH11, 36)], (CH11, 5),
     "Changes in anthropogenic aerosol concentrations have likely affected trends in hot extremes",
     "some regions", "answer"),
    ("MP2-09", "multi_page",
     "Have irrigation and crop expansion influenced summer hot extremes in some regions such as the Midwestern USA?",
     "heatwave", [(CH11, 5), (CH11, 36)], (CH11, 5),
     "Irrigation and crop expansion have attenuated increases in summer hot extremes",
     "Midwestern USA", "answer"),
    ("MP2-10", "multi_page",
     "How will the number of hot days and hot nights and the frequency of heatwaves change over most land areas with warming?",
     "heatwave", [(CH11, 6), (CH11, 44)], (CH11, 6),
     "The number of hot days and hot nights and the length, frequency",
     "most land areas", "answer"),
    ("MP2-11", "multi_page",
     "In which regions is the temperature of the hottest days projected to increase fastest relative to global warming?",
     "heatwave", [(CH11, 6), (CH11, 21)], (CH11, 6),
     "The highest increase of temperature of hottest days is projected",
     "South American Monsoon region", "answer"),
    ("MP2-12", "multi_page",
     "Where is the temperature of the coldest days projected to increase fastest, and by how much relative to global warming?",
     "heatwave", [(CH11, 6), (CH11, 21), (CH11, 45), (SPM, 15)], (CH11, 6),
     "The highest increase of temperature of coldest days is projected in Arctic regions",
     "rate of global warming", "answer"),
    ("MP2-13", "multi_page",
     "With what confidence is the increase in frequency and intensity of heavy precipitation assessed across continents and AR6 regions?",
     "extreme_precip", [(CH11, 6), (CH11, 54)], (CH11, 6),
     "The increase in frequency and intensity is extremely likely for most continents",
     "most AR6 regions", "answer"),
    ("MP2-14", "multi_page",
     "At the global scale, what determines the rate at which heavy precipitation intensifies as the climate warms?",
     "extreme_precip", [(CH11, 6), (CH11, 54)], (CH11, 6),
     "the intensification of heavy precipitation will follow the rate of increase in the maximum amount of moisture",
     "as it warms", "answer"),
    ("MP2-15", "multi_page",
     "Can climate models capture the large-scale spatial distribution of precipitation extremes over land?",
     "extreme_precip", [(CH11, 5), (CH11, 50)], (CH11, 5),
     "Models are able to capture the large-scale spatial distribution of precipitation extremes",
     "over land", "answer"),
]


def main() -> None:
    out = []
    failures = []
    for (qid, slc, question, hazard, gold, qspec, start, end, behavior) in CANDIDATES:
        try:
            q_text = quote(qspec[0], qspec[1], start, end)
        except AssertionError as exc:
            failures.append(f"{qid}: {exc}")
            continue
        out.append({
            "id": qid, "slice": slc, "question": question,
            "answerable": True, "expected_behavior": behavior,
            "gold_pages": [{"source": s, "page": p} for s, p in gold],
            "supporting_quote": q_text,
            "hazard": hazard, "notes": "",
        })
    path = Path(__file__).parent / "v2_mp_osh.json"
    existing = {}
    if path.exists():
        existing = {q["id"]: q for q in json.loads(path.read_text(encoding="utf-8"))["questions"]}
    for q in out:
        existing[q["id"]] = q
    path.write_text(
        json.dumps({"questions": list(existing.values())}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"banked {len(out)} questions -> {path.name} (total {len(existing)})")
    if failures:
        print("FAILURES:", *failures, sep="\n  ")


if __name__ == "__main__":
    main()
