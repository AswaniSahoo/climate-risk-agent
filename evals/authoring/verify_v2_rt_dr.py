"""Self-verification for evals/authoring/v2_rt_dr.json.

Loads the JSON, validates against evals.schema.EvalQuestion, parses the three
IPCC PDFs with rag.parse.extract_pages, and asserts every supporting_quote is
an exact substring of the text of at least one of its gold pages.
"""
import json
import sys

sys.path.insert(0, ".")

from evals.schema import EvalSet
from rag.parse import extract_pages

PDF_DIR = "data/ipcc/"

with open("evals/authoring/v2_rt_dr.json", encoding="utf-8") as f:
    raw = json.load(f)

eval_set = EvalSet(**raw)
print(f"Schema validation OK: {len(eval_set.questions)} questions parsed.")

# Cache extracted pages per source file: {source: {page_num: text}}
page_cache: dict = {}


def pages_for(source: str) -> dict[int, str]:
    if source not in page_cache:
        pages = extract_pages(PDF_DIR + source)
        page_cache[source] = {p.page: p.text for p in pages}
    return page_cache[source]


failures = []
passes = 0

for q in eval_set.questions:
    if not q.supporting_quote:
        continue
    found_on = None
    checked = []
    for ref in q.gold_pages:
        text = pages_for(ref.source).get(ref.page)
        checked.append((ref.source, ref.page, text is not None))
        if text is not None and q.supporting_quote in text:
            found_on = ref
            break
    if found_on is None:
        failures.append((q.id, q.supporting_quote, checked))
    else:
        passes += 1

total = passes + len(failures)
print(f"\n{passes}/{total} supporting_quote checks PASS")

if failures:
    print("\nFAILURES:")
    for qid, quote, checked in failures:
        print(f"  {qid}: quote not found on any gold page {checked}")
        print(f"    quote: {quote[:120]}...")
    sys.exit(1)
else:
    print("ALL PASS")
