"""Pull arXiv paper titles (public arXiv API) for a local stand-in of the live `subset_arxiv`.

The live round's 4th subset is arXiv text short enough to take the leader's short-text path
(median < 130 chars, no newlines), i.e. titles. Categories span every archive so subsets can
mix fields the way a broad arXiv sample would.
    python tools/fetch_arxiv.py                  # data/arxiv/titles.jsonl
    python tools/fetch_arxiv.py --per-cat 500
"""
import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "arxiv"
API = "http://export.arxiv.org/api/query?"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

CATS = [
    "cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.CR", "cs.RO", "cs.DS", "cs.NI", "cs.SE", "cs.HC",
    "cs.IR", "cs.DB", "cs.DC", "cs.GT", "cs.IT", "cs.SI", "cs.CY", "cs.PL", "cs.LO", "cs.SD",
    "math.AP", "math.PR", "math.CO", "math.AG", "math.NT", "math.OC", "math.DG", "math.NA", "math.ST", "math.GR",
    "physics.optics", "physics.flu-dyn", "physics.chem-ph", "physics.soc-ph", "physics.bio-ph", "physics.ins-det",
    "cond-mat.mtrl-sci", "cond-mat.str-el", "cond-mat.mes-hall", "cond-mat.soft", "cond-mat.supr-con", "cond-mat.stat-mech",
    "astro-ph.GA", "astro-ph.CO", "astro-ph.SR", "astro-ph.HE", "astro-ph.EP", "astro-ph.IM",
    "hep-th", "hep-ph", "hep-ex", "hep-lat", "gr-qc", "quant-ph", "nucl-th", "nlin.CD",
    "q-bio.NC", "q-bio.QM", "q-bio.BM", "q-fin.ST", "q-fin.PM", "stat.ML", "stat.ME", "stat.AP",
    "eess.SP", "eess.IV", "eess.SY", "eess.AS", "econ.EM", "econ.GN",
]


def fetch(cat, n):
    q = urllib.parse.urlencode({"search_query": f"cat:{cat}", "start": 0, "max_results": n,
                                "sortBy": "submittedDate", "sortOrder": "descending"})
    with urllib.request.urlopen(API + q, timeout=120) as r:
        root = ET.fromstring(r.read())
    out = []
    for e in root.findall("a:entry", NS):
        title = re.sub(r"\s+", " ", e.findtext("a:title", "", NS)).strip()
        prim = e.find("arxiv:primary_category", NS)
        out.append({"id": e.findtext("a:id", "", NS).rsplit("/", 1)[-1], "title": title,
                    "cat": cat, "primary": prim.get("term") if prim is not None else cat})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-cat", type=int, default=1000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "titles.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["cat"] for l in open(path, encoding="utf-8")}
    with open(path, "a", encoding="utf-8") as f:
        for cat in CATS:
            if cat in done:
                continue
            for attempt in range(3):
                try:
                    rows = fetch(cat, a.per_cat)
                    break
                except Exception as e:
                    print(f"{cat}: {e!r}, retrying", flush=True)
                    time.sleep(10)
            else:
                continue
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"{cat}: {len(rows)}", flush=True)
            time.sleep(3.5)  # arXiv API etiquette: one request every 3 s


if __name__ == "__main__":
    main()
