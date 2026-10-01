"""Edge-case smoke test for a text_clustering solution (run through tools/sandboxed_eval.sh with TOOL=edge_test.py).
Checks: no exception, one int label per input, and timing, on degenerate and mixed inputs.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import random
import sys
import time

from bench_eval import BENCH, load_solution

random.seed(0)
reddit = [json.loads(l)["text"] for l in open(BENCH / "subset_30.jsonl", encoding="utf-8")]
arxiv = [json.loads(l)["text"] for l in open(BENCH / "subset_80.jsonl", encoding="utf-8")]
CASES = {
    "1 text": ["hello world"],
    "2 texts": ["a", "b"],
    "5 texts": reddit[:5],
    "30 short": arxiv[:30],
    "60 long": reddit[:60],
    "all empty": [""] * 50,
    "emoji/links only": ["😀😀", "https://x.com/a", "@user", "#", "!!!"] * 20,
    "mixed scripts": reddit[:300] + ["これは日本語のテキストです"] * 40 + ["Это русский текст про новости"] * 40 + ["مرحبا بالعالم"] * 30,
    "duplicates": [arxiv[0]] * 200 + arxiv[1:300],
    "arxiv 5000": arxiv,
    "reddit 5000": reddit,
}
mod = load_solution(os.path.abspath(sys.argv[1]))
ok = True
for name, texts in CASES.items():
    t0 = time.perf_counter()
    try:
        out = mod.cluster_texts(list(texts))
        good = len(out) == len(texts) and all(isinstance(x, int) for x in out)
        print(f"{'OK  ' if good else 'BAD '} {name:18s} n={len(texts):5d} labels={len(set(out)):5d} {time.perf_counter() - t0:6.1f}s", flush=True)
        ok &= good
    except Exception as e:
        print(f"FAIL {name:18s} {e!r}", flush=True)
        ok = False
print("ALL OK" if ok else "PROBLEMS FOUND")
