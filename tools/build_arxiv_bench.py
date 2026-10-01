"""Local stand-in for the live `subset_arxiv` (arXiv titles, labelled like the validator).

Live arXiv subsets (rounds 49-68 eval metadata): 5000 texts, 33-47 clusters, 26-30% noise,
median cluster ~50, largest 400-540. Run in the ground-truth env (.venv-gt).
    .venv-gt/bin/python tools/build_arxiv_bench.py embed        # data/arxiv/pool.jsonl + pool.emb.npy
    .venv-gt/bin/python tools/build_arxiv_bench.py make --first 80 --n 8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_bench import gt_labels  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ARX = ROOT / "data" / "arxiv"
BENCH = ROOT / "data" / "bench"
SIZE = 5000


def embed(_):
    from sentence_transformers import SentenceTransformer
    seen, rows = set(), []
    for line in open(ARX / "titles.jsonl", encoding="utf-8"):
        r = json.loads(line)
        key = r["title"].lower()
        if r["id"] in seen or key in seen or not r["title"]:
            continue
        seen.update((r["id"], key))
        rows.append(r)
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    emb = model.encode([r["title"] for r in rows], batch_size=128, show_progress_bar=True, convert_to_numpy=True)
    with open(ARX / "pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.save(ARX / "pool.emb.npy", emb.astype(np.float32))
    print(f"pool: {len(rows)} titles")


def make(a):
    rows = [json.loads(l) for l in open(ARX / "pool.jsonl", encoding="utf-8")]
    emb = np.load(ARX / "pool.emb.npy")
    archive = np.array([r[a.group] if a.group == "cat" else r["primary"].split(".")[0] for r in rows])
    rng = np.random.default_rng(a.seed)
    for s in range(a.first, a.first + a.n):
        # Mix archives with random weights so subsets differ in field balance, like independent draws would.
        arcs = np.unique(archive)
        if a.mix:  # fixed archive shares (arXiv all-time mix), jittered per subset
            share = dict((k, float(v)) for k, v in (x.split("=") for x in a.mix.split(",")))
            w = np.array([share.get(c, 0.0) for c in arcs]) * rng.uniform(0.7, 1.3, len(arcs))
            w = w / np.maximum(np.bincount(np.searchsorted(arcs, archive), minlength=len(arcs)), 1)
        else:
            w = rng.dirichlet(np.full(len(arcs), a.alpha))
        p = w[np.searchsorted(arcs, archive)]
        p /= p.sum()
        pick = rng.choice(len(rows), SIZE, replace=False, p=p)
        lab = gt_labels(emb[pick], min_samples=10)  # same calibration as build_bench.py (live GT stats)
        with open(BENCH / f"subset_{s:02d}.jsonl", "w", encoding="utf-8") as f:
            for i, l in zip(pick, lab):
                f.write(json.dumps({"text": rows[i]["title"], "label": int(l), "cat": rows[i]["primary"]}, ensure_ascii=False) + "\n")
        np.save(BENCH / f"subset_{s:02d}.emb.npy", emb[pick])
        ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
        print(f"subset_{s:02d}: k={len(ids)} noise={(lab == -1).mean():.2f} median={int(np.median(cnt))} max={cnt.max()}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["embed", "make"])
    ap.add_argument("--first", type=int, default=80)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=2.0, help="Dirichlet concentration over groups")
    ap.add_argument("--group", choices=["archive", "cat"], default="archive")
    ap.add_argument("--mix", default="", help="archive=share,... (overrides --alpha)")
    a = ap.parse_args()
    {"embed": embed, "make": make}[a.cmd](a)
