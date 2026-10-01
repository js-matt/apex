"""Fast local scorer: imports each solution and calls cluster_texts() on every benchmark subset.

    python tools/bench_eval.py solutions/text_clustering/solution.py [more.py ...] [--subsets 0-23] [--workers 12]

Score per subset = (max(0, ARI) + NMI) / 2 against data/bench/subset_XX.jsonl "label"
(GT noise -1 is kept as its own label, as sklearn does). Workers are single-threaded, like the sandbox.
"""
import argparse
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import importlib.util
import resource
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

BENCH = Path(__file__).resolve().parent.parent / "data" / "bench"
_MODS = {}


def load_solution(path):
    if path not in _MODS:
        spec = importlib.util.spec_from_file_location(f"sol_{len(_MODS)}", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _MODS[path] = mod
    return _MODS[path]


def run_one(args):
    path, subset = args
    rows = [json.loads(l) for l in open(BENCH / f"subset_{subset:02d}.jsonl", encoding="utf-8")]
    perm = np.arange(len(rows))
    if os.environ.get("SHUFFLE"):  # reproducible input-order permutation (texts and both label sets move together)
        perm = np.random.default_rng(int(os.environ["SHUFFLE"]) * 1000 + subset).permutation(len(rows))
        rows = [rows[i] for i in perm]
    texts, gt = [r["text"] for r in rows], np.array([r.get("label", 0) for r in rows])  # unlabelled: predictions only (SAVE_PRED)
    mod = load_solution(path)
    t0 = time.perf_counter()
    pred = np.asarray(mod.cluster_texts(texts))
    dt = time.perf_counter() - t0
    ari, nmi = adjusted_rand_score(gt, pred), normalized_mutual_info_score(gt, pred)
    alt = BENCH / f"subset_{subset:02d}.alt_ms3.npy"  # optional lower-noise GT (min_samples=3)
    if alt.exists():
        gb = np.load(alt)[perm]
        scb = (max(0.0, adjusted_rand_score(gb, pred)) + normalized_mutual_info_score(gb, pred)) / 2
    else:
        scb = float("nan")
    if os.environ.get("SAVE_PRED"):
        tag = f"_sh{os.environ['SHUFFLE']}" if os.environ.get("SHUFFLE") else ""
        out = np.empty_like(pred)
        out[perm] = pred  # store in original order so runs are comparable
        np.save(Path(os.environ["SAVE_PRED"]) / f"pred_{Path(path).stem}_{subset:02d}{tag}.npy", out)
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # peak MB of this worker (one job per worker)
    return path, subset, (max(0.0, ari) + nmi) / 2, ari, nmi, dt, len(np.unique(pred)), scb, rss


def parse_range(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("solutions", nargs="+")
    ap.add_argument("--subsets", default=None)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--json", help="append results as JSON lines here")
    ap.add_argument("-v", action="store_true", help="per-subset output")
    a = ap.parse_args()
    subsets = parse_range(a.subsets) if a.subsets else sorted(
        int(p.stem.split("_")[1].split(".")[0]) for p in BENCH.glob("subset_*.jsonl"))
    paths = [str(Path(p).resolve()) for p in a.solutions]
    for p in paths:
        n = len(Path(p).read_text(encoding="utf-8"))
        if n >= 50_000:
            print(f"WARNING {p}: {n} chars (limit 50,000)")
    jobs = [(p, s) for p in paths for s in subsets]
    res = {p: {} for p in paths}
    with ProcessPoolExecutor(a.workers, max_tasks_per_child=1) as ex:
        for p, s, sc, ari, nmi, dt, k, scb, rss in ex.map(run_one, jobs):
            res[p][s] = (sc, ari, nmi, dt, k, scb, rss)
            if a.v:
                print(f"  {Path(p).name} subset {s:02d}: {sc:.4f} ari={ari:.3f} nmi={nmi:.3f} {dt:.1f}s k={k}", flush=True)
    for p in paths:
        r = res[p]
        sc = np.array([r[s][0] for s in subsets])
        print(f"{np.mean(sc):.4f}  B={np.nanmean([r[s][5] for s in subsets]):.4f}  ari={np.mean([r[s][1] for s in subsets]):.4f} nmi={np.mean([r[s][2] for s in subsets]):.4f} "
              f"maxT={max(r[s][3] for s in subsets):.0f}s maxRSS={max(r[s][6] for s in subsets):.0f}MB  {Path(p).name}")
        if a.json:
            with open(a.json, "a") as f:
                f.write(json.dumps({"path": p, "scores": {s: r[s] for s in subsets}}) + "\n")


if __name__ == "__main__":
    sys.exit(main())
