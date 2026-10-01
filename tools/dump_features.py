"""Run a leader-lineage solution on benchmark subsets and save the feature matrix it clusters.

Hooks the solution's final distance probe (B0 in the long-text path, BU in the short-text path),
which receives the fully built feature matrix right before agglomerative clustering.
Run it through tools/sandboxed_eval.sh (TOOL=dump_features.py) because it imports third-party code.

Output: $OUT/feat_<solution>_<subset>.npz with G (float32), pred (final labels), regime.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from bench_eval import BENCH, load_solution, parse_range


def run(args):
    path, subset, out = args
    mod = load_solution(path)
    rows = [json.loads(l) for l in open(BENCH / f"subset_{subset:02d}.jsonl", encoding="utf-8")]
    texts = [r["text"] for r in rows]
    cap = {}
    # short path components (leader lineage): BL = BM25 word+char SVD, BM = PPMI word vectors, BP = distilled BD
    # long path components: t = tfidf SVD, Aw = PPMI word vectors, Ay = distilled Ak, Az = smoothing, A_ = spectral
    for name in ("BL", "BM", "BP", "t", "Aw", "Ay", "Az", "A_"):
        if hasattr(mod, name):
            orig = getattr(mod, name)

            def comp(*a, _orig=orig, _name=name, **k):
                r = _orig(*a, **k)
                if r is not None and "comp_" + _name not in cap:  # first call only (the solution's own)
                    cap["comp_" + _name] = np.asarray(r, np.float32)
                return r
            setattr(mod, name, comp)
    for name, regime in (("B0", "long"), ("BU", "short")):
        orig = getattr(mod, name)

        def hook(Z, *a, _orig=orig, _regime=regime, **k):
            cap["G"], cap["regime"] = np.asarray(Z, np.float32).copy(), _regime
            return _orig(Z, *a, **k)
        setattr(mod, name, hook)
    pred = np.asarray(mod.cluster_texts(texts))
    # rows of G: long path drops texts that normalise to empty; short path keeps all texts
    keep = np.array([len(mod.Ar(t)) > 0 for t in texts]) if cap.get("regime") == "long" else np.ones(len(texts), bool)
    extra = {k: v for k, v in cap.items() if k.startswith("comp_")}
    if hasattr(mod, "Ay") and hasattr(mod, "BP"):  # leader lineage: the two distilled projections, for all texts
        extra["proj_ak"] = np.asarray(mod.Ay([mod.Ar(t) or "x" for t in texts]), np.float32)
        extra["proj_bd"] = np.asarray(mod.BP(texts, True), np.float32)
    dest = Path(out) / f"feat_{Path(path).stem}_{subset:02d}.npz"
    np.savez_compressed(dest, G=cap.get("G", np.zeros((0, 0), np.float32)), pred=pred, keep=keep,
                        regime=cap.get("regime", "none"), **extra)
    return subset, cap.get("regime"), cap.get("G", np.zeros((0, 0))).shape


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("solution")
    ap.add_argument("--subsets", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = os.environ.get("OUT", ".")
    jobs = [(str(Path(a.solution).resolve()), s, out) for s in parse_range(a.subsets)]
    with ProcessPoolExecutor(a.workers) as ex:
        for s, regime, shape in ex.map(run, jobs):
            print(f"subset {s:02d}: regime={regime} G={shape}", flush=True)
