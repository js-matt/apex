"""Re-implementation of the leader's short-path back-end (BV, L306-L317) on dumped features, plus variants.

Works on data/sandbox_out/feat/feat_<sol>_<ss>.npz (G = BV's smoothed feature matrix O). Trusted code only.
    .venv/bin/python tools/short_lab.py --subsets 80-87
First checks the re-implementation reproduces the leader's labels, then grids over variants:
  k      linkage cut (leader 130)
  sing   fraction of most-isolated points made singletons (leader .25)
  noise  rule for points labelled -1 as one group (leader: none)
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_lab import parse  # noqa: E402
from noise_lab import score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH, FEAT = ROOT / "data" / "bench", ROOT / "data" / "sandbox_out" / "feat"
BIG = 10 ** 6


def protect(T, lab, Z, n, cap=3, lo=.006, hi=.12, ms=20):
    """Leader's At(): the 3 most coherent mid-size clusters never lose points to singletons."""
    T = T.copy()
    cand = []
    for g in np.unique(lab):
        m = lab == g
        sz = int(m.sum())
        if sz < max(ms, int(lo * n)) or sz > int(hi * n):
            continue
        c = Z[m].mean(0)
        c /= np.linalg.norm(c) + 1e-12
        cand.append((sz * float((Z[m] @ c).mean()), m))
    cand.sort(key=lambda x: -x[0])
    for _, m in cand[:cap]:
        T[m] = -1.0
    return T


def backend(Z, L, D, k, sing, noise, kn=12):
    n = len(Z)
    lab = fcluster(L, k, "maxclust").astype(np.int64)
    T = protect(D[:, kn - 1], lab, Z, n)
    order = np.argsort(-T)
    out = lab.copy()
    rule, _, arg = noise.partition(":")
    nz = np.zeros(n, bool)
    if rule == "iso":  # most isolated fraction -> -1, the next `sing` -> singletons
        nz[order[:int(float(arg) * n)]] = True
        rest = order[int(float(arg) * n):]
        out[rest[:int(sing * n)]] = BIG + np.arange(len(rest[:int(sing * n)]))
    else:
        s_idx = order[:int(sing * n)]
        out[s_idx] = BIG + np.arange(len(s_idx))
        if rule == "small":  # members of clusters smaller than arg (before singletons) -> -1
            ids, inv, cnt = np.unique(lab, return_inverse=True, return_counts=True)
            nz = (cnt[inv] < int(arg)) & (out < BIG)
        elif rule == "single":  # the singletons themselves -> one -1 group
            nz = out >= BIG
    out[nz] = -1
    return out


def run(job):
    s, grid = job
    gt = np.array([json.loads(l)["label"] for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")])
    z = np.load(FEAT / f"feat_181049_{s:02d}.npz", allow_pickle=False)
    Z = z["G"]
    L = linkage(Z, "average", "cosine")
    D = NearestNeighbors(n_neighbors=31, metric="cosine").fit(Z).kneighbors(Z)[0][:, 1:]
    ref = backend(Z, L, D, 130, .25, "none")
    same = np.array_equal(np.unique(ref, return_inverse=True)[1], np.unique(z["pred"], return_inverse=True)[1])
    res = {}
    for k, sing, noise in grid:
        res[(k, sing, noise)] = score(gt, backend(Z, L, D, k, sing, noise))
    return s, same, score(gt, z["pred"]), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="80-87")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    subs = parse(a.subsets)
    noises = ["none", "single", "small:5", "small:10", "small:25", "iso:.05", "iso:.1", "iso:.15", "iso:.2"]
    grid = list(itertools.product((40, 60, 90, 130, 180), (0, .1, .2, .25, .3), noises))
    with ProcessPoolExecutor(8) as ex:
        out = list(ex.map(run, [(s, grid) for s in subs]))
    print("re-implementation matches leader labels:", [same for _, same, _, _ in out])
    base = np.mean([lead for _, _, lead, _ in out])
    print(f"leader actual: {base:.4f}")
    mean = {g: np.mean([r[g] for _, _, _, r in out]) for g in grid}
    wins = {g: sum(r[g] > r[(130, .25, 'none')] for _, _, _, r in out) for g in grid}
    print(f"re-impl (130, .25, none): {mean[(130, .25, 'none')]:.4f}")
    for g in sorted(grid, key=lambda g: -mean[g])[:a.top]:
        print(f"{mean[g]:.4f} ({(mean[g] / mean[(130, .25, 'none')] - 1) * 100:+.1f}%, better on {wins[g]}/{len(subs)})  k={g[0]} sing={g[1]} noise={g[2]}")


if __name__ == "__main__":
    main()
