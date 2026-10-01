"""Fast proxy: recall@15 of exact-mpnet neighbours, per feature set, on the 12 eval subsets."""
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

K = 15
FEATS = {}


def knn(E, k=K):
    S = E @ E.T
    S = S.toarray() if sp.issparse(S) else S
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, k, axis=1)[:, :k]


def _one(args):
    name, sub = args
    texts, _ = bench.load(sub)
    T = knn(np.load(f"{bench.BENCH}/{sub}.emb.npy"))
    P = knn(FEATS[name](texts))
    return sub, np.mean([len(set(a) & set(b)) / K for a, b in zip(T, P)])


def recall(name, subs=None):
    subs = subs or bench.REDDIT + bench.ARXIV
    with Pool(12) as p:
        r = dict(p.map(_one, [(name, s) for s in subs]))
    red = np.mean([r[s] for s in subs if s in bench.REDDIT])
    arx = np.mean([r[s] for s in subs if s in bench.ARXIV])
    print(f"recall@15 {name:40s} reddit {red:.3f}  arxiv {arx:.3f}  weighted {0.75 * red + 0.25 * arx:.3f}", flush=True)
    return r
