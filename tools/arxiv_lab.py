"""Feature x back-end experiments for the arXiv-title subsets (trusted code only, no third-party solutions).

    .venv/bin/python tools/arxiv_lab.py --subsets 80-87
Features: tfidf word+char SVD, leader BD projection (decoded by leader_blobs.py), an in-domain ridge
projection trained on pool titles that are NOT in the evaluated subsets, and mpnet itself (ceiling).
Back-ends: average-linkage cut + singleton fraction (the leader's short path), numpy UMAP + HDBSCAN.
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "2"
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import hstack
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from np_umap import umap_embed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH, ARX = ROOT / "data" / "bench", ROOT / "data" / "arxiv"
_hw = HashingVectorizer(n_features=8192, ngram_range=(1, 2), analyzer="word", alternate_sign=False, norm=None, dtype=np.float32)
_hc = HashingVectorizer(n_features=4096, ngram_range=(3, 5), analyzer="char_wb", alternate_sign=False, norm=None, dtype=np.float32)
_url, _at, _ent = re.compile(r"https?://\S+|www\.\S+"), re.compile(r"@\w+"), re.compile(r"&\w+;")
_pun, _ws = re.compile(r"[^\w\s一-鿿]"), re.compile(r"\s+")


def bs_norm(t):
    """Leader's BS(): text normalisation in front of the BD projection."""
    t = _at.sub(" ", _url.sub(" ", t)).lower()
    return _ws.sub(" ", _pun.sub(" ", _ent.sub(" ", t))).strip()[:1000]


def hashed(texts):
    T = [bs_norm(t) for t in texts]
    X = hstack([_hw.transform(T), _hc.transform(T)]).tocsr()
    X.data = np.log1p(X.data).astype(np.float32)
    return normalize(X)


def score(gt, pred):
    return (max(0.0, adjusted_rand_score(gt, pred)) + normalized_mutual_info_score(gt, pred)) / 2


def tfidf_svd(texts, dim=128):
    mats = [TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32).fit_transform(texts),
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_df=.5, sublinear_tf=True, dtype=np.float32).fit_transform(texts)]
    X = normalize(hstack([normalize(m) for m in mats]).tocsr())
    return normalize(TruncatedSVD(dim, random_state=0).fit_transform(X))


def avg_link(F, k, frac, kn=12):
    lab = fcluster(linkage(F, "average", "cosine"), k, "maxclust").astype(np.int64)
    if frac > 0:
        d = NearestNeighbors(n_neighbors=kn + 1, metric="cosine").fit(F).kneighbors(F)[0][:, -1]
        idx = np.argsort(-d)[:int(frac * len(F))]
        lab[idx] = np.arange(10 ** 6, 10 ** 6 + len(idx))
    return lab


def umap_hdb(F, mcs=25, epochs=300, noise="minus1"):
    lab = HDBSCAN(min_cluster_size=mcs).fit_predict(umap_embed(F, n_epochs=epochs))
    if noise == "single":
        m = lab == -1
        lab[m] = np.arange(10 ** 6, 10 ** 6 + m.sum())
    return lab


def run(job):
    s, fname, W = job
    rows = [json.loads(l) for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")]
    texts, gt = [r["text"] for r in rows], np.array([r["label"] for r in rows])
    if fname == "mpnet":
        F = normalize(np.load(BENCH / f"subset_{s:02d}.emb.npy"))
    elif fname == "tfidf":
        F = tfidf_svd(texts)
    elif fname.startswith("tfidf+"):
        P = normalize(np.asarray(hashed(texts) @ W))
        F = normalize(np.hstack([tfidf_svd(texts), P * float(fname.split("*")[1])]))
    else:
        F = normalize(np.asarray(hashed(texts) @ W))
    out = {}
    for k in (40, 80, 130):
        for frac in (0, .25):
            out[f"avg k={k} single={frac}"] = score(gt, avg_link(F, k, frac))
    out["umap+hdb noise=-1"] = score(gt, umap_hdb(F))
    out["umap+hdb noise=single"] = score(gt, umap_hdb(F, noise="single"))
    return s, fname, out


def ridge_W(exclude, dim):
    pool = [json.loads(l) for l in open(ARX / "pool.jsonl", encoding="utf-8")]
    emb = np.load(ARX / "pool.emb.npy")
    keep = np.array([r["title"] not in exclude for r in pool])
    X = hashed([r["title"] for r, k in zip(pool, keep) if k])
    Y = PCA(dim, random_state=0).fit_transform(emb[keep])
    A = (X.T @ X).toarray()
    A[np.diag_indices_from(A)] += 1.0
    return np.linalg.solve(A, np.asarray(X.T @ Y)).astype(np.float32)


def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="80-87")
    ap.add_argument("--feats", default="mpnet,tfidf,leaderBD,ridge48,tfidf+ridge48*1")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    subs = parse(a.subsets)
    exclude = {json.loads(l)["text"] for s in subs for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")}
    Ws = {"leaderBD": np.load(ROOT / "data" / "leader_W_BD.npy")}
    for f in a.feats.split(","):
        m = re.search(r"ridge(\d+)", f)
        if m and f"ridge{m.group(1)}" not in Ws:
            Ws[f"ridge{m.group(1)}"] = ridge_W(exclude, int(m.group(1)))
    jobs = []
    for f in a.feats.split(","):
        m = re.search(r"(ridge\d+|leaderBD)", f)
        jobs += [(s, f, Ws[m.group(1)] if m else None) for s in subs]
    res = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for s, f, out in ex.map(run, jobs):
            res.setdefault(f, {})[s] = out
    for f, by in res.items():
        print(f"== {f}")
        for key in next(iter(by.values())):
            print(f"  {np.mean([by[s][key] for s in subs]):.4f}  {key}")
