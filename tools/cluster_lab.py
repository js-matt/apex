"""Clustering experiments on dumped feature matrices (no third-party code runs here).

Loads data/sandbox_out/feat/feat_<solution>_<subset>.npz (from dump_features.py) plus the
benchmark ground truth, and scores alternative clustering back-ends on the same features.
"""
import json
import os
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
FEAT = ROOT / "data" / "sandbox_out" / "feat"
LEADER = "r65_180873_5HWJVc12_v1_0.3963"


def load(subset, sol=LEADER):
    feat_p = FEAT / f"feat_{sol}_{subset:02d}.npz"
    d = dict(np.load(feat_p)) if feat_p.exists() else {}
    rows = [json.loads(l) for l in open(BENCH / f"subset_{subset:02d}.jsonl", encoding="utf-8")]
    gt = np.array([r.get("label", -2) for r in rows])
    emb_p = BENCH / f"subset_{subset:02d}.emb.npy"
    emb = np.load(emb_p) if emb_p.exists() else None
    out = {k: v for k, v in d.items()}
    out.update(G=d.get("G"), keep=d.get("keep"), pred=d.get("pred"), regime=str(d.get("regime", "")), gt=gt,
               emb=emb, texts=[r["text"] for r in rows])
    return out


def score(gt, pred):
    a, n = adjusted_rand_score(gt, pred), normalized_mutual_info_score(gt, pred)
    return (max(0.0, a) + n) / 2, a, n


def expand(lab_kept, keep, fill=-1):
    """Map labels for kept rows back to all texts (dropped texts get `fill`)."""
    out = np.full(len(keep), fill, np.int64)
    out[keep] = lab_kept
    return out


def singletons(lab, dist, frac, protect=None):
    """Leader-style noise: the `frac` of points with the largest kNN distance become singletons."""
    lab = lab.copy()
    n = int(len(lab) * frac)
    if n <= 0:
        return lab
    d = dist.copy()
    if protect is not None:
        d[protect] = -1
    idx = np.argpartition(-d, n - 1)[:n]
    idx = idx[d[idx] >= 0]
    lab[idx] = np.arange(10**6, 10**6 + len(idx))
    return lab


def knn_dist(Z, k):
    return NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z).kneighbors(Z)[0][:, 1:]


def avg_linkage(Z, n_clusters):
    return fcluster(linkage(Z, "average", "cosine"), n_clusters, "maxclust").astype(np.int64)


def env_threads(n=1):
    for v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = str(n)


def smooth(Z, k=20, alpha=0.6, iters=4):
    """kNN feature smoothing: each point moves toward the mean of its k cosine neighbours."""
    Z = normalize(np.asarray(Z, np.float32))
    nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Z).kneighbors(Z)[1][:, 1:]
    for _ in range(iters):
        Z = normalize((1 - alpha) * Z + alpha * Z[nb].mean(1))
    return Z


def short_core(Z, n_clusters=100, frac=0.3, k_dist=12):
    """Leader's short-text back-end on already-smoothed features: average linkage + kNN-distance singletons."""
    return singletons(avg_linkage(Z, n_clusters), knn_dist(Z, k_dist)[:, -1], frac)


def remove_pcs(X, n):
    """Centre, then remove the top-n principal directions (common-component removal)."""
    from sklearn.decomposition import TruncatedSVD
    X = np.asarray(X, np.float32)
    if n < 1:
        return normalize(X)
    X = X - X.mean(0, keepdims=True)
    svd = TruncatedSVD(n_components=n, random_state=42).fit(X)
    return normalize(X - svd.inverse_transform(svd.transform(X)))


def first_pc_ratio(X):
    from sklearn.decomposition import TruncatedSVD
    X = np.asarray(X, np.float32)
    X = X - X.mean(0, keepdims=True)
    return float(TruncatedSVD(n_components=1, random_state=42).fit(X).explained_variance_ratio_[0])


def spectral(Z, dim, k=15):
    """Top eigenvectors of the normalised symmetric kNN connectivity graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    n = Z.shape[0]
    A = NearestNeighbors(n_neighbors=min(k, n - 1), metric="cosine").fit(Z).kneighbors_graph(Z, mode="connectivity")
    A = ((A + A.T) > 0).astype(np.float32)
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    h = 1 / np.sqrt(deg)
    M = csr_matrix(A.multiply(h[:, None]).multiply(h[None, :]))
    _, V = eigsh(M, k=min(dim, n - 2), which="LA", tol=1e-3, maxiter=300, v0=np.full(n, 1 / np.sqrt(n), np.float32))
    return normalize(V.astype(np.float32))


def long_features(d, w_aw=1.0, w_t=None, w_ay=None, mid=True, extra=(), smooth_k=None, smooth_a=None, smooth_it=1, spec_dim=None):
    """Leader-style long-path feature assembly from dumped components (comp_t, comp_Aw, comp_Ay)."""
    t, aw, ay = d["comp_t"], d["comp_Aw"], d["comp_Ay"]
    a = first_pc_ratio(t)
    j = a > 0.045
    l = 4 if j else 1 if a < 0.03 else (3 if mid else 2)
    parts = [normalize(remove_pcs(aw, l)) * w_aw]
    tt = remove_pcs(t, 1) if j else t
    parts.append(normalize(tt) * (w_t if w_t is not None else (1.0 if j else 0.8 if a < 0.03 else 0.94)))
    parts.append(normalize(ay) * (w_ay if w_ay is not None else (0.85 if mid else 0.7)))
    for X, wt in extra:
        parts.append(normalize(X) * wt)
    G = normalize(np.hstack(parts))
    k = smooth_k or (12 if mid else 8)
    alpha = smooth_a if smooth_a is not None else (0.2 if mid else 0.1)
    nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(G).kneighbors(G)[1][:, 1:]
    for _ in range(smooth_it):
        G = normalize((1 - alpha) * G + alpha * G[nb].mean(1))
    dim = spec_dim or ((36 if j else 16 if a < 0.03 else 32) if mid else (28 if j else 12 if a < 0.03 else 32))
    return normalize(np.hstack([G, spectral(G, dim)]))


def long_core(G, n_clusters=28, frac=0.25, k_dist=20):
    return singletons(avg_linkage(G, n_clusters), knn_dist(G, k_dist)[:, -1], frac)
