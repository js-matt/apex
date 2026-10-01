"""UMAP (numpy/scipy re-implementation) + sklearn HDBSCAN, mirroring the GT recipe:
UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine) -> HDBSCAN(25, min_samples=10).
"""
import numpy as np
import scipy.sparse as sp
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh
from sklearn.cluster import HDBSCAN


def find_ab(spread=1.0, min_dist=0.0):
    x = np.linspace(0, spread * 3, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    (a, b), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), x, y)
    return float(a), float(b)


def knn_cosine(E, k, block=2048):
    """Exact kNN on L2-normalized rows. Returns (idx, dist) sorted, self included."""
    n = E.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        S = E[s:s + block] @ E.T
        D = 1.0 - (S.toarray() if sp.issparse(S) else S)
        np.maximum(D, 0, out=D)
        D[np.arange(D.shape[0]), np.arange(s, s + D.shape[0])] = 0.0
        p = np.argpartition(D, k - 1, axis=1)[:, :k]
        d = np.take_along_axis(D, p, 1)
        o = np.argsort(d, axis=1)
        idx[s:s + block] = np.take_along_axis(p, o, 1)
        dist[s:s + block] = np.take_along_axis(d, o, 1)
    return idx, dist


def fuzzy_graph(idx, dist):
    n, k = idx.shape
    target = np.log2(k)
    nz = np.where(dist > 0, dist, np.inf)
    rho = nz.min(1)
    rho[~np.isfinite(rho)] = 0.0
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    mid = np.ones(n)
    d = dist[:, 1:] - rho[:, None]
    for _ in range(64):
        psum = np.where(d > 0, np.exp(-np.maximum(d, 0) / mid[:, None]), 1.0).sum(1)
        up = psum > target
        hi = np.where(up, mid, hi)
        lo = np.where(up, lo, mid)
        mid = np.where(np.isfinite(hi), (lo + hi) / 2, mid * 2)
    sigma = np.maximum(mid, 1e-3 * np.where(rho > 0, dist.mean(1), dist.mean()))
    val = np.exp(-np.maximum(dist - rho[:, None], 0) / sigma[:, None])
    val[dist - rho[:, None] <= 0] = 1.0
    val[idx == np.arange(n)[:, None]] = 0.0
    A = sp.csr_matrix((val.ravel(), (np.repeat(np.arange(n), k), idx.ravel())), shape=(n, n))
    A.eliminate_zeros()
    At = A.T.tocsr()
    P = A + At - A.multiply(At)
    P.eliminate_zeros()
    return P.tocoo()


def spectral_init(P, dim, rng):
    n = P.shape[0]
    deg = np.asarray(P.sum(1)).ravel()
    Dm = sp.diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    L = sp.identity(n) - Dm @ P @ Dm
    try:
        vals, vecs = eigsh(L, dim + 1, which="SM", ncv=max(2 * dim + 1, int(np.sqrt(n))),
                           tol=1e-4, v0=np.ones(n), maxiter=n * 5)
        Y = vecs[:, np.argsort(vals)[1:dim + 1]]
    except Exception:
        Y = rng.uniform(-10, 10, (n, dim))
    Y = Y * (10.0 / np.abs(Y).max())
    Y = Y + rng.normal(scale=1e-4, size=Y.shape)
    return 10.0 * (Y - Y.min(0)) / (Y.max(0) - Y.min(0))


def umap(E, n_neighbors=15, dim=5, n_epochs=200, neg_rate=5, seed=42, graph=None):
    rng = np.random.default_rng(seed)
    if graph is None:
        idx, dist = knn_cosine(E, n_neighbors)
        graph = fuzzy_graph(idx, dist)
    P = graph
    n = P.shape[0]
    w = P.data.copy()
    keep = w >= w.max() / n_epochs
    head, tail, w = P.row[keep], P.col[keep], w[keep]
    a, b = find_ab()
    Y = spectral_init(sp.coo_matrix((w, (head, tail)), shape=(n, n)).tocsr(), dim, rng).astype(np.float64)
    eps = w.max() / w  # epochs per sample
    nxt = eps.copy()
    eps_neg = eps / neg_rate
    nxt_neg = eps_neg.copy()
    for ep in range(n_epochs):
        alpha = 1.0 - ep / n_epochs
        due = np.nonzero(nxt <= ep)[0]
        if due.size == 0:
            continue
        h, t = head[due], tail[due]
        diff = Y[h] - Y[t]
        d2 = (diff * diff).sum(1)
        gc = np.where(d2 > 0, -2.0 * a * b * np.power(d2, b - 1.0) / (a * np.power(d2, b) + 1.0), 0.0)
        g = np.clip(gc[:, None] * diff, -4, 4) * alpha
        upd = np.zeros_like(Y)
        np.add.at(upd, h, g)
        np.add.at(upd, t, -g)
        nxt[due] += eps[due]
        nneg = np.floor((ep - nxt_neg[due]) / eps_neg[due]).astype(np.int64)
        nneg = np.maximum(nneg, 0)
        nxt_neg[due] += nneg * eps_neg[due]
        rep_h = np.repeat(h, nneg)
        if rep_h.size:
            rep_t = rng.integers(0, n, rep_h.size)
            diff = Y[rep_h] - Y[rep_t]
            d2 = (diff * diff).sum(1)
            gc = np.where(d2 > 0, 2.0 * b / ((0.001 + d2) * (a * np.power(d2, b) + 1.0)), 0.0)
            g = np.clip(gc[:, None] * diff, -4, 4) * alpha
            g[rep_h == rep_t] = 0.0
            np.add.at(upd, rep_h, g)
        Y += upd
    return Y.astype(np.float32)


def hdbscan(Y, min_cluster_size=25, min_samples=10, guard=True):
    lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(Y)
    n = len(lab)
    big = np.bincount(lab[lab >= 0]).max() / n if (lab >= 0).any() else 1.0
    if guard and (big > 0.35 or lab.max() + 1 < 8):
        # EOM picked a few giant blobs: forbid clusters above 15% of the points
        lab = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                      max_cluster_size=int(0.15 * n)).fit_predict(Y)
    return lab


def cluster(E, mcs=25, ms=10, guard=True, **kw):
    return hdbscan(umap(E, **kw), mcs, ms, guard)
