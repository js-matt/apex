"""Minimal UMAP in numpy/scipy/sklearn (the sandbox has no umap-learn).

Follows umap-learn's pipeline: cosine kNN graph -> smooth-kNN fuzzy weights -> fuzzy union ->
spectral init scaled to [0, 10] -> epoch-scheduled SGD with negative sampling (a, b fitted for
min_dist=0). Updates are applied per epoch in a vectorised batch rather than edge by edge.
"""
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

A_MD0, B_MD0 = 1.8956, 0.8006  # umap.find_ab_params(spread=1, min_dist=0)


def knn(Z, k):
    d, i = NearestNeighbors(n_neighbors=k, metric="cosine").fit(Z).kneighbors(Z)
    return i, np.maximum(d, 0.0)


def fuzzy_graph(idx, dist, k):
    n = idx.shape[0]
    rho = dist[:, 1]
    target = np.log2(k)
    lo, hi, mid = np.zeros(n), np.full(n, np.inf), np.ones(n)
    d = dist[:, 1:] - rho[:, None]
    for _ in range(64):
        psum = np.exp(-np.maximum(d, 0) / mid[:, None]).sum(1)
        big = psum > target
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
        mid = np.where(np.isinf(hi), mid * 2, (lo + hi) / 2)
    mean_d = dist[:, 1:].mean()
    sigma = np.maximum(mid, 1e-3 * np.where(rho > 0, dist[:, 1:].mean(1), mean_d))
    w = np.exp(-np.maximum(d, 0) / sigma[:, None])
    rows = np.repeat(np.arange(n), k - 1)
    P = coo_matrix((w.ravel(), (rows, idx[:, 1:].ravel())), shape=(n, n)).tocsr()
    P = P + P.T - P.multiply(P.T)
    return P.tocoo()


def spectral_init(P, dim, seed):
    n = P.shape[0]
    deg = np.asarray(P.sum(1)).ravel()
    Dm = diags(1.0 / np.sqrt(np.maximum(deg, 1e-12)))
    M = csr_matrix(Dm @ P @ Dm)
    rng = np.random.default_rng(seed)
    try:
        _, vec = eigsh(M, k=dim + 1, which="LA", tol=1e-4, maxiter=n * 5, v0=rng.random(n))
        Y = vec[:, :-1][:, ::-1]
    except Exception:
        Y = rng.normal(size=(n, dim))
    Y = Y - Y.min(0)
    Y = 10 * Y / np.maximum(Y.max(0), 1e-12)
    return (Y + rng.normal(scale=1e-4, size=Y.shape)).astype(np.float32)


def optimise(Y, P, n_epochs, seed, neg=5, a=A_MD0, b=B_MD0):
    rng = np.random.default_rng(seed)
    n = Y.shape[0]
    keep = P.data >= P.data.max() / n_epochs
    head, tail, w = P.row[keep], P.col[keep], P.data[keep]
    eps = w.max() / w  # epochs_per_sample
    nxt = eps.copy()
    for ep in range(n_epochs):
        lr = 1.0 - ep / n_epochs
        fire = nxt <= ep + 1
        if not fire.any():
            continue
        nxt[fire] += eps[fire]
        h, t = head[fire], tail[fire]
        diff = Y[h] - Y[t]
        d2 = (diff * diff).sum(1)
        d2c = np.maximum(d2, 1e-12)
        coef = np.where(d2 > 0, -2 * a * b * d2c ** (b - 1) / (1 + a * d2c ** b), 0.0)
        g = np.clip(coef[:, None] * diff, -4, 4) * lr
        dY = np.zeros_like(Y)
        np.add.at(dY, h, g)
        np.add.at(dY, t, -g)
        hn = np.repeat(h, neg)
        tn = rng.integers(0, n, hn.size)
        diff = Y[hn] - Y[tn]
        d2 = (diff * diff).sum(1)
        coef = 2 * b / ((0.001 + d2) * (1 + a * d2 ** b))
        g = np.clip(coef[:, None] * diff, -4, 4) * lr
        g[tn == hn] = 0
        np.add.at(dY, hn, g)
        Y += dY.astype(np.float32)
    return Y


def umap_embed(Z, n_neighbors=15, n_components=5, n_epochs=200, seed=0):
    idx, dist = knn(Z, n_neighbors)
    P = fuzzy_graph(idx, dist, n_neighbors)
    Y = spectral_init(P, n_components, seed)
    return optimise(Y, P, n_epochs, seed)
