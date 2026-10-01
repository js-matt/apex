"""Apex SN1 text clustering (independent build, <50k chars).
TF-IDF (word + char_wb 3-5) blended with a 4096x16 hashed embedding table distilled from
all-mpnet-base-v2 neighbourhoods (4-bit, embedded below), then numpy UMAP + HDBSCAN.
GET /health; POST /cluster {"texts": [...]} -> {"cluster_ids": [...]} (-1 = noise).
"""
import argparse
import base64
import lzma

import numpy as np
import scipy.sparse as sp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import HashingVectorizer as HV, TfidfVectorizer as TV
from sklearn.preprocessing import normalize as nz

# __DATA__


def table():
    r = np.frombuffer(lzma.decompress(base64.b85decode(BLOB)), np.uint8)
    cb = np.frombuffer(r[:512].tobytes(), np.float16).reshape(16, 16).astype(np.float32)
    c = np.stack([r[512:] >> 4, r[512:] & 15], 1).reshape(4096, 16)
    return cb[np.arange(16), c]  # W[i, j] = codebook_j[code[i, j]]


W = table()
HASH = [HV(n_features=2048, alternate_sign=False, norm=None, token_pattern=r"(?u)\b\w+\b"),
        HV(n_features=2048, alternate_sign=False, norm=None, analyzer="char_wb", ngram_range=(3, 5))]


def features(t):
    X = sp.hstack([h.transform(t) for h in HASH]).tocsr()
    X.data = np.log1p(X.data)
    E = nz(X @ W)
    E = nz(E - E.mean(0))
    w = 0.35 if np.median([len(x) for x in t]) < 120 else 0.1  # short titles lean on the table
    a = dict(sublinear_tf=True, min_df=2, dtype=np.float32)
    parts = [sp.csr_matrix(E * np.sqrt(w))]
    for v in (TV(max_df=0.5, stop_words="english", **a), TV(analyzer="char_wb", ngram_range=(3, 5), max_df=0.3, **a)):
        try:
            parts.append(nz(v.fit_transform(t)) * np.sqrt((1 - w) / 2))
        except ValueError:  # empty vocabulary on tiny/degenerate input
            pass
    return sp.hstack(parts).tocsr()


# ---------------------------------------------------------------- UMAP (k=15, min_dist=0, cosine)
x0 = np.linspace(0, 3, 300)
AB = curve_fit(lambda x, a, b: 1 / (1 + a * x ** (2 * b)), x0, np.exp(-x0))[0]


def knn(F, k=15, blk=1024):
    n = F.shape[0]
    I, D = np.empty((n, k), int), np.empty((n, k), np.float32)
    for s in range(0, n, blk):
        d = 1 - (F[s:s + blk] @ F.T).toarray()
        np.maximum(d, 0, out=d)
        d[np.arange(len(d)), np.arange(s, s + len(d))] = 0
        p = np.argpartition(d, k - 1, 1)[:, :k]
        q = np.take_along_axis(d, p, 1)
        o = np.argsort(q, 1)
        I[s:s + blk], D[s:s + blk] = np.take_along_axis(p, o, 1), np.take_along_axis(q, o, 1)
    return I, D


def graph(I, D):
    n, k = I.shape
    rho = np.where(D > 0, D, np.inf).min(1)
    rho[~np.isfinite(rho)] = 0
    lo, hi, mid = np.zeros(n), np.full(n, np.inf), np.ones(n)
    d = D[:, 1:] - rho[:, None]
    for _ in range(64):
        up = np.where(d > 0, np.exp(-np.maximum(d, 0) / mid[:, None]), 1).sum(1) > np.log2(k)
        hi, lo = np.where(up, mid, hi), np.where(up, lo, mid)
        mid = np.where(np.isfinite(hi), (lo + hi) / 2, mid * 2)
    sig = np.maximum(mid, 1e-3 * np.where(rho > 0, D.mean(1), D.mean()))
    v = np.exp(-np.maximum(D - rho[:, None], 0) / sig[:, None])
    v[D - rho[:, None] <= 0] = 1
    v[I == np.arange(n)[:, None]] = 0
    A = sp.csr_matrix((v.ravel(), (np.repeat(np.arange(n), k), I.ravel())), (n, n))
    A = A + A.T - A.multiply(A.T)
    A.eliminate_zeros()
    return A.tocoo()


def umap(F, dim=5, seed=42):
    rng = np.random.default_rng(seed)
    P = graph(*knn(F))
    n = P.shape[0]
    ep = int(min(200, max(30, 1e6 / n)))
    m = P.data >= P.data.max() / ep
    h, t, w = P.row[m], P.col[m], P.data[m]
    L = sp.coo_matrix((w, (h, t)), (n, n)).tocsr()
    dg = sp.diags(1 / np.sqrt(np.maximum(np.asarray(L.sum(1)).ravel(), 1e-12)))
    try:
        val, vec = eigsh(sp.identity(n) - dg @ L @ dg, dim + 1, which="SM", ncv=max(2 * dim + 1, int(n ** .5)),
                         tol=1e-4, v0=np.ones(n), maxiter=n * 5)
        Y = vec[:, np.argsort(val)[1:]]
    except Exception:
        Y = rng.uniform(-10, 10, (n, dim))
    Y = Y * (10 / np.abs(Y).max()) + rng.normal(0, 1e-4, Y.shape)
    Y = 10 * (Y - Y.min(0)) / (Y.max(0) - Y.min(0))
    a, b = AB
    eps = w.max() / w
    nxt, en = eps.copy(), eps / 5
    nn = en.copy()
    for e in range(ep):
        al = 1 - e / ep
        u = np.nonzero(nxt <= e)[0]
        if not len(u):
            continue
        i, j = h[u], t[u]
        df = Y[i] - Y[j]
        d2 = (df * df).sum(1)
        g = np.where(d2 > 0, -2 * a * b * d2 ** (b - 1) / (a * d2 ** b + 1), 0)
        g = np.clip(g[:, None] * df, -4, 4) * al
        U = np.zeros_like(Y)
        np.add.at(U, i, g)
        np.add.at(U, j, -g)
        nxt[u] += eps[u]
        c = np.maximum(np.floor((e - nn[u]) / en[u]).astype(int), 0)
        nn[u] += c * en[u]
        i = np.repeat(i, c)
        if len(i):
            j = rng.integers(0, n, len(i))
            df = Y[i] - Y[j]
            d2 = (df * df).sum(1)
            g = np.where(d2 > 0, 2 * b / ((1e-3 + d2) * (a * d2 ** b + 1)), 0)
            g = np.clip(g[:, None] * df, -4, 4) * al
            g[i == j] = 0
            np.add.at(U, i, g)
        Y += U
    return Y.astype(np.float32)


def cluster_texts(t):
    n = len(t)
    if n < 30:
        return [0] * n
    Y = umap(features([str(x) for x in t]))
    lab = HDBSCAN(min_cluster_size=25, min_samples=10).fit_predict(Y)
    big = np.bincount(lab[lab >= 0]).max() / n if (lab >= 0).any() else 1
    if big > 0.35 or lab.max() < 7:  # EOM collapsed into a few blobs: cap cluster size
        lab = HDBSCAN(min_cluster_size=25, min_samples=10, max_cluster_size=int(0.15 * n)).fit_predict(Y)
    return [int(x) for x in lab]


class Req(BaseModel):
    texts: list[str]


class Resp(BaseModel):
    cluster_ids: list[int]


app = FastAPI()


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/cluster", response_model=Resp)
def cluster(r: Req):
    if not r.texts:
        raise HTTPException(400, "No texts provided")
    return Resp(cluster_ids=cluster_texts(r.texts))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--host", default="0.0.0.0")
    a = p.parse_args()
    uvicorn.run(app, host=a.host, port=a.port)
