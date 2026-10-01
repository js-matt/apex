"""Text clustering for Apex SN1: a 3-layer transformer distilled from all-mpnet-base-v2
(the ground-truth embedder), run in numpy, then UMAP (own numpy implementation) + HDBSCAN.

Weights are embedded below (fp16, base64), so this file is large by design.

Contract: GET /health, POST /cluster {"texts": [...]} -> {"cluster_ids": [...]} (-1 = noise).
"""
import argparse
import base64
import math
import time
import unicodedata

import numpy as np
import scipy.sparse as sp
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from scipy.optimize import curve_fit
from scipy.sparse.linalg import eigsh
from sklearn.cluster import HDBSCAN

TIME_LIMIT = 90.0
SAFETY = 0.70  # plan to use at most this share of the limit
MAX_LEN = 128
REF_RATE = 7000.0  # Encoder.calibrate() tokens/s on the dev machine (1 core)
UMAP_COST = 8e-6  # dev-machine seconds per (text x epoch) for UMAP, incl. kNN/HDBSCAN slack

# __DATA__


# ------------------------------------------------------------------ tokenizer (BERT uncased WordPiece)
def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _is_punc(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


class WordPiece:
    def __init__(self, tokens):
        self.vocab = {t: i for i, t in enumerate(tokens)}
        self.unk = self.vocab["[UNK]"]
        self.cache = {}

    def normalize(self, text):
        out = []
        for ch in text:
            if ch in "\t\n\r":
                out.append(" ")
            elif ch == "\0" or ch == "�" or unicodedata.category(ch)[0] == "C":
                continue
            elif ch.isspace():
                out.append(" ")
            elif _is_cjk(ord(ch)):
                out.append(" " + ch + " ")
            else:
                out.append(ch)
        text = unicodedata.normalize("NFD", "".join(out))
        return "".join(c for c in text if unicodedata.category(c) != "Mn").lower()

    def words(self, text):
        for w in text.split():
            cur = []
            for ch in w:
                if _is_punc(ch):
                    if cur:
                        yield "".join(cur)
                        cur = []
                    yield ch
                else:
                    cur.append(ch)
            if cur:
                yield "".join(cur)

    def word_ids(self, w):
        r = self.cache.get(w)
        if r is not None:
            return r
        if len(w) > 100:
            r = [self.unk]
        else:
            r, start, n = [], 0, len(w)
            while start < n:
                end, tok = n, None
                while start < end:
                    tok = self.vocab.get(w[start:end] if start == 0 else "##" + w[start:end])
                    if tok is not None:
                        break
                    end -= 1
                if tok is None:
                    r = [self.unk]
                    break
                r.append(tok)
                start = end
        self.cache[w] = r
        return r

    def encode(self, text, max_len):
        ids = []
        for w in self.words(self.normalize(text.strip())):
            ids.extend(self.word_ids(w))
            if len(ids) >= max_len - 2:
                break
        return [101] + ids[:max_len - 2] + [102]


# ------------------------------------------------------------------ encoder
def _ln(x, g, b, eps):
    x = x - x.mean(-1, keepdims=True)
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g + b


class Encoder:
    def __init__(self):
        raw = np.frombuffer(base64.b64decode(_BLOB), dtype=np.float16)
        self.W, o = [], 0
        for shp in _SHAPES:
            k = int(np.prod(shp))
            self.W.append(raw[o:o + k].astype(np.float32).reshape(shp))
            o += k
        self.nl, self.nh, self.eps = _META
        self.hid = self.W[0].shape[1]
        self.tok = WordPiece(_VOCAB.split("\n"))
        self.rate = None

    def forward(self, ids, mask):
        W, B, L, hid = self.W, ids.shape[0], ids.shape[1], self.hid
        x = W[0][ids] + W[1][:L][None] + W[2]
        x = _ln(x, W[3], W[4], self.eps).reshape(B * L, hid)
        am = ((1.0 - mask[:, None, None, :]) * -1e9).astype(np.float32)
        dh = hid // self.nh
        sc = np.float32(1.0 / math.sqrt(dh))
        for i in range(self.nl):
            wqkv, bqkv, wo, bo, g1, b1, wi, bi, wo2, bo2, g2, b2 = W[5 + 12 * i:17 + 12 * i]
            qkv = (x @ wqkv + bqkv).reshape(B, L, 3, self.nh, dh).transpose(2, 0, 3, 1, 4)
            s = (qkv[0] * sc) @ qkv[1].transpose(0, 1, 3, 2) + am
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            c = (s @ qkv[2]).transpose(0, 2, 1, 3).reshape(B * L, hid)
            x = _ln(c @ wo + bo + x, g1, b1, self.eps)
            h = x @ wi + bi
            h = 0.5 * h * (1.0 + np.tanh(np.float32(0.7978845608) * (h + np.float32(0.044715) * h * h * h)))
            x = _ln(h @ wo2 + bo2 + x, g2, b2, self.eps)
        m = mask.reshape(B, L, 1)
        e = (x.reshape(B, L, hid) * m).sum(1) / m.sum(1)
        e = e @ W[-2] + W[-1]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def embed(self, seqs, tok_budget=8192):
        order = np.argsort([len(s) for s in seqs], kind="stable")
        out = np.zeros((len(seqs), self.W[-2].shape[1]), dtype=np.float32)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= max(tok_budget, len(seqs[order[j]])):
                j += 1
            idx = order[i:j]
            L = max(len(seqs[k]) for k in idx)
            ids = np.zeros((len(idx), L), dtype=np.int64)
            mask = np.zeros((len(idx), L), dtype=np.float32)
            for r, k in enumerate(idx):
                ids[r, :len(seqs[k])] = seqs[k]
                mask[r, :len(seqs[k])] = 1
            out[idx] = self.forward(ids, mask)
            i = j
        return out

    def calibrate(self):
        """Measure padded-tokens/second on this machine (two sequence lengths)."""
        rng = np.random.default_rng(0)
        self.forward(rng.integers(1000, 20000, (8, 16)), np.ones((8, 16), np.float32))
        rates = []
        for B, L in ((256, 32), (64, 128)):
            t = time.time()
            self.forward(rng.integers(1000, 20000, (B, L)), np.ones((B, L), np.float32))
            rates.append(B * L / (time.time() - t))
        self.rate = min(rates)


# ------------------------------------------------------------------ UMAP (n_neighbors=15, min_dist=0, cosine)
def _find_ab(spread=1.0, min_dist=0.0):
    x = np.linspace(0, spread * 3, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))
    (a, b), _ = curve_fit(lambda x, a, b: 1.0 / (1.0 + a * x ** (2 * b)), x, y)
    return float(a), float(b)


_AB = _find_ab()


def knn_cosine(E, k, block=2048):
    n = E.shape[0]
    idx = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    for s in range(0, n, block):
        D = 1.0 - E[s:s + block] @ E.T
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
    lo, hi, mid = np.zeros(n), np.full(n, np.inf), np.ones(n)
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
    Y = Y * (10.0 / np.abs(Y).max()) + rng.normal(scale=1e-4, size=Y.shape)
    return 10.0 * (Y - Y.min(0)) / (Y.max(0) - Y.min(0))


def umap(E, n_neighbors=15, dim=5, n_epochs=200, neg_rate=5, seed=42):
    rng = np.random.default_rng(seed)
    P = fuzzy_graph(*knn_cosine(E, n_neighbors))
    n = P.shape[0]
    keep = P.data >= P.data.max() / n_epochs
    head, tail, w = P.row[keep], P.col[keep], P.data[keep]
    a, b = _AB
    Y = spectral_init(sp.coo_matrix((w, (head, tail)), shape=(n, n)).tocsr(), dim, rng)
    eps = w.max() / w
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
        nneg = np.maximum(np.floor((ep - nxt_neg[due]) / eps_neg[due]).astype(np.int64), 0)
        nxt_neg[due] += nneg * eps_neg[due]
        rh = np.repeat(h, nneg)
        if rh.size:
            rt = rng.integers(0, n, rh.size)
            diff = Y[rh] - Y[rt]
            d2 = (diff * diff).sum(1)
            gc = np.where(d2 > 0, 2.0 * b / ((0.001 + d2) * (a * np.power(d2, b) + 1.0)), 0.0)
            g = np.clip(gc[:, None] * diff, -4, 4) * alpha
            g[rh == rt] = 0.0
            np.add.at(upd, rh, g)
        Y += upd
    return Y


# ------------------------------------------------------------------ pipeline
ENC = None


def cluster_texts(texts):
    t0 = time.time()
    n = len(texts)
    if n < 20:
        return [0] * n
    texts = [t if isinstance(t, str) else str(t) for t in texts]
    # tokenize once at MAX_LEN; shorter limits are prefixes (+ [SEP])
    full = [ENC.tok.encode(t, MAX_LEN) for t in texts]
    n_epochs = 200
    slow = REF_RATE / ENC.rate  # >1 when this machine is slower than the dev machine
    umap_est = UMAP_COST * n * n_epochs * slow + 1.0
    budget = TIME_LIMIT * SAFETY - (time.time() - t0) - umap_est
    max_len = MAX_LEN
    while max_len > 16:
        tokens = sum(min(len(s), max_len) for s in full) * 1.15  # padding overhead
        if tokens / ENC.rate <= budget:
            break
        max_len = int(max_len * 0.8)
    if budget < 0:
        n_epochs = 100
    seqs = full if max_len == MAX_LEN else [s if len(s) <= max_len else s[:max_len - 1] + [102] for s in full]
    E = ENC.embed(seqs)
    Y = umap(E, n_epochs=n_epochs)
    labels = HDBSCAN(min_cluster_size=25, min_samples=10).fit_predict(Y.astype(np.float32))
    return [int(x) for x in labels]


class ClusterRequest(BaseModel):
    texts: list[str]


class ClusterResponse(BaseModel):
    cluster_ids: list[int]


def make_app():
    global ENC
    ENC = Encoder()
    ENC.calibrate()
    app = FastAPI(title="Text Clustering Miner")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/cluster", response_model=ClusterResponse)
    def cluster(req: ClusterRequest):
        if not req.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        return ClusterResponse(cluster_ids=cluster_texts(req.texts))

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", type=str, default="0.0.0.0")
    args = ap.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="info")
