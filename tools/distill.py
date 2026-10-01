"""Distil mpnet into a small hashed-n-gram linear projection (the leader's `Ak` block format).

Features match the leader's Ay(): leader-normalised text -> HashingVectorizer word 1-2gram (8192) +
char_wb 3-5gram (4096), no sign, log1p counts, L2 row norm -> @ W (12288 x 48) -> L2 norm.
The target is the top-48 PCA of all-mpnet-base-v2 embeddings (the benchmark subsets' .emb.npy).

    python tools/distill.py --train 0,1,30-36,50-56 --test 37,57,60
"""
import argparse
import json
import lzma
import re
from pathlib import Path

import numpy as np
from scipy.sparse import hstack, vstack
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"
N_WORD, N_CHAR, DIM = 8192, 4096, 48
_url, _at, _tag = re.compile(r"http\S+|www\S+|https\S+"), re.compile(r"@\w+"), re.compile(r"#(\w+)")
_punct, _ws = re.compile(r"[^\w\s]"), re.compile(r"\s+")
_hw = HashingVectorizer(n_features=N_WORD, ngram_range=(1, 2), analyzer="word", alternate_sign=False, norm=None, dtype=np.float32)
_hc = HashingVectorizer(n_features=N_CHAR, ngram_range=(3, 5), analyzer="char_wb", alternate_sign=False, norm=None, dtype=np.float32)


def norm_text(t):
    """Same normalisation as the leader's Ar()."""
    t = _url.sub(" ", t.lower())
    t = _at.sub(" ", t)
    t = _tag.sub(r"\1", t)
    return _ws.sub(" ", _punct.sub(" ", t)).strip()


def features(texts):
    T = [norm_text(t) or "x" for t in texts]
    X = hstack([_hw.transform(T), _hc.transform(T)]).tocsr()
    X.data = np.log1p(X.data).astype(np.float32)
    return normalize(X)


def load(s):
    rows = [json.loads(l) for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")]
    return [r["text"] for r in rows], np.load(BENCH / f"subset_{s:02d}.emb.npy")


def ridge(X, Y, lam):
    A = (X.T @ X).toarray() if hasattr(X, "toarray") else X.T @ X
    A[np.diag_indices_from(A)] += lam
    return np.linalg.solve(A, np.asarray(X.T @ Y)).astype(np.float32)


def quantize(W, levels=4, iters=15):
    """Per-column 1-D k-means codebook (the leader's Ak format: 2-bit codes + 4 floats per column).
    One level is pinned at 0 so pruned weights cost nothing after lzma."""
    codes = np.zeros(W.shape, np.uint8)
    books = np.zeros((W.shape[1], levels), np.float32)
    for j in range(W.shape[1]):
        w = W[:, j]
        nz = w[w != 0]
        c = np.quantile(nz, np.linspace(0.1, 0.9, levels - 1)) if nz.size else np.zeros(levels - 1)
        for _ in range(iters):
            a = np.abs(nz[:, None] - c[None, :]).argmin(1) if nz.size else np.array([], int)
            c = np.array([nz[a == k].mean() if np.any(a == k) else c[k] for k in range(levels - 1)])
        book = np.concatenate([[0.0], c]).astype(np.float32)
        codes[:, j] = np.abs(w[:, None] - book[None, :]).argmin(1)
        books[j] = book
    return codes, books


def pack(codes, books):
    """Leader Ak byte layout: 2-bit codes, 4 per byte (low bits first, column-major per row), then float32 books."""
    flat = codes.reshape(-1)
    pad = (-len(flat)) % 4
    flat = np.concatenate([flat, np.zeros(pad, np.uint8)])
    b = flat[0::4] | (flat[1::4] << 2) | (flat[2::4] << 4) | (flat[3::4] << 6)
    return lzma.compress(b.astype(np.uint8).tobytes() + books.astype(np.float32).tobytes(), preset=9 | lzma.PRESET_EXTREME)


def dequantize(codes, books):
    return books[np.arange(books.shape[0])[None, :], codes]  # W[i, j] = books[j, codes[i, j]]


def prune(W, keep_frac):
    """Zero all but the largest-|w| fraction of weights."""
    thr = np.quantile(np.abs(W), 1 - keep_frac)
    W = W.copy()
    W[np.abs(W) < thr] = 0
    return W


def knn_overlap(A, B, k=10):
    ia = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(A).kneighbors(A)[1][:, 1:]
    ib = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(B).kneighbors(B)[1][:, 1:]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(ia, ib)]))


def parse(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="0,1,30-36,50-56")
    ap.add_argument("--test", default="37,57,60")
    ap.add_argument("--lam", type=float, default=1.0)
    a = ap.parse_args()
    tr, te = parse(a.train), parse(a.test)
    Xs, Es = zip(*(load(s) for s in tr))
    X = vstack([features(t) for t in Xs]).tocsr()
    E = np.vstack(Es)
    pca = PCA(DIM, random_state=0).fit(E)
    W = ridge(X, pca.transform(E), a.lam)
    print(f"trained on {X.shape[0]} texts; W {W.shape}")
    np.save(ROOT / "data" / "distill_W.npy", W)
