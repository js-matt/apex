"""Experiments for the <50k-char version: references, PCA ceilings, hashed distilled embeddings."""
import importlib.util
import os
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench  # noqa: E402

BENCH = bench.BENCH


def _mod(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def tfidf_svd(texts, dim=100, ngram=(1, 1), min_df=2, max_df=0.5, stop="english"):
    X = TfidfVectorizer(sublinear_tf=True, min_df=min_df, max_df=max_df, stop_words=stop, ngram_range=ngram,
                        dtype=np.float32).fit_transform(texts)
    return normalize(TruncatedSVD(dim, random_state=0).fit_transform(X)).astype(np.float32)


_PCA = {}


def pca_exact(d):
    def f(texts):
        # look up the exact mpnet rows by text (bench files are aligned)
        key = hash(texts[0])
        if "U" not in _PCA:
            _PCA["U"] = np.load("/home/abc/apex/data/indep/pca768.npy")
        for s in bench.REDDIT + bench.ARXIV:
            t, _ = bench.load(s)
            if t[0] == texts[0] and len(t) == len(texts):
                E = np.load(f"{BENCH}/{s}.emb.npy")
                return normalize(E @ _PCA["U"][:, :d]).astype(np.float32)
        raise KeyError(key)
    return f


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "refs":
        base = _mod("/home/abc/apex/apex/shared/competition/src/competition/text_clustering/baseline.py", "base")
        user = _mod("/home/abc/apex/solutions/text_clustering/solution.py", "usersol")
        bench.LABELERS["official_baseline"] = base.cluster_texts
        bench.LABELERS["user_solution_py"] = user.cluster_texts
        bench.main("official_baseline", quiet=True)
        bench.main("user_solution_py", quiet=True)
        bench.EMBEDDERS["tfidf_svd100"] = tfidf_svd
        bench.main("tfidf_svd100", quiet=True)
    elif what == "pca":
        d = np.load("/home/abc/apex/data/indep/train.npz")["emb"].astype(np.float32)
        C = np.cov(d[np.random.default_rng(0).choice(len(d), 60000, replace=False)].T)
        w, U = np.linalg.eigh(C)
        U = U[:, ::-1].astype(np.float32)
        np.save("/home/abc/apex/data/indep/pca768.npy", U)
        w = w[::-1]
        print("variance share:", {k: round(float(w[:k].sum() / w.sum()), 3) for k in (8, 16, 32, 64, 128)})
        for k in map(int, sys.argv[2:]):
            bench.EMBEDDERS[f"pca{k}"] = pca_exact(k)
            bench.main(f"pca{k}", quiet=True)


# ------------------------------------------------------------------ hashed distilled embeddings
_HD = {}


def hd_embed(name):
    import hash_distill as H

    def f(texts):
        if name not in _HD:
            z = np.load(f"/home/abc/apex/data/indep/{name}.npz")
            bw, bc, d = (int(v) for v in z["meta"])
            _HD[name] = (bw, bc, (z["g"][:, None] * z["T"]).astype(np.float32))
        bw, bc, W = _HD[name]
        return normalize(H.featurize(texts, bw, bc) @ W).astype(np.float32)
    return f


def tfidf_wc(texts):
    import scipy.sparse as sp
    a = dict(sublinear_tf=True, min_df=2, dtype=np.float32)
    Xw = TfidfVectorizer(max_df=0.5, stop_words="english", **a).fit_transform(texts)
    Xc = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_df=0.3, **a).fit_transform(texts)
    return normalize(sp.hstack([Xw, Xc]).tocsr())


def mix(f_dense, w):
    """Cosine = w * dense-cos + (1-w) * tfidf-cos, as one sparse matrix."""
    import scipy.sparse as sp

    def f(texts):
        return sp.hstack([sp.csr_matrix(f_dense(texts) * np.sqrt(w)), tfidf_wc(texts) * np.sqrt(1 - w)]).tocsr()
    return f


# ------------------------------------------------------------------ quantization
def kmeans1d(x, k, iters=30):
    c = np.quantile(x, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        a = np.abs(x[:, None] - c[None]).argmin(1)
        for j in range(k):
            if (a == j).any():
                c[j] = x[a == j].mean()
    a = np.abs(x[:, None] - c[None]).argmin(1)
    return c.astype(np.float16), a.astype(np.uint8)


def quantize(W, bits=4):
    """Per-column 2**bits-level codebooks. Returns (codebooks [d, k] fp16, codes [B, d] uint8)."""
    k = 2 ** bits
    cbs, codes = [], []
    for j in range(W.shape[1]):
        c, a = kmeans1d(W[:, j].astype(np.float64), k)
        cbs.append(c)
        codes.append(a)
    return np.stack(cbs), np.stack(codes, 1)


def dequantize(cbs, codes):
    return np.take_along_axis(cbs.T.astype(np.float32), codes.astype(np.int64), 0) if False else \
        np.stack([cbs[j].astype(np.float32)[codes[:, j]] for j in range(codes.shape[1])], 1)


def hd_quant(name, bits=4):
    import hash_distill as H

    def f(texts):
        key = (name, bits)
        if key not in _HD:
            z = np.load(f"/home/abc/apex/data/indep/{name}.npz")
            bw, bc, d = (int(v) for v in z["meta"])
            W = (z["g"][:, None] * z["T"]).astype(np.float32)
            _HD[key] = (bw, bc, dequantize(*quantize(W, bits)))
        bw, bc, W = _HD[key]
        return normalize(H.featurize(texts, bw, bc) @ W).astype(np.float32)
    return f


def centered(f_dense):
    def f(texts):
        E = f_dense(texts)
        return normalize(E - E.mean(0)).astype(np.float32)
    return f


def tfidf_eq(texts, wc=0.5):
    """Word TF-IDF and char_wb(3,5) TF-IDF, each L2-normalized, mixed by cosine weight wc."""
    import scipy.sparse as sp
    a = dict(sublinear_tf=True, min_df=2, dtype=np.float32)
    Xw = normalize(TfidfVectorizer(max_df=0.5, stop_words="english", **a).fit_transform(texts))
    Xc = normalize(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_df=0.3, **a).fit_transform(texts))
    return sp.hstack([Xw * np.sqrt(1 - wc), Xc * np.sqrt(wc)]).tocsr()


def mix_eq(f_dense, w, lsa=0.0):
    """cos = w*dense + lsa*LSA(100) + (1-w-lsa)*tfidf_eq."""
    import scipy.sparse as sp

    def f(texts):
        X = tfidf_eq(texts)
        parts = [X * np.sqrt(1 - w - lsa)]
        if w > 0:
            parts.append(sp.csr_matrix(f_dense(texts) * np.sqrt(w)))
        if lsa > 0:
            L = normalize(TruncatedSVD(100, random_state=0).fit_transform(X))
            parts.append(sp.csr_matrix(L * np.sqrt(lsa)))
        return sp.hstack(parts).tocsr()
    return f
