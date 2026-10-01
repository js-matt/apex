"""Re-implementation of the leader's long path after feature building (181049 L380-L414) on dumped features,
plus variants of the spectral block and the back-end. Trusted code only; inputs are feat_181049_<ss>.npz dumps.
    .venv/bin/python tools/long_lab.py --subsets 30-37
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_lab import parse  # noqa: E402
from noise_lab import score  # noqa: E402
from np_umap import fuzzy_graph, knn as np_knn, umap_embed  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH, FEAT = ROOT / "data" / "bench", ROOT / "data" / "sandbox_out" / "feat"
BIG = 10 ** 6
_al, _am, _an = re.compile(r"http\S+|www\S+|https\S+"), re.compile(r"@\w+"), re.compile(r"#(\w+)")
_ao, _ap = re.compile(r"[^\w\s]"), re.compile(r"\s+")
SCRIPTS = ((1024, 1327, 1), (19968, 40959, 2), (12352, 12543, 2), (44032, 55215, 2), (1536, 1791, 3), (1424, 1535, 4), (3584, 3711, 5), (2304, 2431, 6))


def ar(t):
    t = _an.sub(r"\1", _am.sub(" ", _al.sub(" ", t.lower())))
    return _ap.sub(" ", _ao.sub(" ", t)).strip()


def script_id(t):
    n, c = len(t) or 1, [0] * 7
    for ch in t:
        o = ord(ch)
        for lo, hi, s in SCRIPTS:
            if lo <= o <= hi:
                c[s] += 1
                break
    s = max(range(1, 7), key=lambda i: c[i])
    return s if c[s] * 8 >= n else 0


def script_merge(lab, sid, lo=.05, hi=.4, pur=.55, ms=3):
    lab = lab.copy()
    for s in range(1, 7):
        frac = float((sid == s).mean())
        if frac < lo or frac > hi:
            continue
        pure = [int(c) for c in np.unique(lab[sid == s]) if (lab == c).sum() >= ms and (sid[lab == c] == s).mean() >= pur]
        if len(pure) >= 2:
            lab[np.isin(lab, pure)] = pure[0]
    return lab


def protect(d, lab, Z, n, cap=3, lo=.012, hi=.12, ms=40):
    d, cand = d.copy(), []
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
        d[m] = -1.0
    return d


def merge_top_pair(docs, lab, thr=.7):
    ids = np.unique(lab)
    pseudo = [" ".join(np.asarray(docs)[lab == c][:80]) for c in ids]
    try:
        H = normalize(TfidfVectorizer(max_features=4000, stop_words="english", dtype=np.float32).fit_transform(pseudo))
    except ValueError:
        return lab
    S = (H @ H.T).toarray()
    np.fill_diagonal(S, 0)
    a, b = divmod(int(S.argmax()), len(ids))
    lab = lab.copy()
    if S[a, b] >= thr:
        lab[lab == ids[b]] = ids[a]
    return lab


def rescue(lab, Z, t=.77, ms=5):
    lab = lab.copy()
    ids, c = np.unique(lab, return_counts=True)
    big, small = ids[c >= ms], ids[c < ms]
    if len(big) < 1 or len(small) < 1:
        return lab
    M = normalize(np.stack([Z[lab == i].mean(0) for i in big]))
    for i in small:
        idx = np.where(lab == i)[0]
        s = Z[idx] @ M.T
        ok = s.max(1) >= t
        lab[idx[ok]] = big[s.argmax(1)[ok]]
    return lab


def mnn(lab, Z, stages=((.8, 1, 1), (.86, 5, 4))):
    lab = lab.copy()
    for t, ms, rounds in stages:
        for _ in range(rounds):
            ids, c = np.unique(lab, return_counts=True)
            s = ids[c <= ms]
            if len(s) < 2:
                break
            C = normalize(np.stack([Z[lab == i].mean(0) for i in s]))
            d, I = NearestNeighbors(n_neighbors=2, metric="cosine").fit(C).kneighbors(C)
            hit = 0
            for a in range(len(s)):
                b = int(I[a, 1])
                if 1 - d[a, 1] >= t and int(I[b, 1]) == a and s[a] < s[b]:
                    lab[lab == s[b]] = s[a]
                    hit = 1
            if not hit:
                break
            _, lab = np.unique(lab, return_inverse=True)
    return lab


def b1(rel):
    beta = float(np.clip((rel - .1) / (.32 - .1), 0, 1))
    return int(round(40 + 180 * beta)), .25 + .15 * beta


def spectral(Z, dim, k=15, weighted=False):
    if weighted:  # UMAP fuzzy-simplicial-set weights instead of a binary kNN graph
        idx, dist = np_knn(Z, k)
        P = fuzzy_graph(idx, dist, k).tocsr()
        P = P + diags(np.ones(Z.shape[0]))
    else:
        F = NearestNeighbors(n_neighbors=min(k, len(Z) - 1), metric="cosine").fit(Z).kneighbors_graph(Z, mode="connectivity")
        P = ((F + F.T) > 0).astype(np.float32)
    deg = np.asarray(P.sum(1)).ravel()
    deg[deg == 0] = 1
    h = 1 / np.sqrt(deg)
    M = csr_matrix(P.multiply(h[:, None]).multiply(h[None, :]))
    n = len(Z)
    _, V = eigsh(M, k=min(dim, n - 2), which="LA", tol=1e-3, maxiter=300, v0=np.full(n, 1 / np.sqrt(n), np.float32))
    return normalize(V.astype(np.float32))


def backend(G, docs, sid, a, band, k_cut=28, sing=.20, kB0=None, use_mnn=True):
    n = len(G)
    kB0 = kB0 or (14 if band else 12)
    nn = NearestNeighbors(n_neighbors=min(kB0 + 1, n - 1), metric="cosine").fit(G)
    D = nn.kneighbors(G)[0][:, 1:]
    smp = np.random.default_rng(0).choice(n, min(1500, n), replace=False)
    pw = 1 - G[smp] @ G[smp].T
    rel = float(D[:, -1].mean()) / max(float(pw[np.triu_indices(len(smp), 1)].mean()), 1e-9)
    L = linkage(G, "average", "cosine")
    E = fcluster(L, min(k_cut, n - 1), "maxclust").astype(np.int64)
    E = script_merge(E, sid)
    flat = a < .03
    if flat:
        E = merge_top_pair(docs, E)
    W = int(n * sing)
    if W > 0:
        O = D[:, -1].copy()
        O[sid > 0] = -1
        if flat:
            O = protect(O, E, G, n)
        M = np.argpartition(-O, W - 1)[:W]
        M = M[O[M] >= 0]
        E = E.copy()
        E[M] = BIG + np.arange(len(M))
    if not band:
        E = np.unique(rescue(E, G, .77), return_inverse=True)[1]
    if use_mnn:
        E = np.unique(mnn(E, G), return_inverse=True)[1]
    for g in np.unique(E):
        m = E == g
        if m.sum() > .15 * n:
            _, sub = np.unique(fcluster(L, b1(rel)[0], "maxclust")[m], return_inverse=True)
            E[m] = sub + E.max() + 1
    return np.unique(E, return_inverse=True)[1]


def load(s):
    rows = [json.loads(l) for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")]
    texts, gt = [r["text"] for r in rows], np.array([r["label"] for r in rows])
    z = np.load(FEAT / f"feat_181049_{s:02d}.npz", allow_pickle=False)
    keep = z["keep"]
    docs = [ar(t) for t in texts]
    docs = [d for d, k in zip(docs, keep) if k]
    L40 = [len(d) for d in docs if len(d) >= 40]
    AI = float(np.median(L40 if len(L40) >= 50 else [len(d) for d in docs]))
    Zt = z["comp_t"] - z["comp_t"].mean(0, keepdims=True)
    a = float(TruncatedSVD(1, random_state=42).fit(Zt).explained_variance_ratio_[0])
    sid = np.array([script_id(d) for d in docs], np.int16)
    alt = BENCH / f"subset_{s:02d}.alt_ms3.npy"  # lower-noise GT (min_samples=3), closer to live noise rates
    gtB = np.load(alt) if alt.exists() else gt
    return texts, (gt, gtB), keep, docs, sid, a, 230 < AI < 330, z


def full(gt, keep, lab):
    out = np.full(len(gt[0]), -1)
    out[keep] = lab
    return out


def run(job):
    s, variants = job
    texts, gt, keep, docs, sid, a, band, z = load(s)
    sc = lambda lab: (score(gt[0], lab), score(gt[1], lab))
    base = z["comp_Az"]
    res = {"leader (actual)": sc(z["pred"])}
    res["re-impl on dumped G"] = sc(full(gt, keep, backend(z["G"], docs, sid, a, band)))
    dim = z["comp_A_"].shape[1]
    for name, v in variants.items():
        blocks = [base]
        if v.get("spec", True):
            sp = spectral(base, v.get("dim", int(dim * v.get("dim_mul", 1))), v.get("k", 15), v.get("weighted", False))
            blocks.append(sp * v.get("w", 1.0))
        if v.get("umap"):
            Y = umap_embed(base, n_neighbors=v.get("unn", 15), n_components=v["umap"], n_epochs=v.get("uep", 200))
            Y = normalize(Y - Y.mean(0))
            blocks.append(Y * v.get("uw", 1.0))
        G = normalize(np.hstack(blocks))
        lab = backend(G, docs, sid, a, band, k_cut=v.get("cut", 28), sing=v.get("sing", .20), use_mnn=v.get("mnn", True))
        res[name] = sc(full(gt, keep, lab))
    return s, res


VARIANTS = {
    "same spectral (recomputed)": {},
    "spectral w=0.7": {"w": .7},
    "spectral w=1.4": {"w": 1.4},
    "spectral k=25": {"k": 25},
    "spectral k=10": {"k": 10},
    "spectral fuzzy-weighted": {"weighted": True},
    "spectral dim x1.5": {"dim_mul": 1.5},
    "no spectral": {"spec": False},
    "+umap5 w=1": {"umap": 5},
    "+umap10 w=1": {"umap": 10},
    "cut 36": {"cut": 36},
    "cut 22": {"cut": 22},
    "sing .15": {"sing": .15},
    "sing .25": {"sing": .25},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", default="30-37")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    subs = parse(a.subsets)
    vs = {k: v for k, v in VARIANTS.items() if not a.only or re.search(a.only, k)}
    with ProcessPoolExecutor(8) as ex:
        out = dict(ex.map(run, [(s, vs) for s in subs]))
    ref = "same spectral (recomputed)" if "same spectral (recomputed)" in vs else "leader (actual)"
    for key in out[subs[0]]:
        cols = []
        for j, nm in ((0, "A"), (1, "B")):
            m = np.mean([out[s][key][j] for s in subs])
            r = np.mean([out[s][ref][j] for s in subs])
            wins = sum(out[s][key][j] > out[s][ref][j] for s in subs)
            cols.append(f"{nm} {m:.4f} {(m / r - 1) * 100:+.1f}% ({wins}/{len(subs)})")
        print("  ".join(cols) + f"  {key}", flush=True)


if __name__ == "__main__":
    main()
