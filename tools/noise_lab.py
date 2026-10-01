"""Predict the validator's HDBSCAN noise (-1) from cheap per-text features, then score the
labelling "predicted noise -> one -1 cluster, rest -> leader labels".

Inputs: leader feature dumps (tools/dump_features.py, sandboxed) data/sandbox_out/feat/feat_<sol>_<ss>.npz
plus the benchmark ground truth. Only trusted code runs here.
    .venv/bin/python tools/noise_lab.py --sol 181049 --subsets 30-37,80-83
Leave-one-subset-out: each subset is scored by a model trained on the other subsets.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_lab import parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH, FEAT = ROOT / "data" / "bench", ROOT / "data" / "sandbox_out" / "feat"
_tok = re.compile(r"[^\W\d_][\w\-']+", re.UNICODE)


def score(gt, pred):
    return (max(0.0, adjusted_rand_score(gt, pred)) + normalized_mutual_info_score(gt, pred)) / 2


def rankpct(x):
    return np.argsort(np.argsort(x)) / max(len(x) - 1, 1)


def point_features(texts, G, keep, pred):
    """Per-text features; rows for texts the leader dropped (keep=False) get G-based features of 0."""
    n = len(texts)
    Gf = np.zeros((n, G.shape[1]), np.float32)
    Gf[keep] = normalize(G)
    cv = CountVectorizer(stop_words="english", min_df=1, token_pattern=r"(?u)\b[^\W\d_][\w\-']+\b")
    X = cv.fit_transform(texts)
    df = np.asarray((X > 0).sum(0)).ravel()
    idf = np.log(n / (1 + df))
    Xb = (X > 0).astype(np.float32)
    ntok = np.asarray(Xb.sum(1)).ravel()
    mean_idf = np.asarray(Xb @ idf).ravel() / np.maximum(ntok, 1)
    mx = Xb.multiply(idf).tocsr().max(1).toarray().ravel()
    # in-subset document frequency of each text's rarest-but-shared word: topical anchors
    shared = Xb.multiply((df >= 5) & (df <= n * .05)).tocsr()
    n_topical = np.asarray(shared.sum(1)).ravel()
    L = np.array([len(t) for t in texts], np.float32)
    nw = np.array([len(_tok.findall(t)) for t in texts], np.float32)
    k = 30
    d, nb = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Gf).kneighbors(Gf)
    d, nb = d[:, 1:], nb[:, 1:]
    ids, inv, cnt = np.unique(pred, return_inverse=True, return_counts=True)
    csize = cnt[inv].astype(np.float32)
    same = (pred[nb[:, :10]] == pred[:, None]).mean(1)
    dens = d[:, 9]
    nb_dens = dens[nb[:, :10]].mean(1)
    # similarity to own cluster centroid
    cent = np.zeros((len(ids), Gf.shape[1]), np.float32)
    np.add.at(cent, inv, Gf)
    cent = normalize(cent)
    own = (Gf * cent[inv]).sum(1)
    feats = np.column_stack([
        np.log1p(L), np.log1p(nw), np.log1p(ntok), mean_idf, mx, n_topical,
        d[:, 0], d[:, 4], d[:, 9], d[:, 29], rankpct(d[:, 9]), rankpct(d[:, 0]),
        dens / np.maximum(nb_dens, 1e-6), np.log(csize), (csize == 1).astype(np.float32), same, own,
        np.full(n, np.median(L)), np.full(n, d[:, 9].mean()),
    ]).astype(np.float32)
    return feats


def load(sol, s):
    rows = [json.loads(l) for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")]
    texts, gt = [r["text"] for r in rows], np.array([r["label"] for r in rows])
    z = np.load(FEAT / f"feat_{sol}_{s:02d}.npz", allow_pickle=False)
    pred = z["pred"].astype(np.int64)
    return texts, gt, pred, point_features(texts, z["G"], z["keep"], pred)


def evaluate(gt, pred, p, fracs):
    out = {}
    order = np.argsort(-p)
    for f in fracs:
        q = pred.copy()
        q[order[:int(f * len(q))]] = -(10 ** 9)
        out[f] = score(gt, q)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sol", default="181049")
    ap.add_argument("--subsets", default="30-37,80-83")
    a = ap.parse_args()
    subs = parse(a.subsets)
    with ProcessPoolExecutor(8) as ex:
        data = dict(zip(subs, ex.map(load, [a.sol] * len(subs), subs)))
    fracs = (0, .05, .1, .15, .2, .25, .3)
    res = {}
    for s in subs:
        tr = [t for t in subs if t != s]
        Xtr = np.vstack([data[t][3] for t in tr])
        ytr = np.concatenate([data[t][1] == -1 for t in tr])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=.05, max_leaf_nodes=31, random_state=0).fit(Xtr, ytr)
        texts, gt, pred, X = data[s]
        p = clf.predict_proba(X)[:, 1]
        y = gt == -1
        top = np.argsort(-p)[:int(.15 * len(p))]
        res[s] = dict(auc=roc_auc_score(y, p), prec15=y[top].mean(), base=y.mean(), **{f"f={f}": v for f, v in evaluate(gt, pred, p, fracs).items()})
        print(f"subset {s}: auc={res[s]['auc']:.3f} base={res[s]['base']:.2f} prec@15%={res[s]['prec15']:.2f} " +
              " ".join(f"{f}:{res[s][f'f={f}']:.4f}" for f in fracs), flush=True)
    for grp, ss in (("reddit", [s for s in subs if 30 <= s < 40]), ("arxiv", [s for s in subs if s >= 80])):
        if ss:
            print(f"== {grp}: auc={np.mean([res[s]['auc'] for s in ss]):.3f} " +
                  " ".join(f"{f}:{np.mean([res[s][f'f={f}'] for s in ss]):.4f}" for f in fracs))


if __name__ == "__main__":
    main()
