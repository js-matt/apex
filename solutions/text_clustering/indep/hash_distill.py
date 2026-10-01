"""Distill mpnet neighbourhoods into a hashed bag-of-features embedding (tiny enough to ship).

    ~/.venvs/apex-train/bin/python hash_distill.py prep
    ~/.venvs/apex-train/bin/python hash_distill.py train --bw 18 --bc 0 --d 64 --out h_w18_d64

Embedding: e = normalize( sum_f log1p(count_f) * softplus(g[f]) * T[f] ) over hashed word (and char) features.
Loss: within-subset neighbour distillation, KL(softmax(S_teacher/tt) || softmax(S_student/ts)) per row.
"""
import argparse
import glob
import json
import math
import os
import time

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer

DATA = "/home/abc/apex/data"
OUT = f"{DATA}/indep"
EVAL = {f"subset_{i}" for i in list(range(30, 38)) + list(range(80, 84))}


def featurizers(bw, bc):
    """Word unigram hashing (2**bw buckets) and optional char_wb 3-5 grams (2**bc buckets)."""
    f = [HashingVectorizer(n_features=2 ** bw, alternate_sign=False, norm=None, lowercase=True,
                           token_pattern=r"(?u)\b\w+\b", dtype=np.float32)]
    if bc:
        f.append(HashingVectorizer(n_features=2 ** bc, alternate_sign=False, norm=None, lowercase=True,
                                   analyzer="char_wb", ngram_range=(3, 5), dtype=np.float32))
    return f


def featurize(texts, bw, bc):
    mats = [h.transform(texts) for h in featurizers(bw, bc)]
    X = sp.hstack(mats).tocsr() if len(mats) > 1 else mats[0].tocsr()
    X.data = np.log1p(X.data)
    return X


def prep():
    eval_texts = set()
    for s in EVAL:
        eval_texts.update(json.loads(l)["text"] for l in open(f"{DATA}/bench/{s}.jsonl", encoding="utf-8"))
    texts, embs, groups, seen = [], [], [], set()
    g = 0
    for f in sorted(glob.glob(f"{DATA}/bench/subset_*.jsonl")):
        s = os.path.basename(f)[:-6]
        if s in EVAL or not os.path.exists(f"{DATA}/bench/{s}.emb.npy"):
            continue
        E = np.load(f"{DATA}/bench/{s}.emb.npy")
        n0 = len(texts)
        for r, e in zip((json.loads(l) for l in open(f, encoding="utf-8")), E):
            t = r["text"]
            if t in seen or t in eval_texts or not t.strip():
                continue
            seen.add(t)
            texts.append(t)
            embs.append(e)
            groups.append(g)
        if len(texts) > n0:
            g += 1
    E = np.load(f"{DATA}/arxiv/pool.emb.npy")
    for r, e in zip((json.loads(l) for l in open(f"{DATA}/arxiv/pool.jsonl", encoding="utf-8")), E):
        t = r["title"]
        if t in seen or t in eval_texts or not t.strip():
            continue
        seen.add(t)
        texts.append(t)
        embs.append(e)
        groups.append(g)
    with open(f"{OUT}/htrain.jsonl", "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps(t) + "\n")
    np.savez(f"{OUT}/htrain.npz", emb=np.stack(embs).astype(np.float16), group=np.array(groups))
    print(f"{len(texts)} texts in {g + 1} groups; sizes {np.bincount(groups)}")


def train(a):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    torch.set_num_threads(a.threads)
    texts = [json.loads(l) for l in open(f"{OUT}/htrain.jsonl", encoding="utf-8")]
    z = np.load(f"{OUT}/htrain.npz")
    Y = torch.tensor(z["emb"].astype(np.float32))
    group = z["group"]
    t0 = time.time()
    X = featurize(texts, a.bw, a.bc)
    nf = X.shape[1]
    print(f"featurized {X.shape} nnz/row {X.nnz / X.shape[0]:.1f} in {time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    # hold out 2 social groups + 2k arXiv for validation of neighbour recall
    groups = np.unique(group)
    val_groups = [int(v) for v in a.val.split(",")] if a.val else [groups[3], groups[17]]
    tr_mask = ~np.isin(group, val_groups)
    if a.groups:
        tr_mask &= np.isin(group, [int(v) for v in a.groups.split(",")])

    T = torch.nn.Parameter(torch.randn(nf, a.d) * 0.1)
    gw = torch.nn.Parameter(torch.zeros(nf))
    lts = torch.nn.Parameter(torch.tensor(math.log(a.ts)))
    opt = torch.optim.Adam([T, gw, lts], lr=a.lr)

    def embed(idx):
        sub = X[idx]
        coo = sub.tocoo()
        ii = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        v = torch.tensor(coo.data) * F.softplus(gw[ii[1]] + 1.0)
        A = torch.sparse_coo_tensor(ii, v, (len(idx), nf))
        return F.normalize(torch.sparse.mm(A, T), dim=-1)

    def batch_idx(pool_mask):
        g = rng.choice(np.unique(group[pool_mask]))
        cand = np.nonzero((group == g) & pool_mask)[0]
        return rng.choice(cand, min(a.bs, len(cand)), replace=False)

    def val_recall():
        with torch.no_grad():
            rs = []
            for g in val_groups:
                idx = np.nonzero(group == g)[0][:3000]
                E = embed(idx)
                St = Y[idx] @ Y[idx].T
                Ss = E @ E.T
                St.fill_diagonal_(-9)
                Ss.fill_diagonal_(-9)
                kt = St.topk(15, dim=1).indices
                ks = Ss.topk(15, dim=1).indices
                rs.append(np.mean([len(set(p.tolist()) & set(q.tolist())) / 15 for p, q in zip(kt, ks)]))
        return float(np.mean(rs))

    for step in range(1, a.steps + 1):
        idx = batch_idx(tr_mask)
        E = embed(idx)
        y = Y[idx]
        St = (y @ y.T) / a.tt
        Ss = (E @ E.T) / lts.exp()
        eye = torch.eye(len(idx), dtype=torch.bool)
        St = St.masked_fill(eye, -1e9)
        Ss = Ss.masked_fill(eye, -1e9)
        loss = F.kl_div(Ss.log_softmax(1), St.log_softmax(1), log_target=True, reduction="batchmean")
        opt.zero_grad()
        loss.backward()
        opt.step()
        for gr in opt.param_groups:
            gr["lr"] = a.lr * min(1.0, 0.5 * (1 + math.cos(math.pi * step / a.steps)) + 0.02)
        if step % 250 == 0 or step == a.steps:
            print(f"step {step} loss {loss.item():.4f} ts {lts.exp().item():.4f} val recall {val_recall():.3f} "
                  f"{time.time() - t0:.0f}s", flush=True)
            np.savez(f"{OUT}/{a.out}.npz", T=T.detach().numpy(), g=(F.softplus(gw + 1.0)).detach().numpy(),
                     meta=np.array([a.bw, a.bc, a.d]))
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--bw", type=int, default=18)
    ap.add_argument("--bc", type=int, default=0)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tt", type=float, default=0.03)
    ap.add_argument("--ts", type=float, default=0.05)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="h")
    ap.add_argument("--groups", default="")  # restrict training to these group ids
    ap.add_argument("--val", default="")  # validation group ids
    a = ap.parse_args()
    prep() if a.cmd == "prep" else train(a)
