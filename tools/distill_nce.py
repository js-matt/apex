"""Distil mpnet into a hashed-n-gram linear projection with a neighbourhood-preserving loss.

Same input features as the leader's BD projection (leader BS() normalisation, hashed word 1-2gram 8192 +
char_wb 3-5gram 4096, log1p, L2), output W (12288 x dim). Loss: in-batch InfoNCE between the projected
text and a learned linear map of its mpnet embedding, plus cosine regression to mpnet's top-`dim` PCA.
Titles in the --exclude subsets (the test set) are dropped from training.
    .venv-gt/bin/python tools/distill_nce.py --domain arxiv --exclude 80-87 --dim 48 --out data/W_arxiv_48.npy
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_lab import hashed, parse  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data" / "bench"


def load_domain(domain, exclude):
    ex = {json.loads(l)["text"] for s in exclude for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")}
    if domain == "arxiv":
        pool = [json.loads(l)["title"] for l in open(ROOT / "data" / "arxiv" / "pool.jsonl", encoding="utf-8")]
        emb = np.load(ROOT / "data" / "arxiv" / "pool.emb.npy")
    else:  # bench subsets of the given ids (reddit-like / X)
        pool, embs = [], []
        for s in parse(domain):
            pool += [json.loads(l)["text"] for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")]
            embs.append(np.load(BENCH / f"subset_{s:02d}.emb.npy"))
        emb = np.vstack(embs)
    keep = np.array([t not in ex for t in pool])
    return [t for t, k in zip(pool, keep) if k], emb[keep].astype(np.float32)


def to_torch(X):
    X = X.tocoo()
    return torch.sparse_coo_tensor(np.vstack([X.row, X.col]), X.data, X.shape).coalesce().to_sparse_csr()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="arxiv")
    ap.add_argument("--exclude", default="80-87")
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=1.0, help="weight of the PCA cosine-regression term")
    ap.add_argument("--l2", type=float, default=1e-6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(0)
    texts, E = load_domain(a.domain, parse(a.exclude) if a.exclude else [])
    X = hashed(texts)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    P = PCA(a.dim, random_state=0).fit(E)
    T = torch.tensor(P.transform(E), dtype=torch.float32)
    Et = torch.tensor(E)
    n = X.shape[0]
    print(f"train {n} texts, {X.shape[1]} features -> {a.dim}", flush=True)
    W = torch.nn.Parameter(torch.randn(X.shape[1], a.dim) * 0.01)
    M = torch.nn.Parameter(torch.tensor(P.components_.T.copy(), dtype=torch.float32))  # mpnet -> dim
    opt = torch.optim.Adam([W, M], lr=3e-3)
    bs = 1024
    for ep in range(a.epochs):
        perm = np.random.default_rng(ep).permutation(n)
        tot = 0.0
        for i in range(0, n - bs + 1, bs):
            b = perm[i:i + bs]
            z = torch.nn.functional.normalize(to_torch(X[b]) @ W, dim=1)
            t = torch.nn.functional.normalize(Et[b] @ M, dim=1)
            logits = z @ t.T / a.tau
            lbl = torch.arange(len(b))
            loss = (torch.nn.functional.cross_entropy(logits, lbl) + torch.nn.functional.cross_entropy(logits.T, lbl)) / 2
            if a.reg:
                loss = loss + a.reg * (1 - (z * torch.nn.functional.normalize(T[b], dim=1)).sum(1)).mean()
            loss = loss + a.l2 * (W ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if ep % 5 == 4 or ep == a.epochs - 1:
            print(f"epoch {ep + 1}: loss {tot / (n // bs):.4f}", flush=True)
    np.save(a.out, W.detach().numpy().astype(np.float32))


if __name__ == "__main__":
    main()
