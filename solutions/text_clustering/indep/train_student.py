"""Distill all-mpnet-base-v2 into a small BERT student (torch, CPU).

    ~/.venvs/apex-train/bin/python train_student.py prep
    ~/.venvs/apex-train/bin/python train_student.py train --init paraphrase-MiniLM-L3-v2 --out l3 --epochs 4

Exports student weights as .npz in mpnet_np.Bert's packed order (plus head) for numpy inference.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpnet_np  # noqa: E402

DATA = "/home/abc/apex/data"
OUT = f"{DATA}/indep"
MODELS = f"{DATA}/models"
EVAL = {f"subset_{i}" for i in list(range(30, 38)) + list(range(80, 84))}


def prep(extra_glob=None):
    os.makedirs(OUT, exist_ok=True)
    eval_texts = set()
    for s in EVAL:
        eval_texts.update(json.loads(l)["text"] for l in open(f"{DATA}/bench/{s}.jsonl", encoding="utf-8"))
    texts, embs, seen = [], [], set()

    def add(t, e):
        if t in seen or t in eval_texts or not t.strip():
            return
        seen.add(t)
        texts.append(t)
        embs.append(e)

    for f in sorted(glob.glob(f"{DATA}/bench/subset_*.jsonl")):
        s = os.path.basename(f)[:-6]
        if s in EVAL or not os.path.exists(f"{DATA}/bench/{s}.emb.npy"):
            continue
        E = np.load(f"{DATA}/bench/{s}.emb.npy")
        for r, e in zip((json.loads(l) for l in open(f, encoding="utf-8")), E):
            add(r["text"], e)
    n_social = len(texts)
    E = np.load(f"{DATA}/arxiv/pool.emb.npy")
    for r, e in zip((json.loads(l) for l in open(f"{DATA}/arxiv/pool.jsonl", encoding="utf-8")), E):
        add(r["title"], e)
    for f in sorted(glob.glob(extra_glob or "/nonexistent")):
        E = np.load(f.replace(".jsonl", ".emb.npy"))
        for r, e in zip((json.loads(l) for l in open(f, encoding="utf-8")), E):
            add(r["text"], e)
    print(f"social {n_social}  arxiv {len(texts) - n_social}  total {len(texts)}")
    vocab = json.load(open(f"{MODELS}/paraphrase-MiniLM-L3-v2/tokenizer.json", encoding="utf-8"))["model"]["vocab"]
    tok = mpnet_np.WordPiece(vocab)
    t = time.time()
    ids = [tok.encode(x, 256, 101, 102) for x in texts]
    print(f"tokenized in {time.time() - t:.0f}s, mean len {np.mean([len(i) for i in ids]):.1f}")
    lens = np.array([len(i) for i in ids], dtype=np.int32)
    flat = np.concatenate([np.array(i, dtype=np.int32) for i in ids])
    np.savez(f"{OUT}/train.npz", flat=flat, lens=lens, emb=np.stack(embs).astype(np.float16))


# ------------------------------------------------------------------ torch student
def build(init, max_pos=None):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    cfg = json.load(open(f"{MODELS}/{init}/config.json"))
    W = mpnet_np.pack_bert(mpnet_np.load_safetensors(f"{MODELS}/{init}/model.safetensors"),
                           cfg["num_hidden_layers"])
    nl, nh, eps = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["layer_norm_eps"]

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.p = nn.ParameterList([nn.Parameter(torch.tensor(w)) for w in W])
            hid = W[0].shape[1]
            self.head = nn.Linear(hid, 768)
            nn.init.normal_(self.head.weight, std=1 / math.sqrt(hid))
            nn.init.zeros_(self.head.bias)
            self.hid, self.nl, self.nh, self.eps = hid, nl, nh, eps

        def forward(self, ids, mask):
            p, B, L, hid = self.p, ids.shape[0], ids.shape[1], self.hid
            x = p[0][ids] + p[1][:L][None] + p[2]
            x = F.layer_norm(x, (hid,), p[3], p[4], self.eps)
            am = (1.0 - mask[:, None, None, :]) * -1e9
            dh = hid // self.nh
            for i in range(self.nl):
                wqkv, bqkv, wo, bo, g1, b1, wi, bi, wo2, bo2, g2, b2 = p[5 + 12 * i: 17 + 12 * i]
                qkv = (x @ wqkv + bqkv).reshape(B, L, 3, self.nh, dh).permute(2, 0, 3, 1, 4)
                s = (qkv[0] / math.sqrt(dh)) @ qkv[1].transpose(-1, -2) + am
                c = (s.softmax(-1) @ qkv[2]).transpose(1, 2).reshape(B, L, hid)
                x = F.layer_norm(c @ wo + bo + x, (hid,), g1, b1, self.eps)
                h = F.gelu(x @ wi + bi, approximate="tanh")
                x = F.layer_norm(h @ wo2 + bo2 + x, (hid,), g2, b2, self.eps)
            m = mask[:, :, None]
            e = (x * m).sum(1) / m.sum(1)
            return F.normalize(self.head(e), dim=-1)

        def export(self, path):
            ws = [q.detach().numpy().astype(np.float32) for q in self.p]
            ws += [self.head.weight.detach().numpy().T.copy(), self.head.bias.detach().numpy()]
            np.savez(path, *ws, meta=np.array([self.nl, self.nh, self.eps], dtype=np.float64))

    return Student()


def train(a):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(0)
    random.seed(0)
    torch.set_num_threads(a.threads)
    d = np.load(f"{OUT}/train.npz")
    lens = np.minimum(d["lens"], a.max_len)
    offs = np.concatenate([[0], np.cumsum(d["lens"])])
    flat, Y = d["flat"], torch.tensor(d["emb"].astype(np.float32))
    n = len(lens)
    rng = np.random.default_rng(0)
    hold = rng.choice(n, 2000, replace=False)
    train_idx = np.setdiff1d(np.arange(n), hold)

    def seq(i):
        s = flat[offs[i]:offs[i + 1]]
        return s if len(s) <= a.max_len else np.concatenate([s[:a.max_len - 1], s[-1:]])

    def batch(idx):
        L = int(lens[idx].max())
        ids = np.zeros((len(idx), L), dtype=np.int64)
        mask = np.zeros((len(idx), L), dtype=np.float32)
        for r, i in enumerate(idx):
            s = seq(i)
            ids[r, :len(s)] = s
            mask[r, :len(s)] = 1
        return torch.tensor(ids), torch.tensor(mask)

    def batches(idx):
        # length-bucketed, shuffled batches under a padded-token budget
        idx = idx[np.argsort(lens[idx] + rng.random(len(idx)) * 8)]
        out, cur = [], []
        for i in idx:
            if cur and (len(cur) + 1) * max(lens[cur[-1]], lens[i]) > a.tok_budget or len(cur) >= a.max_bs:
                out.append(np.array(cur))
                cur = []
            cur.append(i)
        if cur:
            out.append(np.array(cur))
        random.shuffle(out)
        return out

    model = build(a.init)
    body = [p for p in model.p]
    opt = torch.optim.AdamW([{"params": body, "lr": a.lr}, {"params": model.head.parameters(), "lr": a.lr * 10}],
                            weight_decay=0.01)
    plan = [batches(train_idx) for _ in range(a.epochs)]
    total = sum(len(p) for p in plan)
    warm = max(1, int(0.03 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * max(0.02, 0.5 * (1 + math.cos(math.pi * s / total))))

    def evaluate():
        model.eval()
        with torch.no_grad():
            cs = []
            for b in np.array_split(np.sort(hold), 20):
                ids, mask = batch(b)
                cs.append((model(ids, mask) * Y[b]).sum(1))
        model.train()
        return float(torch.cat(cs).mean())

    print(f"train {len(train_idx)}  steps {total}  hold cos {evaluate():.4f}", flush=True)
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        for b in plan[ep]:
            ids, mask = batch(b)
            e = model(ids, mask)
            y = Y[b]
            loss = (1 - (e * y).sum(1)).mean()
            if a.rel > 0:
                loss = loss + a.rel * F.mse_loss(e @ e.T, y @ y.T)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        c = evaluate()
        model.export(f"{OUT}/{a.out}.npz")
        print(f"== epoch {ep} hold cos {c:.4f}  exported {a.out}.npz  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--init", default="paraphrase-MiniLM-L3-v2")
    ap.add_argument("--out", default="l3")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rel", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--max-bs", type=int, default=128)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--extra", default=None)
    a = ap.parse_args()
    prep(a.extra) if a.cmd == "prep" else train(a)
