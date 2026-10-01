"""Generate mpnet (teacher) targets for extra SN13 Reddit posts, in torch on CPU.

    ~/.venvs/apex-train/bin/python gen_targets.py --n 150000 --out /home/abc/apex/data/indep/extra_reddit
Writes <out>_XX.jsonl / <out>_XX.emb.npy shards (eval-subset texts excluded).
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mpnet_np  # noqa: E402

DATA = "/home/abc/apex/data"
SNAP = glob.glob("/home/abc/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/*")[0]
EVAL = [f"subset_{i}" for i in list(range(30, 38)) + list(range(80, 84))]


class TorchMPNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        W = mpnet_np.pack_weights(mpnet_np.load_safetensors(SNAP + "/model.safetensors"))
        self.W = [torch.tensor(w) for w in W]
        self.bias_cache = {}

    @torch.no_grad()
    def forward(self, ids, mask):
        W, (B, L) = self.W, ids.shape
        pos = (torch.cumsum(mask, 1) * mask + 1).long()
        x = F.layer_norm(W[0][ids] + W[1][pos], (768,), W[2], W[3], 1e-5)
        if L not in self.bias_cache:
            self.bias_cache[L] = W[4][torch.tensor(mpnet_np._rel_bucket(L))].permute(2, 0, 1)
        ab = self.bias_cache[L][None] + (1.0 - mask[:, None, None, :]) * -1e9
        for i in range(12):
            wqkv, bqkv, wo, bo, g1, b1, wi, bi, wo2, bo2, g2, b2 = W[5 + 12 * i:17 + 12 * i]
            qkv = (x @ wqkv + bqkv).reshape(B, L, 3, 12, 64).permute(2, 0, 3, 1, 4)
            s = (qkv[0] / 8.0) @ qkv[1].transpose(-1, -2) + ab
            c = (s.softmax(-1) @ qkv[2]).transpose(1, 2).reshape(B, L, 768)
            x = F.layer_norm(c @ wo + bo + x, (768,), g1, b1, 1e-5)
            x = F.layer_norm(F.gelu(x @ wi + bi) @ wo2 + bo2 + x, (768,), g2, b2, 1e-5)
        m = mask[:, :, None]
        return F.normalize((x * m).sum(1) / m.sum(1), dim=-1)


def main(a):
    torch.set_num_threads(a.threads)
    eval_texts = set()
    for s in EVAL:
        eval_texts.update(json.loads(l)["text"] for l in open(f"{DATA}/bench/{s}.jsonl", encoding="utf-8"))
    have = set()
    for f in glob.glob(f"{DATA}/bench/subset_*.jsonl"):
        have.update(json.loads(l)["text"] for l in open(f, encoding="utf-8"))
    rng = np.random.default_rng(a.seed)
    files = sorted(glob.glob(f"{DATA}/sn13/*reddit*.parquet"))
    pool = []
    for f in files:
        t = pq.read_table(f, columns=["text"]).column("text").to_pylist()
        pick = rng.choice(len(t), min(len(t), a.n), replace=False)
        pool += [t[i] for i in pick]
    rng.shuffle(pool)
    texts, seen = [], set()
    for t in pool:
        if t and t.strip() and len(t) >= a.min_chars and t not in seen and t not in eval_texts and t not in have:
            seen.add(t)
            texts.append(t)
        if len(texts) >= a.n:
            break
    print(f"{len(texts)} texts", flush=True)
    tok = mpnet_np.WordPiece(
        json.load(open(SNAP + "/tokenizer.json", encoding="utf-8"))["model"]["vocab"])
    model = TorchMPNet()
    t0 = time.time()
    for shard, s in enumerate(range(0, len(texts), a.shard)):
        chunk = texts[s:s + a.shard]
        seqs = [tok.encode(t, 384) for t in chunk]
        order = np.argsort([len(q) for q in seqs])
        out = np.zeros((len(chunk), 768), dtype=np.float32)
        i = 0
        while i < len(order):
            j = i
            while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= max(a.tok_budget, len(seqs[order[j]])):
                j += 1
            idx = order[i:j]
            L = max(len(seqs[k]) for k in idx)
            ids = torch.ones((len(idx), L), dtype=torch.long)
            mask = torch.zeros((len(idx), L))
            for r, k in enumerate(idx):
                ids[r, :len(seqs[k])] = torch.tensor(seqs[k])
                mask[r, :len(seqs[k])] = 1
            out[idx] = model(ids, mask).numpy()
            i = j
        with open(f"{a.out}_{shard:02d}.jsonl", "w", encoding="utf-8") as f:
            for t in chunk:
                f.write(json.dumps({"text": t}) + "\n")
        np.save(f"{a.out}_{shard:02d}.emb.npy", out.astype(np.float16))
        done = s + len(chunk)
        print(f"shard {shard}: {done}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150000)
    ap.add_argument("--out", default=f"{DATA}/indep/extra_reddit")
    ap.add_argument("--shard", type=int, default=10000)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--tok-budget", type=int, default=8192)
    ap.add_argument("--min-chars", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    main(ap.parse_args())
