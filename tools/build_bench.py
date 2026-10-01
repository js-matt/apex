"""Build a local text_clustering benchmark that mimics the validator's ground truth.

Validator (from round metadata + apex repo comments): 4 subsets x 5000 SN13 X/Reddit
posts per round, labelled by mpnet embeddings -> UMAP -> HDBSCAN(min_cluster_size=25).
Observed ground truth per subset: 31-51 clusters, 12-30% noise, min cluster size 25.

    python tools/build_bench.py sample  --n 24          # data/bench/subset_XX.jsonl
    python tools/build_bench.py embed                   # data/bench/subset_XX.emb.npy (all-mpnet-base-v2)
    python tools/build_bench.py label                   # data/bench/subset_XX.jsonl gets "label"; prints GT stats

Needs the ground-truth env (torch, sentence-transformers, umap-learn, hdbscan, pyarrow).
"""
import argparse
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SN13 = ROOT / "data" / "sn13"
BENCH = ROOT / "data" / "bench"
SIZE = 5000


def load_pool():
    import pyarrow.parquet as pq
    frames = []
    for f in sorted(glob.glob(str(SN13 / "*.parquet"))):
        t = pq.read_table(f, columns=["text", "label"]).to_pandas()
        t["src"] = "reddit" if "reddit" in f else "x"
        t = t[t.text.str.strip().str.len() > 0]
        frames.append(t)
    import pandas as pd
    df = pd.concat(frames, ignore_index=True)
    df["label"] = df.label.fillna("NULL")
    return df


def sample_kind(kind: str, n_subsets: int, start: int, seed: int, src=None, files_glob=None):
    """Subsets shaped like the real rounds (inferred from which branch of the leader's
    code fired on each real subset):
      reddit: subsets 1-3 of a round - long posts, median length (>=40 chars) 230-330
      x:      subset 4 of a round - short single-line tweets, <8% newlines, <3% URLs
    """
    import pyarrow.parquet as pq
    import pandas as pd
    rng = np.random.default_rng(seed)
    files = sorted(glob.glob(str(Path(src or SN13) / (files_glob or f"*{'reddit' if kind == 'reddit' else 'x_'}*.parquet"))))
    cols = ["text", "label", "dataType"] if kind == "reddit" else ["text", "label"]
    df = pd.concat([pq.read_table(f, columns=cols).to_pandas() for f in files], ignore_index=True)
    df["label"] = df.label.fillna("NULL")
    if kind == "x":
        df["text"] = df.text.str.replace(r"\s*\n\s*", " ", regex=True)
        df = df[~df.text.str.contains("http|www\\.", regex=True)]
    df = df[df.text.str.strip().str.len() > 0].drop_duplicates("text")
    BENCH.mkdir(parents=True, exist_ok=True)
    for s in range(start, start + n_subsets):
        if kind == "reddit":
            post_frac = float(rng.uniform(0.6, 1.0))
            parts = [(df[df.dataType == "post"], post_frac), (df[(df.dataType == "comment") & (df.text.str.len() >= 40)], 1 - post_frac)]
        else:
            parts = [(df, 1.0)]
        k_focus = int(rng.integers(15, 50))
        focus_frac = float(rng.uniform(0.6, 0.85))
        rows = []
        for g, frac in parts:
            n_src = int(round(SIZE * frac))
            if n_src == 0:
                continue
            idx = g.groupby("label").indices
            big = [l for l, ix in idx.items() if len(ix) >= 150 and l != "NULL"]
            labs = rng.choice(big, size=min(len(big), max(3, int(round(k_focus * frac)))), replace=False)
            w = rng.pareto(1.2, size=len(labs)) + 0.3
            n_focus = int(n_src * focus_frac)
            for lab, cnt in zip(labs, rng.multinomial(n_focus, w / w.sum())):
                ix = idx[lab]
                rows += list(g.index[rng.choice(ix, size=min(cnt, len(ix)), replace=False)])
            rows += list(g.index[rng.choice(len(g), size=n_src - n_focus, replace=False)])
        rows = list(dict.fromkeys(rows))[:SIZE]
        while len(rows) < SIZE:
            rows = list(dict.fromkeys(rows + list(df.index[rng.choice(len(df), SIZE - len(rows))])))[:SIZE]
        rows = np.array(rows)
        rng.shuffle(rows)
        sub = df.loc[rows]
        with open(BENCH / f"subset_{s:02d}.jsonl", "w") as f:
            for t, l in zip(sub.text, sub.label):
                f.write(json.dumps({"text": t, "src_label": l, "src": kind}, ensure_ascii=False) + "\n")
        L = sub.text.str.len()
        print(f"subset {s:02d} [{kind}]: k_focus={k_focus} focus={focus_frac:.2f} median(len>=40)={L[L >= 40].median():.0f} "
              f"nl={sub.text.str.contains(chr(10)).mean():.2f}", flush=True)


def sample_fresh(seed: int, size: int = SIZE):
    """Subsets from fresh SN13 on-demand posts (tools/fetch_sn13.py -> data/sn13_fresh/*.jsonl).
    60 = short X subset (URLs stripped, newlines flattened: the real short subset has <3% URLs, <8% newlines)
    70 = long Reddit subset."""
    import re
    rng = np.random.default_rng(seed)
    fresh = ROOT / "data" / "sn13_fresh"
    for sid, prefix in ((60, "x_"), (70, "reddit_")):
        rows, seen = [], set()
        for p in sorted(fresh.glob(f"{prefix}*.jsonl")):
            for r in read(p):
                t = r["text"]
                if prefix == "x_":
                    t = re.sub(r"https?://\S+|www\.\S+", " ", t)
                    t = re.sub(r"\s*\n\s*", " ", t)
                t = t.strip()
                if len(t) < 2 or t in seen:
                    continue
                seen.add(t)
                rows.append({"text": t, "src_label": r["keyword"], "src": prefix.rstrip("_")})
        rng.shuffle(rows)
        rows = rows[:size]
        with open(BENCH / f"subset_{sid:02d}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        L = np.array([len(r["text"]) for r in rows])
        print(f"subset {sid} [{prefix.rstrip('_')}]: {len(rows)} texts, median(len>=40)={np.median(L[L >= 40]):.0f}, "
              f"nl={np.mean([chr(10) in r['text'] for r in rows]):.2f}, "
              f"url={np.mean(['http' in r['text'] for r in rows]):.2f}")


def sample(n_subsets: int, seed: int):
    df = load_pool()
    rng = np.random.default_rng(seed)
    groups = {src: g for src, g in df.groupby("src")}
    label_idx = {src: g.groupby("label").indices for src, g in groups.items()}
    big = {src: [l for l, ix in label_idx[src].items() if len(ix) >= 150 and l != "NULL"] for src in groups}
    BENCH.mkdir(parents=True, exist_ok=True)
    for s in range(n_subsets):
        p_reddit = [0.0, 0.25, 0.5, 0.75, 1.0][s % 5]
        k_focus = int(rng.integers(12, 45))           # topics that should form clusters
        focus_frac = float(rng.uniform(0.55, 0.85))   # rest is background -> noise / small clusters
        rows = []
        for src, frac in (("reddit", p_reddit), ("x", 1 - p_reddit)):
            n_src = int(round(SIZE * frac))
            if n_src == 0:
                continue
            g = groups[src]
            n_focus = int(n_src * focus_frac)
            labs = rng.choice(big[src], size=max(1, int(round(k_focus * max(frac, 0.2)))), replace=False)
            w = rng.pareto(1.2, size=len(labs)) + 0.3
            w = w / w.sum()
            for lab, cnt in zip(labs, rng.multinomial(n_focus, w)):
                ix = label_idx[src][lab]
                rows += list(g.index[rng.choice(ix, size=min(cnt, len(ix)), replace=False)])
            rows += list(g.index[rng.choice(len(g), size=n_src - n_focus, replace=False)])
        rows = np.array(rows)[:SIZE]
        if len(rows) < SIZE:  # top up from the whole pool
            extra = rng.choice(len(df), size=SIZE - len(rows), replace=False)
            rows = np.concatenate([rows, df.index[extra]])
        rng.shuffle(rows)
        sub = df.loc[rows]
        with open(BENCH / f"subset_{s:02d}.jsonl", "w") as f:
            for t, l, src in zip(sub.text, sub.label, sub.src):
                f.write(json.dumps({"text": t, "src_label": l, "src": src}, ensure_ascii=False) + "\n")
        print(f"subset {s:02d}: reddit={p_reddit:.2f} k_focus={k_focus} focus={focus_frac:.2f}")


def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def embed(model_name: str, only=None, threads=None):
    from sentence_transformers import SentenceTransformer
    import time
    import torch
    if threads:
        torch.set_num_threads(threads)
    model = SentenceTransformer(model_name, device="cpu")
    for p in sorted(BENCH.glob("subset_*.jsonl")):
        if only and int(p.stem.split("_")[1]) not in only:
            continue
        out = p.with_suffix(".emb.npy")
        if out.exists():
            continue
        texts = [r["text"] for r in read(p)]
        t0 = time.time()
        e = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        np.save(out, e.astype(np.float32))
        print(f"{p.name}: {len(texts)} texts in {time.time() - t0:.0f}s", flush=True)


def gt_labels(emb, n_neighbors=15, n_components=5, min_cluster_size=25, min_samples=None, method="eom", seed=42):
    import umap
    import hdbscan
    red = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, min_dist=0.0,
                    metric="cosine", random_state=seed).fit_transform(emb)
    return hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric="euclidean",
                           cluster_selection_method=method).fit_predict(red)


def label(args):
    for p in sorted(BENCH.glob("subset_*.jsonl")):
        emb_p = p.with_suffix(".emb.npy")
        if not emb_p.exists() or (args.only_set and int(p.stem.split("_")[1]) not in args.only_set):
            continue
        lab = gt_labels(np.load(emb_p), args.n_neighbors, args.n_components, args.min_cluster_size, args.min_samples)
        rows = read(p)
        for r, l in zip(rows, lab):
            r["label"] = int(l)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        ids, cnt = np.unique(lab[lab >= 0], return_counts=True)
        print(f"{p.name}: clusters={len(ids)} noise={(lab < 0).mean():.2f} min={cnt.min() if len(cnt) else 0} max={cnt.max() if len(cnt) else 0}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sample", "sample-kind", "sample-fresh", "embed", "label"])
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--kind", choices=["reddit", "x"])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--only", help="subset ids, e.g. 30-37")
    ap.add_argument("--threads", type=int)
    ap.add_argument("--src", help="parquet dir (default data/sn13)")
    ap.add_argument("--files", help="glob within --src")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--n-neighbors", type=int, default=15)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--min-cluster-size", type=int, default=25)
    ap.add_argument("--min-samples", type=int, default=10)  # eom + min_samples=10 best matches real GT stats
    a = ap.parse_args()
    only = None
    if a.only:
        lo, _, hi = a.only.partition("-")
        only = set(range(int(lo), int(hi or lo) + 1))
    a.only_set = only
    {"sample": lambda: sample(a.n, a.seed),
     "sample-kind": lambda: sample_kind(a.kind, a.n, a.start, a.seed, a.src, a.files),
     "sample-fresh": lambda: sample_fresh(a.seed),
     "embed": lambda: embed(a.model, only, a.threads),
     "label": lambda: label(a)}[a.cmd]()
