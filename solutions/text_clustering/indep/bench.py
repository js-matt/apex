"""Score an embedding->labels method on the local bench (independent of tools/).

    python bench.py exact            # saved mpnet embeddings (= numpy mpnet, verified) -> ceiling
    python bench.py gtseed           # GT recipe re-run with another seed via own UMAP (noise floor)

Weighted score = 0.75 * mean(Reddit 30-37) + 0.25 * mean(arXiv 80-83).
"""
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
BENCH = "/home/abc/apex/data/bench"
REDDIT = [f"subset_{i}" for i in range(30, 38)]
ARXIV = [f"subset_{i}" for i in range(80, 84)]


def load(sub):
    rows = [json.loads(l) for l in open(f"{BENCH}/{sub}.jsonl", encoding="utf-8")]
    return [r["text"] for r in rows], np.array([r["label"] for r in rows])


def score(gt, pred):
    return (max(0.0, adjusted_rand_score(gt, pred)) + normalized_mutual_info_score(gt, pred)) / 2


MODELS = "/home/abc/apex/data/models"
_cache = {}


def _bert(name, **kw):
    def f(texts):
        if name not in _cache:
            import mpnet_np
            _cache[name] = mpnet_np.load_bert(f"{MODELS}/{name}", **kw)
        return _cache[name].encode(texts)
    return f


def _student(name, max_len=128):
    def f(texts):
        if name not in _cache:
            import mpnet_np
            _cache[name] = mpnet_np.load_student(f"/home/abc/apex/data/indep/{name}.npz",
                                                 f"{MODELS}/paraphrase-MiniLM-L3-v2", max_len)
        return _cache[name].encode(texts)
    return f


EMBEDDERS = {
    "L3": _bert("paraphrase-MiniLM-L3-v2"),
    "L6": _bert("all-MiniLM-L6-v2"),
    "L12": _bert("all-MiniLM-L12-v2"),
}


LABELERS = {}


def run(args):
    method, sub, kw = args
    texts, gt = load(sub)
    t = time.time()
    if method in LABELERS:
        pred = np.asarray(LABELERS[method](texts))
        return sub, score(gt, pred), time.time() - t, 0.0, int(pred.max() + 1), float((pred == -1).mean())
    cache = f"/home/abc/apex/data/indep/emb_cache/{method.replace(':', '_')}_{sub}.npy"
    if method == "exact":
        E = np.load(f"{BENCH}/{sub}.emb.npy")
    elif method.startswith("s:") and os.path.exists(cache):
        E = np.load(cache)
    elif method.startswith("s:"):
        E = _student(method[2:])(texts)
    else:
        E = EMBEDDERS[method](texts)
    te = time.time() - t
    if method.startswith("s:") and not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, E)
    import cluster_np
    pred = cluster_np.cluster(E, **kw)
    tc = time.time() - t - te
    return sub, score(gt, pred), te, tc, int(pred.max() + 1), float((pred == -1).mean())


def main(method, kw=None, subs=None, procs=12, quiet=False):
    kw = kw or {}
    subs = subs or REDDIT + ARXIV
    with Pool(procs) as p:
        res = p.map(run, [(method, s, kw) for s in subs])
    r = {s: v for s, v, *_ in res}
    for s, v, te, tc, k, noise in ([] if quiet else res):
        print(f"  {s}: {v:.4f}  embed {te:5.1f}s  cluster {tc:5.1f}s  k={k} noise={noise:.0%}")
    red = np.mean([r[s] for s in subs if s in REDDIT] or [np.nan])
    arx = np.mean([r[s] for s in subs if s in ARXIV] or [np.nan])
    print(f"{method} {kw}: reddit {red:.4f}  arxiv {arx:.4f}  weighted {0.75 * red + 0.25 * arx:.4f}")
    return r


if __name__ == "__main__":
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    main(sys.argv[1], kw)
