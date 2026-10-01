"""Compare saved predictions (bench_eval SAVE_PRED) of several solutions against a baseline, under any label variant.

    .venv/bin/python tools/compare_preds.py --base lead_181053 --sols cand_L,cand_L_spec \
        --groups "reddit=30-37,40-44;arxiv=80-87" --labels A,B [--tag _sh1]
Label variants: A = jsonl "label"; any other name X = data/bench/subset_NN.X.npy (e.g. B -> alt_ms3, or gt_seed1).
Reference labels that are degenerate (largest cluster > 50% of texts) are skipped and reported, as the live GT never is.
Weighted = 3 reddit : 1 arxiv, the live round mix.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arxiv_lab import parse  # noqa: E402
from noise_lab import score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BENCH, PRED = ROOT / "data" / "bench", ROOT / "data" / "sandbox_out" / "preds"
ALIAS = {"B": "alt_ms3"}


def labels(s, variant):
    if variant == "A":
        return np.array([json.loads(l)["label"] for l in open(BENCH / f"subset_{s:02d}.jsonl", encoding="utf-8")])
    p = BENCH / f"subset_{s:02d}.{ALIAS.get(variant, variant)}.npy"
    return np.load(p) if p.exists() else None


def degenerate(gt):
    ids, c = np.unique(gt[gt >= 0], return_counts=True)
    return len(c) == 0 or c.max() > .5 * len(gt)


def compare(base, sols, groups, variants, tag="", quiet=False):
    out = {}
    for v in variants:
        for g, ss in groups.items():
            rows = {sol: [] for sol in [base] + sols}
            used, skipped = [], []
            for s in ss:
                gt = labels(s, v)
                files = {sol: PRED / f"pred_{sol}_{s:02d}{tag}.npy" for sol in rows}
                if gt is None or degenerate(gt) or not all(f.exists() for f in files.values()):
                    skipped.append(s)
                    continue
                used.append(s)
                for sol, f in files.items():
                    rows[sol].append(score(gt, np.load(f)))
            if not used:
                continue
            L = np.array(rows[base])
            for sol in sols:
                C = np.array(rows[sol])
                out[(v, g, sol)] = (L.mean(), C.mean(), int((C > L).sum()), len(used), C - L)
                if not quiet:
                    print(f"[{v:8s}] {g:7s} {sol:14s} base {L.mean():.4f} sol {C.mean():.4f} {(C.mean() / L.mean() - 1) * 100:+6.2f}%  better {int((C > L).sum())}/{len(used)}"
                          + (f"  (skipped degenerate/missing {skipped})" if skipped else ""))
        for sol in sols:
            if (v, "reddit", sol) in out and (v, "arxiv", sol) in out:
                r, x = out[(v, "reddit", sol)], out[(v, "arxiv", sol)]
                Lw, Cw = .75 * r[0] + .25 * x[0], .75 * r[1] + .25 * x[1]
                out[(v, "weighted", sol)] = (Lw, Cw)
                if not quiet:
                    print(f"[{v:8s}] WEIGHTED {sol:14s} base {Lw:.4f} sol {Cw:.4f} {(Cw / Lw - 1) * 100:+6.2f}%")
    return out


def parse_groups(s):
    return {k: parse(v) for k, v in (part.split("=") for part in s.split(";"))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lead_181053")
    ap.add_argument("--sols", default="cand_L")
    ap.add_argument("--groups", default="reddit=30-37,40-44;arxiv=80-87")
    ap.add_argument("--labels", default="A,B")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    compare(a.base, a.sols.split(","), parse_groups(a.groups), a.labels.split(","), a.tag)
