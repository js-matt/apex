"""Statistics over all saved predictions: paired tests, bootstrap CI, and simulated live rounds.

    .venv/bin/python tools/robustness_stats.py --base lead_181053 --sol cand_L3
Units: (subset, label variant, input order). Degenerate reference labels (largest cluster > 50%) are skipped.
Simulated round = 3 reddit-like subsets + 1 arXiv subset, one random label variant and input order per subset,
scored like the validator (mean of the 4 subset scores).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_preds import PRED, degenerate, labels  # noqa: E402
from noise_lab import score  # noqa: E402

REDDIT = list(range(30, 38)) + list(range(40, 50))
ARXIV = list(range(80, 92))
VARIANTS = ["A", "B", "gt_seed1", "gt_seed2", "gt_ms5", "gt_ms15"]
ORDERS = ["", "_sh1", "_sh2"]


def collect(base, sol):
    """{(subset, variant, order): (base_score, sol_score)} for every available, non-degenerate unit."""
    out = {}
    for s in REDDIT + ARXIV:
        for v in VARIANTS:
            gt = labels(s, v)
            if gt is None or degenerate(gt):
                continue
            for o in ORDERS:
                fb, fs = PRED / f"pred_{base}_{s:02d}{o}.npy", PRED / f"pred_{sol}_{s:02d}{o}.npy"
                if fb.exists() and fs.exists():
                    out[(s, v, o)] = (score(gt, np.load(fb)), score(gt, np.load(fs)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lead_181053")
    ap.add_argument("--sol", default="cand_L3")
    ap.add_argument("--rounds", type=int, default=20000)
    a = ap.parse_args()
    U = collect(a.base, a.sol)
    print(f"{len(U)} paired units (subset x label variant x input order)")

    # Test 13a: paired tests per domain, over all units
    for name, dom in (("reddit-like", REDDIT), ("arXiv", ARXIV)):
        d = np.array([v[1] - v[0] for k, v in U.items() if k[0] in dom])
        pos, neg = int((d > 0).sum()), int((d < 0).sum())
        p_sign = binomtest(pos, pos + neg, .5, alternative="greater").pvalue if pos + neg else float("nan")
        p_wil = wilcoxon(d[d != 0], alternative="greater").pvalue if (d != 0).sum() > 5 else float("nan")
        print(f"[13] {name:11s} units={len(d):3d}  better={pos} worse={neg} tie={len(d) - pos - neg}  "
              f"mean diff={d.mean():+.4f}  sign-test p={p_sign:.2e}  wilcoxon p={p_wil:.2e}")

    # Test 13b: bootstrap CI of the weighted relative gain, resampling subsets (primary order, labels A and B)
    rng = np.random.default_rng(0)
    for v in ("A", "B"):
        R = [(U[(s, v, "")]) for s in REDDIT if (s, v, "") in U]
        X = [(U[(s, v, "")]) for s in ARXIV if (s, v, "") in U]
        R, X = np.array(R), np.array(X)
        gains = []
        for _ in range(10000):
            r, x = R[rng.integers(len(R), size=len(R))].mean(0), X[rng.integers(len(X), size=len(X))].mean(0)
            L, C = .75 * r[0] + .25 * x[0], .75 * r[1] + .25 * x[1]
            gains.append((C / L - 1) * 100)
        g = np.array(gains)
        L, C = .75 * R[:, 0].mean() + .25 * X[:, 0].mean(), .75 * R[:, 1].mean() + .25 * X[:, 1].mean()
        print(f"[13] bootstrap labels {v}: weighted gain {(C / L - 1) * 100:+.2f}%  95% CI [{np.percentile(g, 2.5):+.2f}%, {np.percentile(g, 97.5):+.2f}%]  "
              f"P(gain>0)={np.mean(g > 0):.3f}  ({len(R)} reddit-like, {len(X)} arXiv subsets)")

    # Test 14: simulated live rounds
    byR, byX = {}, {}
    for (s, v, o), val in U.items():
        (byR if s in REDDIT else byX).setdefault(s, []).append(val)
    rs, xs = sorted(byR), sorted(byX)
    gains = []
    for _ in range(a.rounds):
        picks = [byR[s][rng.integers(len(byR[s]))] for s in rng.choice(rs, 3, replace=False)]
        sx = xs[rng.integers(len(xs))]
        picks.append(byX[sx][rng.integers(len(byX[sx]))])
        P = np.array(picks)
        gains.append((P[:, 1].mean() / P[:, 0].mean() - 1) * 100)
    g = np.array(gains)
    print(f"[14] {a.rounds} simulated rounds (3 reddit-like + 1 arXiv, random labels/order): median gain {np.median(g):+.2f}%  "
          f"5th pct {np.percentile(g, 5):+.2f}%  95th pct {np.percentile(g, 95):+.2f}%")
    for t in (0, 1, 1.5, 2, 3):
        print(f"[14]   P(round gain > {t}%) = {np.mean(g > t):.3f}")


if __name__ == "__main__":
    main()
