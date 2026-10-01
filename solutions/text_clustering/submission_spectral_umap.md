# submission_spectral_umap.py

Built by `tools/build_candidate.py data/harvest/code/181053.py OUT.py` (defaults) on top of the current leader's revealed code
(5Ft9P2PY round-68 v2, #181053; the round-69 leader is that author's next version). 49,920 / 50,000 characters.

## Changes
1. **arXiv (short path, `BV`)**: after the leader's kNN smoothing, a spectral block is appended
   (UMAP-style fuzzy kNN graph, 15 neighbours, 8 eigenvectors, weight 0.5).
2. **Reddit (long path)**: a 10-d numpy UMAP layout (200 epochs, seeded) of the pre-spectral features is appended at weight 0.5,
   **only when the median text length is > 210 chars** and under 35 s have elapsed. Otherwise the output is identical to the leader's.
3. **Space (behaviour-neutral, verified identical labels):** dead constants and import, the unused >15,000-text branch, no-op `title=`
   plumbing; `RED` vectorised; web-service code flattened; weight blobs re-packed as raw LZMA2 (byte-identical weights).

## Robustness testing vs #181053 (all in the bubblewrap sandbox; A = calibrated labels, B = lower-noise live-like labels)
| # | Perspective | Result |
|---|---|---|
| 1 | Ablation | each change helps its own path only: arXiv block +2.7/+3.0%, Reddit block +1.6/+1.1% |
| 2 | Input order (original + 2 shuffles) | weighted +1.82/+1.59, +1.94/+1.82, +1.36/+1.37 % |
| 3 | GT UMAP seed 1 / 2 | weighted +1.56% / +1.66% |
| 4 | GT min_samples 5 / 15 | weighted +1.45% / +1.91% |
| 5 | 2,500-text subsets | Reddit +0.4/+0.5% (4/6), arXiv +2.4/+3.0% (4/4) |
| 6 | Non-live types | tweets +0.16%, short-text long path ±0 (gated), long X +1.0% |
| 7 | New arXiv mixes (category-skewed, uniform) | included in the 12-subset arXiv set: better on 12/12 under both A and B |
| 8 | Fresh unseen Reddit subsets 45–49 | +1.64% (4/4), +1.33% (4/5) |
| 9 | Single CPU core | ~50 s worst case (limit 90 s); when slow, falls back to exactly the leader's output |
| 10 | Determinism | identical labels on re-run |
| 11 | Peak memory | 1.25 GB (leader 1.31 GB; limit 1.5 GB) |
| 12 | Perturbed text (URLs, @, emoji, RT, caps) | Reddit +4.1/+4.5% (1/2); arXiv rerouted by leader's router → identical |
| 13 | Statistics | bootstrap 95% CI of weighted gain: A [+1.24%, +2.51%], B [+1.11%, +2.20%]; arXiv better on 206/213 units, Reddit 154 vs 53 |
| 14 | 20,000 simulated live rounds | median +1.60%; P(>0) 95.5%, P(>1%) 73%, P(>1.5%) 54%, P(>3%) 11% |

Edge cases (1–5 texts, empty, emoji/links, mixed scripts, duplicates) pass; HTTP `/health` + `/cluster` verified.

## Risks
- All evidence is local. Live data and reference labels may differ from the benchmark in ways the 14 tests do not cover.
- Taking the top spot needs ≥ +1% over the round's top score; the simulation puts that at about 73%.
- The Reddit gain is the less stable part (positive on average in every test, but per-subset wins drop to ~50% in one input order).

Submit (costs a TAO fee): `cd apex && apex submit ../solutions/text_clustering/submission_spectral_umap.py -c 10`
