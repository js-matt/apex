# Round 70 plan (Text Clustering, competition 10)

Round 70 opens **2026-10-01 14:15:47 UTC** (08:15 local in the CLI). The bar is the round-69 leader's code (#181059)
re-run on round-70 data × 1.01, unless someone in the opening burst clears it first.

## Prepared files (all on the leader's v3 code #181054, all edge-tested in the sandbox)
| File | Changes vs leader v3 | Local gain (A / B labels) |
|---|---|---|
| `r70_full.py` | arXiv spectral block + Reddit UMAP block | +1.59% / +1.67% |
| `r70_spec.py` | arXiv spectral block only | arXiv +2.67% / +2.98% (12/12); Reddit identical to leader |
| `r70_umap.py` | Reddit UMAP block only | Reddit +1.27% / +1.23% (11/16, 13/18); arXiv identical to leader |

## Why not submit blind
Round 69: our v2-based candidate scored 0.4638 vs leader v3 0.4658. Local estimates have shrunk live before
(the round-66 entry and #181160), so a variant is only worth the fee if round-69's live data says it clears the bar.

## Timeline at the round boundary
1. 14:15:47: round 69 ends; per-subset logs unlock.
2. 14:17-14:24: round-69 code unlocks (#181055, #181056, #181059, #181065, #181066).
3. ~14:20-14:25: run the go/no-go script (read-only):
   `cd /home/abc/apex/apex && /home/abc/.local/share/uv/tools/cli/bin/python ../tools/r70_decision.py`
   It isolates each change's live effect on round-69 data. The arXiv effect is exact; the Reddit effect is exact if the
   0.4604 trio ran the leader's v2. It recommends a file only if the predicted gain is ≥ +1.5%.
4. If it says SUBMIT: `cd /home/abc/apex/apex && apex submit ../solutions/text_clustering/<file> -c 10` (needs your coldkey password).
