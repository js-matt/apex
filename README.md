# Apex (Bittensor Subnet 1) miner workspace

Subnet 1 is **Apex**, run by Macrocosmos. It works like Kaggle: you don't run a server that
answers live requests. Instead you **submit a solution file** to a competition, it is scored
in an isolated sandbox, and **the top submission on the leaderboard gets that competition's
emissions** (winner takes all).

Official repo: https://github.com/macrocosm-os/apex · Docs: https://docs.macrocosmos.ai/subnets/subnet-1-apex

## Layout

```
solutions/text_clustering/   solution.py + pinned sandbox requirements
tools/local_eval.py          local scorer (same ARI/NMI metric, size/import checks, timing)
scripts/setup_apex.sh        clones the official repo and installs the `apex` CLI
```

## 1. Develop and test locally (works on Windows)

```powershell
cd C:\Bit\subnet1
py -m venv .venv
.venv\Scripts\activate
pip install -r solutions\text_clustering\requirements.txt

python tools\local_eval.py solutions\text_clustering\solution.py
```

Only submit when your local score beats your last submission. Each submission costs TAO.

## 2. Install the Apex CLI (needs WSL 2 / Linux)

The `apex` CLI does not run on native Windows. From an admin PowerShell:

```powershell
wsl --install -d Ubuntu     # reboot when it asks you to
```

Then inside Ubuntu (Python 3.12+ and git are required):

```bash
cd /mnt/c/Bit/subnet1
bash scripts/setup_apex.sh
```

## 3. Wallet, registration, linking (one time)

```bash
cd apex
uv run btcli wallet new_coldkey --wallet.name my-apex-wallet
uv run btcli wallet new_hotkey  --wallet.name my-apex-wallet --wallet.hotkey miner1
# Write down both mnemonics offline. Anyone who has them controls your funds.

uv run btcli wallet balance --wallet.name my-apex-wallet        # you need TAO first
uv run btcli subnet register --wallet.name my-apex-wallet --wallet.hotkey miner1 --netuid 1

apex link          # pick the wallet and hotkey
apex competitions  # a 403 here means you aren't registered yet, or you need to wait ~5 min
```

## 4. Submit

```bash
apex competitions -c <ID>                                   # rules, fee, deadline
apex submit ../solutions/text_clustering/solution.py -c <ID>
apex result <SUBMISSION_ID>
apex list -c <ID> -t                                        # leaderboard
```

If payment goes through but the upload fails, **don't pay again**. Reuse the printed
`--payment-block-hash` / `--payment-extrinsic-index`.

## Text clustering competition: what matters

- Contract: `GET /health`, `POST /cluster {"texts": [...]}` returns `{"cluster_ids": [...]}` (`-1` = noise)
- CPU only, **no internet** (no model downloads), 1.5 GB RAM, 90 s, file **< 50,000 chars**
- Only the packages in `requirements.txt` are available
- Score = mean over the round's subsets (4 × 5,000 texts in live rounds) of `(max(0, ARI) + NMI) / 2`
- The ground truth is **sentence embeddings + UMAP + HDBSCAN** on X/Reddit posts from SN13.
  Your goal is to approximate that pipeline cheaply. 20 Newsgroups (the default in
  `local_eval.py`) is only a rough stand-in. For a better local benchmark, pull real posts
  with the Macrocosmos SDK (`pip install macrocosmos`), label them yourself with
  `sentence-transformers` + `umap-learn` + `hdbscan`, and save them as JSONL
  `{"text", "label"}` for `--data`.

Other competitions currently in the repo: `energy_arbitrage` and `aurelius_steering`. Read their
READMEs under `apex/shared/competition/src/competition/`.
