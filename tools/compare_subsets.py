"""Side-by-side per-subset scores for two submissions (read-only; nothing is submitted).

    cd /home/abc/apex/apex && /home/abc/.local/share/uv/tools/cli/bin/python ../tools/compare_subsets.py 181160 181059

Round-70 rule: r70_spec.py = the leader's code + the arXiv spectral block, and on arXiv your round-69 entry
(181160) = the leader's arXiv path + that same block. So (yours - leader) on the arXiv subset is the block's
live effect, and r70_spec's predicted overall gain is that difference / 4 / leader score.
"""
import asyncio
import sys

from cli.utils.client import Client
from cli.utils.config import Config

BAR, MARGIN = 0.01, 0.015  # 1% needed to take the top; 1.5% to submit with some safety


async def subsets(client, sid):
    d = await client.get_submission_detail(sid)
    if not d or not d.eval_metadata:
        return None
    m = d.eval_metadata.model_dump(mode="json")
    return {s["pool_label"].split("_subset_")[-1]: s["score"]["combined"] for s in m["details"]["subsets"]}, m["score"]


async def main(a, b):
    c = Config.load_config()
    async with Client(c.hotkey_file_path, timeout=120) as client:
        ra, rb = await subsets(client, a), await subsets(client, b)
    if not ra or not rb:
        print("Per-subset scores are still locked for", a if not ra else b, "- try again in a minute.")
        return
    (sa, ta), (sb, tb) = ra, rb
    print(f"{'subset':>8} {a:>10} {b:>10} {'diff':>9}")
    for k in sa:
        print(f"{k:>8} {sa[k]:10.4f} {sb.get(k, float('nan')):10.4f} {sa[k] - sb.get(k, float('nan')):+9.4f}")
    print(f"{'overall':>8} {ta:10.4f} {tb:10.4f} {ta - tb:+9.4f} ({(ta / tb - 1) * 100:+.2f}%)")
    if "arxiv" in sa and "arxiv" in sb:
        gain = (sa["arxiv"] - sb["arxiv"]) / 4 / tb
        verdict = "SUBMIT r70_spec.py" if gain >= MARGIN else ("borderline - skip" if gain >= BAR else "SKIP this round")
        print(f"\nr70_spec predicted gain over leader: {gain * 100:+.2f}%  (needs >= {MARGIN * 100:.1f}%)  ->  {verdict}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
