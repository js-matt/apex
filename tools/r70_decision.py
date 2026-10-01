"""Go/no-go for a round-70 text-clustering submission, run right after round 69's code and logs unlock (~14:15-14:30 UTC).

    cd apex && /home/abc/.local/share/uv/tools/cli/bin/python ../tools/r70_decision.py

1. Fetches revealed round-69 code (leader #181059, the 0.4604 trio #181056/#181065/#181066, auto re-run #181055) and says
   which known base each one is (v14 #181036, v2 #181053, v3 #181054).
2. Fetches round-69 per-subset scores for ours (#181160), the leader, and the trio.
   - arXiv: our short path = leader's short path + spectral block, so ours - leader = the spectral block's live effect.
   - Reddit: if the trio ran v2 exactly, ours - trio = the UMAP block's live effect (same data, only change).
3. Predicts each prepared variant's round-69 score vs the leader and recommends one only if it clears +1.5% (margin above the 1% bar).
Read-only API calls; nothing is submitted.
"""
import asyncio
import hashlib
import re
from pathlib import Path

from cli.utils.client import Client
from cli.utils.config import Config
from common.models.api.code import CodeRequest

ROOT = Path(__file__).resolve().parent.parent
CODE = ROOT / "data" / "harvest" / "code"
OURS, LEADER, TRIO, AUTO = 181160, 181059, [181056, 181065, 181066], 181055
BASES = {"v14 #181036": 181036, "v2 #181053": 181053, "v3 #181054": 181054}
MARGIN = 1.5


def norm(src):
    """Statements, ignoring the docstring/Field cosmetics that differ between otherwise identical copies."""
    src = src.replace('"""Public Apex180663 adaptation; v14."""', "").replace("from pydantic import Field", "")
    src = src.replace("=Field(max_length=50000)", "")
    return [s.strip() for s in re.split(r"[;\n]", src) if s.strip()]


def identify(src):
    out = []
    for name, sid in BASES.items():
        p = CODE / f"{sid}.py"
        if p.exists():
            a, b = norm(src), norm(p.read_text(encoding="utf-8"))
            diff = len(set(a) ^ set(b))
            out.append((diff, name))
    diff, name = min(out)
    return f"identical to {name}" if diff == 0 else f"closest: {name} ({diff} differing statements)"


async def fetch_code(client, sid, idx):
    p = CODE / f"{sid}.py"
    if p.exists():
        return p.read_text(encoding="utf-8")
    s = idx[sid]
    chunks, start = [], 0
    while start is not None:
        cr = await client.get_submission_code(CodeRequest(competition_id=10, round_number=s.round_number, hotkey=s.hotkey, version=s.version, start_idx=start))
        chunks.append(cr.code)
        nxt = cr.pagination.next_start_idx if cr.pagination else None
        start = nxt if nxt and nxt > start else None
        await asyncio.sleep(6.5)
    p.write_text("".join(chunks), encoding="utf-8")
    return p.read_text(encoding="utf-8")


async def per_subset(client, sid):
    d = await client.get_submission_detail(sid)
    em = d.eval_metadata.model_dump(mode="json") if d and d.eval_metadata else None
    if not em or "subsets" not in (em.get("details") or {}):
        return None
    return {s["pool_label"].split("_", 3)[-1]: s["score"]["combined"] for s in em["details"]["subsets"]}


async def main():
    from common.models.api.submission import SubmissionRequest, SubmissionResponse
    cfg = Config.load_config()
    async with Client(cfg.hotkey_file_path, timeout=120) as client:
        req = SubmissionRequest(competition_id=10, start_idx=0, count=100, filter_mode="all", sort_mode="time")
        resp = await client._make_request(method="GET", path="/miner/submission", params=req.model_dump())
        recent = SubmissionResponse.model_validate(resp.json()).submissions
        idx = {s.id: s for s in recent}

        print("== 1. What did round 69's entries run?")
        trio_is_v2 = False
        for sid in [LEADER, AUTO] + TRIO:
            try:
                src = await fetch_code(client, sid, idx)
                ident = identify(src)
                print(f"  #{sid}: {ident}")
                if sid in TRIO and ident == "identical to v2 #181053":
                    trio_is_v2 = True
            except Exception as e:
                print(f"  #{sid}: code not available yet ({str(e)[:80]})")

        print("== 2. Round-69 per-subset scores")
        ours, lead = await per_subset(client, OURS), await per_subset(client, LEADER)
        trio = None
        for sid in TRIO:
            trio = await per_subset(client, sid)
            if trio:
                break
        for name, d in (("ours #181160", ours), ("leader #181059", lead), ("trio", trio)):
            print(f"  {name:15s}: " + (", ".join(f"{k}={v:.4f}" for k, v in sorted(d.items())) if d else "locked / unavailable"))
        if not ours or not lead:
            print("\nNot enough data yet - re-run in a few minutes.")
            return

        print("== 3. Live effect of each change on round-69 data")
        keys = sorted(ours)
        arx = [k for k in keys if "arxiv" in k]
        red = [k for k in keys if k not in arx]
        d_spec = sum(ours[k] - lead[k] for k in arx)
        print(f"  arXiv spectral block: {d_spec:+.4f} on the arXiv subset (exact: same short path otherwise)")
        if trio and trio_is_v2:
            d_umap = sum(ours[k] - trio[k] for k in red)
            print(f"  Reddit UMAP block:    {d_umap:+.4f} summed over {len(red)} Reddit subsets (ours vs v2 trio)")
        else:
            d_umap = None
            print("  Reddit UMAP block:    unknown (trio is not exactly v2 or its logs are locked)")
        base = sum(lead.values()) / len(lead)
        print("== 4. Predicted round-69 score of each prepared variant vs the leader")
        cands = {"r70_spec.py": d_spec}
        if d_umap is not None:
            cands["r70_umap.py"] = d_umap
            cands["r70_full.py"] = d_spec + d_umap
        best = None
        for f, d in cands.items():
            g = d / len(lead) / base * 100
            print(f"  {f:12s}: {g:+.2f}% vs leader")
            if g >= MARGIN and (best is None or g > best[1]):
                best = (f, g)
        print("== 5. Recommendation")
        if best:
            print(f"  SUBMIT solutions/text_clustering/{best[0]} (predicted {best[1]:+.2f}%; bar is +1%).")
            print(f"  cd {ROOT}/apex && apex submit ../solutions/text_clustering/{best[0]} -c 10")
        else:
            print(f"  DO NOT SUBMIT: no variant is predicted to clear +{MARGIN}% (bar +1% plus margin for round-to-round variation).")


if __name__ == "__main__":
    asyncio.run(main())
