"""Harvest competition submissions: index, eval metadata and revealed code (read-only API calls).

    cd apex && ../.venv-cli/python ../tools/harvest.py ../data/harvest --comp 10 --min-round 1
Run with the apex CLI's interpreter from the directory holding .apex.config.json (the wallet link).
Existing files are skipped, so re-running only fills gaps.
"""
import argparse
import asyncio
import json
from pathlib import Path

from cli.utils.client import Client
from cli.utils.config import Config
from common.models.api.code import CodeRequest
from common.models.api.submission import SubmissionRequest, SubmissionResponse


async def fetch_index(client, comp):
    subs, start = [], 0
    while True:
        req = SubmissionRequest(competition_id=comp, start_idx=start, count=100, filter_mode="all", sort_mode="time")
        resp = await client._make_request(method="GET", path="/miner/submission", params=req.model_dump())
        page = SubmissionResponse.model_validate(resp.json()).submissions
        subs += page
        if len(page) < 100:
            return subs
        start += 100


async def fetch_one(client, sem, s, out: Path):
    meta_p, code_p = out / "meta" / f"{s.id}.json", out / "code" / f"{s.id}.py"
    async with sem:
        for attempt in range(4):
            if meta_p.exists():
                break
            try:
                d = await client.get_submission_detail(s.id)
                meta_p.write_text(json.dumps(d.eval_metadata.model_dump(mode="json") if d and d.eval_metadata else None))
            except Exception:
                await asyncio.sleep(2 * (attempt + 1))
        for attempt in range(2):
            if code_p.exists():
                break
            chunks, start = [], 0
            try:
                while start is not None:
                    cr = await client.get_submission_code(CodeRequest(
                        competition_id=s.competition_id, round_number=s.round_number,
                        hotkey=s.hotkey, version=s.version, start_idx=start))
                    if cr.is_binary:
                        return
                    chunks.append(cr.code)
                    nxt = cr.pagination.next_start_idx
                    start = nxt if nxt and nxt > start else None
                code_p.write_text("".join(chunks))
            except Exception as e:
                if "locked" in str(e).lower() or "403" in str(e):
                    return
                await asyncio.sleep(2 * (attempt + 1))


async def main(out: Path, comp: int, min_round: int):
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    config = Config.load_config()
    async with Client(config.hotkey_file_path, timeout=120) as client:
        subs = await fetch_index(client, comp)
        (out / "index.json").write_text(json.dumps([s.model_dump(mode="json") for s in subs]))
        print(f"index: {len(subs)} submissions, rounds {min(s.round_number for s in subs)}-{max(s.round_number for s in subs)}", flush=True)
        todo = sorted((s for s in subs if s.round_number >= min_round and s.state == "scored"), key=lambda s: s.id)  # oldest first
        sem = asyncio.Semaphore(6)
        await asyncio.gather(*(fetch_one(client, sem, s, out) for s in todo))
        print(f"done: {len(list((out / 'meta').glob('*.json')))} meta, {len(list((out / 'code').glob('*.py')))} code files", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--comp", type=int, default=10)
    ap.add_argument("--min-round", type=int, default=1)
    a = ap.parse_args()
    asyncio.run(main(a.out.resolve(), a.comp, a.min_round))
