"""Harvest submissions for one competition: index, eval metadata, and revealed code/models.

Run inside the official repo's env (it uses the CLI's signed client):
    cd apex && uv run python ../tools/harvest_submissions.py 13 ../data/harvest_c13
"""
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from cli.utils.client import Client
from cli.utils.config import Config
from common.models.api.code import CodeRequest
from common.models.api.submission import SubmissionRequest, SubmissionResponse


async def list_all(client, comp_id):
    subs, start = [], 0
    while True:
        req = SubmissionRequest(competition_id=comp_id, start_idx=start, count=50, filter_mode="all", sort_mode="time")
        resp = await client._make_request(method="GET", path="/miner/submission", params=req.model_dump())
        page = SubmissionResponse.model_validate(resp.json()).submissions
        subs += page
        if len(page) < 50:
            return subs
        start += 50


async def main(comp_id: int, out: Path):
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "code").mkdir(parents=True, exist_ok=True)
    config = Config.load_config()
    now = datetime.now(timezone.utc)
    async with Client(config.hotkey_file_path, timeout=config.timeout) as client:
        subs = await list_all(client, comp_id)
        (out / "index.json").write_text(json.dumps([s.model_dump(mode="json") for s in subs], indent=1))
        print(f"{len(subs)} submissions")
        # Best first, so the rate limit (10 code fetches/min) spends itself on the models worth reading.
        for s in sorted(subs, key=lambda s: -(s.raw_score or -1)):
            meta_path = out / "meta" / f"{s.id}.json"
            if s.state == "scored" and not meta_path.exists():
                try:
                    detail = await client.get_submission_detail(s.id)
                    if detail and detail.eval_metadata:
                        meta_path.write_text(json.dumps(detail.eval_metadata.model_dump(mode="json")))
                except Exception as e:
                    print(f"meta {s.id}: {e}")
            reveal = s.reveal_at.replace(tzinfo=timezone.utc) if s.reveal_at and s.reveal_at.tzinfo is None else s.reveal_at
            if not reveal or reveal > now or s.state != "scored":
                continue
            if any((out / "code").glob(f"*_{s.id}_*")):
                continue
            await asyncio.sleep(6.5)
            try:
                code = await client.get_submission_code(
                    CodeRequest(competition_id=comp_id, round_number=s.round_number, hotkey=s.hotkey, version=s.version)
                )
            except Exception as e:
                print(f"code {s.id}: {e}")
                continue
            ext = "onnx" if code.is_binary else "py"
            name = f"r{s.round_number}_{s.id}_{s.hotkey[:8]}_v{s.version}_{s.raw_score or 0:.4f}.{ext}"
            if code.download_url:
                try:
                    async with httpx.AsyncClient(timeout=120) as http:
                        r = await http.get(code.download_url)
                        r.raise_for_status()
                except httpx.HTTPError as e:
                    print(f"download {s.id}: {e!r}")
                    continue
                (out / "code" / name).write_bytes(r.content)
            else:
                (out / "code" / name).write_text(code.code)
            print(f"saved {name}")


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), Path(sys.argv[2])))
