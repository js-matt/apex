"""Pull fresh X / Reddit posts from Macrocosmos SN13 (Data Universe) for the local benchmark.

The API key is read from .env (MACROCOSMOS_API_KEY=...) and never printed.

    python tools/fetch_sn13.py --test                       # one tiny request to check the key
    python tools/fetch_sn13.py --per-keyword 400            # fetch the topic list below
Output: data/sn13_fresh/<source>_<keyword>.jsonl  ({"text", "keyword", "source", "datetime"} per line)
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "sn13_fresh"

# Gravity-style topics (the competition README: "AI, crypto, politics, sports, science, etc.")
KEYWORDS = {
    "X": ["AI", "ChatGPT", "bitcoin", "ethereum", "crypto", "bittensor", "Trump", "election", "NFL", "NBA",
          "football", "Formula1", "SpaceX", "NASA", "climate", "Tesla", "stocks", "gaming", "music", "movies",
          "health", "vaccine", "Ukraine", "Gaza", "iPhone"],
    # plain words: SN13 Reddit on-demand searches post text; "r/<name>" returned nothing
    "Reddit": ["AI", "bitcoin", "crypto", "election", "politics", "NBA", "NFL", "soccer", "Formula1", "science",
               "space", "technology", "gaming", "movies", "music", "stocks", "investing", "relationship", "fitness",
               "cars", "travel", "books", "programming", "health", "climate"],
}


DAYS = 7  # the validator crawls fresh posts; stay recent


def api_key() -> str:
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() == "MACROCOSMOS_API_KEY":
            return v.strip().strip("'\"")
    raise SystemExit("MACROCOSMOS_API_KEY not found in .env")


def fetch(client, source: str, keyword: str, limit: int):
    """One on-demand request. The SDK surface differs between versions, so try the known call shapes."""
    import datetime as dt
    start = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=DAYS)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return client.sn13.OnDemandData(source=source, keywords=[keyword], limit=limit, start_date=start)


def rows_from(resp):
    data = resp.get("data") if isinstance(resp, dict) else getattr(resp, "data", None)
    for post in data or []:
        p = post if isinstance(post, dict) else getattr(post, "__dict__", {})
        text = p.get("content") or p.get("text") or p.get("body") or p.get("title") or ""
        if text.strip():
            yield text, p.get("datetime") or p.get("created_at") or p.get("timestamp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--per-keyword", type=int, default=400)
    ap.add_argument("--reddit-per-keyword", type=int, default=None)
    a = ap.parse_args()
    from macrocosmos import Sn13Client
    client = Sn13Client(api_key=api_key(), max_retries=3)  # a DNS/network drop killed 22 requests on the first run
    if a.test:
        resp = fetch(client, "X", "bitcoin", 5)
        got = list(rows_from(resp))
        print(f"status={resp.get('status')!r}: got {len(got)} posts; meta keys {sorted((resp.get('meta') or {}).keys())}")
        if resp.get("data"):
            print("post fields:", sorted(resp["data"][0].keys()))
        for t, d in got[:3]:
            print(f"  [{d}] {t[:100]!r}")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    for source, kws in KEYWORDS.items():
        for kw in kws:
            dest = OUT / f"{source.lower()}_{kw.replace('/', '_')}.jsonl"
            if dest.exists():
                continue
            try:
                n = a.reddit_per_keyword if (source == "Reddit" and a.reddit_per_keyword) else a.per_keyword
                got = list(rows_from(fetch(client, source, kw, n)))
            except Exception as e:  # keep going; report the failure without echoing request details
                print(f"{source} {kw}: failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
                continue
            with open(dest, "w", encoding="utf-8") as f:
                for t, d in got:
                    f.write(json.dumps({"text": t, "keyword": kw, "source": source, "datetime": str(d)}, ensure_ascii=False) + "\n")
            print(f"{source} {kw}: {len(got)} posts", flush=True)
            time.sleep(1)


if __name__ == "__main__":
    main()
