"""Score a text_clustering solution locally before paying a submission fee.

Starts the solution as a server (same as the sandbox does), POSTs texts to
/cluster, and scores the result with the competition's metric:
    combined = (max(0, ARI) + NMI) / 2, averaged over subsets.

Data (JSONL, one {"text": ..., "label": ...} per line):
    python tools/local_eval.py solutions/text_clustering/solution.py --data my_data.jsonl
Without --data it uses sklearn's 20 Newsgroups (downloaded once). That is only a
proxy: the real ground truth comes from embeddings + UMAP + HDBSCAN on X/Reddit
posts, so build a closer dataset (see README) once you're iterating seriously.
"""

import argparse
import ast
import json
import random
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

MAX_CHARS = 50_000
TIME_LIMIT_S = 90
ALLOWED_IMPORTS = {
    "numpy", "sklearn", "fastapi", "uvicorn", "pydantic",
    # stdlib modules are always fine; this set only covers third-party ones
}


def static_checks(path: Path) -> list[str]:
    src = path.read_text(encoding="utf-8")
    problems = []
    if len(src) >= MAX_CHARS:
        problems.append(f"file is {len(src):,} chars; limit is {MAX_CHARS:,}")
    stdlib = sys.stdlib_module_names
    for node in ast.walk(ast.parse(src)):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        for name in names:
            top = name.split(".")[0]
            if top not in stdlib and top not in ALLOWED_IMPORTS:
                problems.append(f"imports '{top}', which is not in the sandbox requirements")
    return problems


def load_jsonl(path: Path) -> tuple[list[str], list]:
    texts, labels = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                texts.append(row["text"])
                labels.append(row["label"])
    return texts, labels


def load_20newsgroups() -> tuple[list[str], list]:
    from sklearn.datasets import fetch_20newsgroups

    ds = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    # Trim to tweet-ish length so it resembles social media posts.
    return [t[:280] for t in ds.data], list(ds.target)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def post_json(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def wait_healthy(base: str, proc: subprocess.Popen, timeout: float = 60) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("solution server exited during startup")
        try:
            with urllib.request.urlopen(f"{base}/health", timeout=2) as r:
                if json.loads(r.read()).get("status") == "healthy":
                    return
        except OSError:
            time.sleep(0.5)
    raise TimeoutError("solution never reported healthy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("solution", type=Path)
    ap.add_argument("--data", type=Path, help="JSONL with 'text' and 'label' fields")
    ap.add_argument("--subsets", type=int, default=4)
    ap.add_argument("--size", type=int, default=5000, help="texts per subset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    problems = static_checks(args.solution)
    for p in problems:
        print(f"[check] {p}")
    if not problems:
        print("[check] size and imports OK")

    texts, labels = load_jsonl(args.data) if args.data else load_20newsgroups()
    rng = random.Random(args.seed)
    idx = list(range(len(texts)))

    port = free_port()
    base = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen(
        [sys.executable, str(args.solution), "--host", "127.0.0.1", "--port", str(port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_healthy(base, proc)
        scores = []
        for i in range(args.subsets):
            sample = rng.sample(idx, min(args.size, len(idx)))
            sub_texts = [texts[j] for j in sample]
            sub_labels = [labels[j] for j in sample]

            t0 = time.perf_counter()
            pred = post_json(f"{base}/cluster", {"texts": sub_texts}, TIME_LIMIT_S)["cluster_ids"]
            elapsed = time.perf_counter() - t0

            if len(pred) != len(sub_texts):
                raise ValueError(f"got {len(pred)} ids for {len(sub_texts)} texts")
            ari = adjusted_rand_score(sub_labels, pred)
            nmi = normalized_mutual_info_score(sub_labels, pred)
            combined = (max(0.0, ari) + nmi) / 2
            scores.append(combined)
            flag = "  OVER TIME LIMIT" if elapsed > TIME_LIMIT_S else ""
            print(
                f"subset {i}: n={len(sub_texts)} clusters={len(set(pred))} "
                f"ARI={ari:.4f} NMI={nmi:.4f} combined={combined:.4f} time={elapsed:.1f}s{flag}"
            )
        print(f"\nround score: {sum(scores) / len(scores):.4f}")
    finally:
        proc.terminate()
        proc.wait()


if __name__ == "__main__":
    main()
