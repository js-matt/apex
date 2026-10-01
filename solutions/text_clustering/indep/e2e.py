"""End-to-end check: run a solution file as a server pinned to one core, POST bench subsets, score + time.

    python e2e.py solution_student.py [subset ...]
"""
import json
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bench import ARXIV, REDDIT, load, score  # noqa: E402

PY = "/home/abc/apex/.venv/bin/python"


def main(sol, subs, port=8765, core="5"):
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    proc = subprocess.Popen(["taskset", "-c", core, PY, sol, "--port", str(port), "--host", "127.0.0.1"],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        while True:
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1)
                break
            except Exception:
                if proc.poll() is not None:
                    raise RuntimeError(proc.stderr.read().decode()[-2000:])
                time.sleep(0.5)
        print(f"startup {time.time() - t0:.1f}s")
        res = {}
        for s in subs:
            texts, gt = load(s)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/cluster", json.dumps({"texts": texts}).encode(),
                                         {"Content-Type": "application/json"})
            t = time.time()
            pred = json.loads(urllib.request.urlopen(req, timeout=600).read())["cluster_ids"]
            dt = time.time() - t
            res[s] = score(gt, pred)
            print(f"  {s}: {res[s]:.4f}  {dt:5.1f}s  k={max(pred) + 1} noise={pred.count(-1) / len(pred):.0%}",
                  flush=True)
        return res
    finally:
        proc.terminate()


if __name__ == "__main__":
    subs = sys.argv[2:] or REDDIT + ARXIV
    main(sys.argv[1], subs)
