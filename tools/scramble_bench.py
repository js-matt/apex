"""Multi-seed, multi-core benchmark for Humanoid Box Scramble ONNX policies.

The live score is one round seed x 12 instances, and the box field changes every round, so a
single-seed number says little. This runs every model over the same set of seeds in parallel and
reports the mean plus where runs end (fall position, timeout position).

    humanoid_scramble/.venv/bin/python tools/scramble_bench.py MODEL.onnx [MODEL2.onnx ...] \
        --seeds 1-8 -n 12 --json out.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
import sys
import time
from multiprocessing import Pool

ENV_ROOT = pathlib.Path(__file__).resolve().parents[1] / "humanoid_scramble"
sys.path.insert(0, str(ENV_ROOT))

import numpy as np  # noqa: E402

_SESS = {}


def _session(path):
    import onnxruntime as ort
    if path not in _SESS:
        o = ort.SessionOptions()
        o.intra_op_num_threads = o.inter_op_num_threads = 1
        _SESS[path] = ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])
    return _SESS[path]


def run_instance(job):
    path, seed, i, n, max_steps = job
    from env import ParkourSim, instance_score, instance_spec
    from env.sim import STATE_DIM
    sess = _session(path)
    names = [x.name for x in sess.get_inputs()]
    sim = ParkourSim(instance_spec(i, n, seed))
    obs = sim.reset()
    state = np.zeros((1, STATE_DIM), np.float32)
    reason, trace = None, []
    t0 = time.monotonic()
    while reason is None:
        action, state = sess.run(None, {names[0]: obs[None], names[1]: state})
        r = sim.step(np.asarray(action).ravel(), max_steps=max_steps)
        obs, reason = r.obs, r.terminal_reason
        if sim.steps % 100 == 0:
            trace.append(round(float(sim.data.qpos[0]), 2))
    q = sim.data.qpos
    return {"model": path, "seed": seed, "instance": i, "reason": reason, "steps": sim.steps,
            "max_x": round(sim.max_x, 2), "end_x": round(float(q[0]), 2), "end_y": round(float(q[1]), 2),
            "score": round(instance_score(reason, sim.progress, sim.steps, max_steps), 4),
            "trace": trace, "wall_s": round(time.monotonic() - t0, 1)}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def summarize(rows):
    by_seed = {}
    for r in rows:
        by_seed.setdefault(r["seed"], []).append(r["score"])
    seed_means = {s: statistics.fmean(v) for s, v in sorted(by_seed.items())}
    reasons = {}
    for r in rows:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    falls = [r["max_x"] for r in rows if r["reason"] == "fell"]
    return {"mean": round(statistics.fmean(r["score"] for r in rows), 4),
            "seed_means": {s: round(v, 4) for s, v in seed_means.items()},
            "worst_seed": round(min(seed_means.values()), 4),
            "reasons": reasons,
            "mean_max_x": round(statistics.fmean(r["max_x"] for r in rows), 2),
            "fall_x_median": round(statistics.median(falls), 1) if falls else None,
            "timeout_x_mean": round(statistics.fmean([r["max_x"] for r in rows if r["reason"] == "timeout"]), 1)
            if any(r["reason"] == "timeout" for r in rows) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--seeds", default="1-8")
    ap.add_argument("-n", type=int, default=12)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("-j", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--json")
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    models = [str(pathlib.Path(m).resolve()) for m in a.models]
    jobs = [(m, s, i, a.n, a.max_steps) for m in models for s in seeds for i in range(a.n)]
    t0 = time.monotonic()
    with Pool(a.j) as pool:
        rows = pool.map(run_instance, jobs, chunksize=1)
    out = {}
    for m in models:
        mr = [r for r in rows if r["model"] == m]
        out[m] = {"summary": summarize(mr), "rows": mr}
        s = out[m]["summary"]
        print(f"{pathlib.Path(m).name:45s} mean {s['mean']:.4f} worst-seed {s['worst_seed']:.4f} "
              f"max_x {s['mean_max_x']:5.2f} fall@{s['fall_x_median']} timeout@{s['timeout_x_mean']} {s['reasons']}")
        print("    per seed:", " ".join(f"{k}:{v:.3f}" for k, v in s["seed_means"].items()))
    print(f"{len(jobs)} instances in {time.monotonic() - t0:.0f}s")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
