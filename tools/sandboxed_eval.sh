#!/usr/bin/env bash
# Run tools/bench_eval.py on UNTRUSTED solution files (e.g. other miners' revealed code)
# inside bubblewrap: no network, no home directory (wallets unreachable), everything
# read-only except $OUT. Solutions must live under one of the read-only dirs below.
#
#   OUT=/path/to/results tools/sandboxed_eval.sh sol1.py [sol2.py ...] [bench_eval args]
#   TOOL=dump_features.py tools/sandboxed_eval.sh sol.py --subsets 30-37   (any script in tools/)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT:-$ROOT/data/sandbox_out}"
mkdir -p "$OUT"

exec bwrap \
  --unshare-all --die-with-parent --new-session \
  --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 --symlink usr/bin /bin \
  --ro-bind /etc/ld.so.cache /etc/ld.so.cache \
  --proc /proc --dev /dev --tmpfs /tmp \
  --ro-bind "$ROOT/.venv" "$ROOT/.venv" \
  --ro-bind "$ROOT/tools" "$ROOT/tools" \
  --ro-bind "$ROOT/data/bench" "$ROOT/data/bench" \
  --ro-bind "$ROOT/data/harvest/code" "$ROOT/data/harvest/code" \
  --ro-bind "$ROOT/top_solutions" "$ROOT/top_solutions" \
  --ro-bind "$ROOT/solutions" "$ROOT/solutions" \
  --ro-bind-try "$ROOT/data/variants" "$ROOT/data/variants" \
  --bind "$OUT" "$OUT" \
  --setenv HOME /tmp --setenv OUT "$OUT" --chdir "$OUT" \
  "$ROOT/.venv/bin/python" "$ROOT/tools/${TOOL:-bench_eval.py}" "$@"
