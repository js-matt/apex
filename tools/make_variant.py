"""Write a variant of a (minified) solution by exact text substitutions.

    python tools/make_variant.py BASE.py OUT.py "old=>new" ["old2=>new2" ...]

Each substitution must match exactly once, so a typo fails loudly instead of silently
producing the base solution under a new name.
"""
import sys
from pathlib import Path


def make_variant(base: Path, out: Path, subs: list[str]) -> None:
    src = base.read_text(encoding="utf-8")
    for s in subs:
        old, new = s.split("=>", 1)
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"{old!r} matches {n} times in {base.name}")
        src = src.replace(old, new)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(src, encoding="utf-8")
    print(f"{out}: {len(src)} chars")


if __name__ == "__main__":
    make_variant(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:])
