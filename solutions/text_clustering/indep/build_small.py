"""Quantize a hashed table (4-bit per-column codebooks), LZMA + base85 it into runtime_small.py.

    python build_small.py h_w11c11_d16 ../submission_indep.py
"""
import base64
import lzma
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from small_lab import quantize  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def main(name, out):
    z = np.load(f"/home/abc/apex/data/indep/{name}.npz")
    W = (z["g"][:, None] * z["T"]).astype(np.float32)
    assert W.shape == (4096, 16), W.shape
    cbs, codes = quantize(W, 4)
    flat = codes.ravel()
    raw = cbs.astype(np.float16).tobytes() + ((flat[0::2] << 4) | flat[1::2]).astype(np.uint8).tobytes()
    comp = lzma.compress(raw, preset=9 | lzma.PRESET_EXTREME)
    blob = base64.b85encode(comp).decode()
    src = open(os.path.join(HERE, "runtime_small.py"), encoding="utf-8").read()
    src = src.replace("# __DATA__\n", f"BLOB = '{blob}'\n", 1)
    open(out, "w", encoding="utf-8").write(src)
    print(f"raw {len(raw)} B, lzma {len(comp)} B, blob {len(blob)} chars, file {len(src.strip())} chars")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
