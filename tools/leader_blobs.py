"""Decode the projection matrices embedded in leader-lineage solutions, as plain data.

The file is parsed as text; none of its code runs. Two blobs exist in the v14 lineage:
  Ak: 2-bit per-column codebook W (12288 x 48), packed 15 bits per CJK char (decoder B2)
  BD: sparse-row 4-bit W (12288 x D) with per-column float scales (decoders BN + BO)
Both map hashed word 1-2gram (8192) + char_wb 3-5gram (4096) log1p counts to a dense embedding.
    python tools/leader_blobs.py data/harvest/code/181036.py   # saves data/leader_W_{Ak,BD}.npy
"""
import lzma
import re
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
N_ROWS = 8192 + 4096


def literal(src, name):
    m = re.search(rf"^{name}='([^']*)'", src, re.M)
    return m.group(1) if m else None


def decode_ak(s, dim=48):
    acc = nbits = 0
    buf = bytearray()
    for ch in s:
        acc = acc << 15 | (ord(ch) - 19968)
        nbits += 15
        while nbits >= 8:
            nbits -= 8
            buf.append(acc >> nbits & 255)
    raw = lzma.decompress(bytes(buf))
    n = N_ROWS * dim
    nb = (n + 3) // 4
    packed = np.frombuffer(raw[:nb], np.uint8)
    codes = np.zeros(nb * 4, np.uint8)
    for k in range(4):
        codes[k::4] = packed >> 2 * k & 3
    codes = codes[:n].reshape(N_ROWS, dim)
    books = np.frombuffer(raw[nb:nb + dim * 16], np.float32).reshape(dim, 4)
    return books[np.arange(dim)[None, :], codes].astype(np.float32)


def decode_bd(s):
    acc = 0
    for ch in s[2:]:
        acc = acc << 15 | (ord(ch) - 19968)
    raw = lzma.decompress(acc.to_bytes((ord(s[0]) - 19968) << 15 | (ord(s[1]) - 19968), "big"))
    rows, dim, _ = struct.unpack("<IIB", raw[:9])
    idx = np.cumsum(np.frombuffer(raw[9:9 + 2 * rows], np.uint16).astype(np.int64)) - 1
    g = rows * dim
    nb = (g + 1) // 2
    h = np.frombuffer(raw[9 + 2 * rows:9 + 2 * rows + nb], np.uint8)
    q = np.zeros(nb * 2, np.uint8)
    q[0::2], q[1::2] = h & 15, h >> 4
    scale = np.frombuffer(raw[9 + 2 * rows + nb:9 + 2 * rows + nb + dim * 4], np.float32)
    W = np.zeros((N_ROWS, dim), np.float32)
    W[idx] = (q[:g].astype(np.float32) - 8.0).reshape(rows, dim) * scale[None, :]
    return W


if __name__ == "__main__":
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    for name, dec in (("Ak", decode_ak), ("BD", decode_bd)):
        s = literal(src, name)
        if s is None:
            print(f"{name}: not found")
            continue
        W = dec(s)
        np.save(ROOT / "data" / f"leader_W_{name}.npy", W)
        print(f"{name}: {len(s)} chars -> W {W.shape}, nonzero rows {(np.abs(W).sum(1) > 0).sum()}")
