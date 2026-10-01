"""Pack a trained student (.npz) + vocab into runtime.py -> a single self-contained solution file.

    python build_solution.py /home/abc/apex/data/indep/l3.npz solution_student.py
"""
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VOCAB = "/home/abc/apex/data/models/paraphrase-MiniLM-L3-v2/tokenizer.json"


def main(npz, out):
    z = np.load(npz)
    ws = [z[f"arr_{i}"] for i in range(len(z.files) - 1)]
    nl, nh, eps = z["meta"]
    vocab = json.load(open(VOCAB, encoding="utf-8"))["model"]["vocab"]
    tokens = [None] * len(vocab)
    for t, i in vocab.items():
        tokens[i] = t
    assert all(t is not None and "\n" not in t for t in tokens)
    blob = base64.b64encode(np.concatenate([w.astype(np.float16).ravel() for w in ws]).tobytes()).decode()
    data = (f"_META = ({int(nl)}, {int(nh)}, {float(eps)!r})\n"
            f"_SHAPES = {[tuple(int(d) for d in w.shape) for w in ws]!r}\n"
            f"_VOCAB = {chr(10).join(tokens)!r}\n"
            f"_BLOB = {blob!r}\n")
    src = open(os.path.join(HERE, "runtime.py"), encoding="utf-8").read()
    assert "# __DATA__\n" in src
    src = src.replace("# __DATA__\n", data, 1)
    open(out, "w", encoding="utf-8").write(src)
    print(f"wrote {out}: {len(src):,} chars")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
