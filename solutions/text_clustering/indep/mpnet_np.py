"""all-mpnet-base-v2 in pure numpy (tokenizer + encoder + mean pooling + L2 norm).

Written from the HF config/tokenizer.json only; matches sentence-transformers output.
"""
import json
import math
import struct
import unicodedata

import numpy as np
from scipy.special import erf

MAX_LEN = 384
N_LAYERS = 12
N_HEADS = 12
HID = 768


# ---------------------------------------------------------------- tokenizer
def _is_cjk(cp):
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
            or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


def _is_punc(ch):
    cp = ord(ch)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(ch).startswith("P")


class WordPiece:
    def __init__(self, vocab):
        self.vocab = vocab
        self.unk = vocab["[UNK]"]
        self.cache = {}

    def normalize(self, text):
        out = []
        for ch in text:
            if ch in "\t\n\r":
                out.append(" ")
                continue
            if ch == "\0" or ch == "�" or unicodedata.category(ch)[0] == "C":
                continue
            if ch.isspace():
                out.append(" ")
            elif _is_cjk(ord(ch)):
                out.append(" " + ch + " ")
            else:
                out.append(ch)
        text = unicodedata.normalize("NFD", "".join(out))
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        return text.lower()

    def pre_tokenize(self, text):
        words = []
        for w in text.split():
            cur = []
            for ch in w:
                if _is_punc(ch):
                    if cur:
                        words.append("".join(cur))
                        cur = []
                    words.append(ch)
                else:
                    cur.append(ch)
            if cur:
                words.append("".join(cur))
        return words

    def word_ids(self, w):
        r = self.cache.get(w)
        if r is not None:
            return r
        if len(w) > 100:
            r = [self.unk]
        else:
            r, start, n = [], 0, len(w)
            while start < n:
                end, tok = n, None
                while start < end:
                    s = w[start:end] if start == 0 else "##" + w[start:end]
                    tok = self.vocab.get(s)
                    if tok is not None:
                        break
                    end -= 1
                if tok is None:
                    r = [self.unk]
                    break
                r.append(tok)
                start = end
        self.cache[w] = r
        return r

    def encode(self, text, max_len=MAX_LEN, cls=0, sep=2):
        ids = []
        for w in self.pre_tokenize(self.normalize(text.strip())):
            ids.extend(self.word_ids(w))
            if len(ids) >= max_len - 2:
                break
        return [cls] + ids[: max_len - 2] + [sep]


# ---------------------------------------------------------------- weights
def load_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(n))
        buf = f.read()
    out = {}
    for k, v in head.items():
        if k == "__metadata__" or v["dtype"] != "F32":
            continue
        a, b = v["data_offsets"]
        out[k] = np.frombuffer(buf[a:b], dtype=np.float32).reshape(v["shape"])
    return out


def pack_weights(sd):
    """Rearrange HF state dict into the flat list the encoder uses (in this order)."""
    W = [sd["embeddings.word_embeddings.weight"], sd["embeddings.position_embeddings.weight"],
         sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"],
         sd["encoder.relative_attention_bias.weight"]]
    for i in range(N_LAYERS):
        p = f"encoder.layer.{i}."
        a = p + "attention.attn."
        W += [np.concatenate([sd[a + "q.weight"], sd[a + "k.weight"], sd[a + "v.weight"]]).T.copy(),
              np.concatenate([sd[a + "q.bias"], sd[a + "k.bias"], sd[a + "v.bias"]]),
              sd[a + "o.weight"].T.copy(), sd[a + "o.bias"],
              sd[p + "attention.LayerNorm.weight"], sd[p + "attention.LayerNorm.bias"],
              sd[p + "intermediate.dense.weight"].T.copy(), sd[p + "intermediate.dense.bias"],
              sd[p + "output.dense.weight"].T.copy(), sd[p + "output.dense.bias"],
              sd[p + "output.LayerNorm.weight"], sd[p + "output.LayerNorm.bias"]]
    return [np.ascontiguousarray(w, dtype=np.float32) for w in W]


# ---------------------------------------------------------------- encoder
def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    x = x - m
    v = (x * x).mean(-1, keepdims=True)
    return x / np.sqrt(v + eps) * g + b


def _rel_bucket(L, num_buckets=32, max_distance=128):
    ctx = np.arange(L)[:, None]
    mem = np.arange(L)[None, :]
    n = -(mem - ctx)
    nb = num_buckets // 2
    ret = (n < 0).astype(np.int64) * nb
    n = np.abs(n)
    max_exact = nb // 2
    with np.errstate(divide="ignore"):
        large = max_exact + (np.log(np.maximum(n, 1) / max_exact) / math.log(max_distance / max_exact)
                             * (nb - max_exact)).astype(np.int64)
    large = np.minimum(large, nb - 1)
    return ret + np.where(n < max_exact, n, large)


class MPNet:
    def __init__(self, W, vocab):
        self.W = W
        self.tok = WordPiece(vocab)
        self._bias = {}

    def bias(self, L):
        b = self._bias.get(L)
        if b is None:
            b = np.ascontiguousarray(self.W[4][_rel_bucket(L)].transpose(2, 0, 1))  # H,L,L
            self._bias[L] = b
        return b

    def forward(self, ids, mask):
        """ids, mask: (B, L) int arrays (right padded). Returns (B, 768) normalized mean-pooled."""
        W = self.W
        B, L = ids.shape
        pos = (np.cumsum(mask, 1) * mask + 1).astype(np.int64)
        x = W[0][ids] + W[1][pos]
        x = _ln(x, W[2], W[3]).reshape(B * L, HID)
        attn_bias = self.bias(L)[None] + ((1.0 - mask[:, None, None, :]) * -1e9).astype(np.float32)
        dh = HID // N_HEADS
        sc = np.float32(1.0 / math.sqrt(dh))
        for i in range(N_LAYERS):
            (wqkv, bqkv, wo, bo, g1, b1, wi, bi, wo2, bo2, g2, b2) = W[5 + 12 * i: 17 + 12 * i]
            qkv = x @ wqkv + bqkv
            qkv = qkv.reshape(B, L, 3, N_HEADS, dh).transpose(2, 0, 3, 1, 4)  # 3,B,H,L,dh
            q, k, v = qkv[0] * sc, qkv[1], qkv[2]
            s = q @ k.transpose(0, 1, 3, 2) + attn_bias
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            c = (s @ v).transpose(0, 2, 1, 3).reshape(B * L, HID)
            x = _ln(c @ wo + bo + x, g1, b1)
            h = x @ wi + bi
            h = 0.5 * h * (1.0 + erf(h * np.float32(1 / math.sqrt(2))))
            x = _ln(h.astype(np.float32) @ wo2 + bo2 + x, g2, b2)
        x = x.reshape(B, L, HID)
        m = mask[:, :, None].astype(np.float32)
        e = (x * m).sum(1) / m.sum(1)
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def encode(self, texts, max_len=MAX_LEN, tok_budget=8192, progress=False):
        seqs = [self.tok.encode(t, max_len) for t in texts]
        order = np.argsort([len(s) for s in seqs])
        out = np.zeros((len(texts), HID), dtype=np.float32)
        i = 0
        while i < len(order):
            L = len(seqs[order[i]])
            j = i
            # grow the batch while padded size stays within budget
            while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= max(tok_budget, len(seqs[order[j]])):
                j += 1
            idx = order[i:j]
            L = max(len(seqs[k]) for k in idx)
            ids = np.ones((len(idx), L), dtype=np.int64)
            mask = np.zeros((len(idx), L), dtype=np.float32)
            for r, k in enumerate(idx):
                s = seqs[k]
                ids[r, : len(s)] = s
                mask[r, : len(s)] = 1
            out[idx] = self.forward(ids, mask)
            i = j
        return out


def load_local(snapshot_dir):
    sd = load_safetensors(snapshot_dir + "/model.safetensors")
    vocab = json.load(open(snapshot_dir + "/tokenizer.json", encoding="utf-8"))["model"]["vocab"]
    return MPNet(pack_weights(sd), vocab)


# ---------------------------------------------------------------- BERT (MiniLM family)
def _gelu(h, fast):
    if fast:
        return 0.5 * h * (1.0 + np.tanh(np.float32(0.7978845608) * (h + np.float32(0.044715) * h * h * h)))
    return 0.5 * h * (1.0 + erf(h * np.float32(1 / math.sqrt(2))))


def pack_bert(sd, n_layers):
    W = [sd["embeddings.word_embeddings.weight"], sd["embeddings.position_embeddings.weight"],
         sd["embeddings.token_type_embeddings.weight"][0],
         sd["embeddings.LayerNorm.weight"], sd["embeddings.LayerNorm.bias"]]
    for i in range(n_layers):
        p = f"encoder.layer.{i}."
        a = p + "attention.self."
        W += [np.concatenate([sd[a + "query.weight"], sd[a + "key.weight"], sd[a + "value.weight"]]).T.copy(),
              np.concatenate([sd[a + "query.bias"], sd[a + "key.bias"], sd[a + "value.bias"]]),
              sd[p + "attention.output.dense.weight"].T.copy(), sd[p + "attention.output.dense.bias"],
              sd[p + "attention.output.LayerNorm.weight"], sd[p + "attention.output.LayerNorm.bias"],
              sd[p + "intermediate.dense.weight"].T.copy(), sd[p + "intermediate.dense.bias"],
              sd[p + "output.dense.weight"].T.copy(), sd[p + "output.dense.bias"],
              sd[p + "output.LayerNorm.weight"], sd[p + "output.LayerNorm.bias"]]
    return [np.ascontiguousarray(w, dtype=np.float32) for w in W]


class Bert:
    def __init__(self, W, vocab, n_layers, n_heads, max_len, eps=1e-12, fast_gelu=False, head=None):
        self.W, self.nl, self.nh, self.max_len, self.eps, self.fast = W, n_layers, n_heads, max_len, eps, fast_gelu
        self.head = head
        self.hid = W[0].shape[1]
        self.tok = WordPiece(vocab)

    def forward(self, ids, mask):
        W, B, L, hid = self.W, ids.shape[0], ids.shape[1], self.hid
        x = W[0][ids] + W[1][:L][None] + W[2]
        x = _ln(x, W[3], W[4], self.eps).reshape(B * L, hid)
        am = ((1.0 - mask[:, None, None, :]) * -1e9).astype(np.float32)
        dh = hid // self.nh
        sc = np.float32(1.0 / math.sqrt(dh))
        for i in range(self.nl):
            (wqkv, bqkv, wo, bo, g1, b1, wi, bi, wo2, bo2, g2, b2) = W[5 + 12 * i: 17 + 12 * i]
            qkv = (x @ wqkv + bqkv).reshape(B, L, 3, self.nh, dh).transpose(2, 0, 3, 1, 4)
            s = (qkv[0] * sc) @ qkv[1].transpose(0, 1, 3, 2) + am
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            c = (s @ qkv[2]).transpose(0, 2, 1, 3).reshape(B * L, hid)
            x = _ln(c @ wo + bo + x, g1, b1, self.eps)
            h = _gelu(x @ wi + bi, self.fast)
            x = _ln(h @ wo2 + bo2 + x, g2, b2, self.eps)
        x = x.reshape(B, L, hid)
        m = mask[:, :, None]
        e = (x * m).sum(1) / m.sum(1)
        if self.head is not None:
            e = e @ self.head[0] + self.head[1]
        return e / np.linalg.norm(e, axis=1, keepdims=True)

    def encode(self, texts, max_len=None, tok_budget=8192):
        max_len = max_len or self.max_len
        seqs = [self.tok.encode(t, max_len, 101, 102) for t in texts]
        return _batched(self, seqs, tok_budget, pad=0)


def _batched(model, seqs, tok_budget, pad):
    order = np.argsort([len(s) for s in seqs], kind="stable")
    out = None
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and len(seqs[order[j]]) * (j - i + 1) <= max(tok_budget, len(seqs[order[j]])):
            j += 1
        idx = order[i:j]
        L = max(len(seqs[k]) for k in idx)
        ids = np.full((len(idx), L), pad, dtype=np.int64)
        mask = np.zeros((len(idx), L), dtype=np.float32)
        for r, k in enumerate(idx):
            ids[r, :len(seqs[k])] = seqs[k]
            mask[r, :len(seqs[k])] = 1
        e = model.forward(ids, mask)
        if out is None:
            out = np.zeros((len(seqs), e.shape[1]), dtype=np.float32)
        out[idx] = e
        i = j
    return out


def load_bert(d, fast_gelu=False):
    cfg = json.load(open(d + "/config.json"))
    sb = json.load(open(d + "/sentence_bert_config.json"))
    vocab = json.load(open(d + "/tokenizer.json", encoding="utf-8"))["model"]["vocab"]
    nl = cfg["num_hidden_layers"]
    return Bert(pack_bert(load_safetensors(d + "/model.safetensors"), nl), vocab, nl,
                cfg["num_attention_heads"], sb["max_seq_length"], cfg["layer_norm_eps"], fast_gelu)


def load_student(npz_path, vocab_dir, max_len=128):
    z = np.load(npz_path)
    ws = [z[f"arr_{i}"] for i in range(len(z.files) - 1)]
    nl, nh, eps = z["meta"]
    vocab = json.load(open(vocab_dir + "/tokenizer.json", encoding="utf-8"))["model"]["vocab"]
    return Bert(ws[:-2], vocab, int(nl), int(nh), max_len, float(eps), fast_gelu=True, head=(ws[-2], ws[-1]))
