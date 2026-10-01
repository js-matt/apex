"""Build the candidate submission from the leader-lineage file by exact, checked edits.

    python tools/build_candidate.py BASE.py OUT.py [--umap-dim 10] [--umap-w .5] [--spec-w .5] [--no-umap] [--no-spec]
Every substitution must match exactly once (typos fail loudly). Prints the final character count (limit 50,000).

Changes vs base (181049):
  1. arXiv/short path (BV): append a UMAP-weighted spectral block (16 dims, 10 neighbours) to the smoothed
     features before linkage. Back-end parameters unchanged.
  2. Reddit/long path: append a numpy-UMAP layout (10 dims, 200 epochs) of the pre-spectral features as an extra block.
  3. Space: drop never-used constants/import, the >15,000-text branch (live subsets are 5,000; larger inputs still
     take the existing 6,000-sample path), two no-op expressions, and share one eigen-solver helper.
"""
import argparse
import lzma
import re
from pathlib import Path

import numpy as np

RAW = "l.FORMAT_RAW,None,[{'id':l.FILTER_LZMA2}]"


def b2_decode(s):  # leader's B2: 15 bits per char, MSB-first byte stream
    acc = nb = 0
    out = bytearray()
    for ch in s:
        acc = acc << 15 | (ord(ch) - 19968)
        nb += 15
        while nb >= 8:
            nb -= 8
            out.append(acc >> nb & 255)
    return bytes(out)


def b2_encode(b):
    bits = "".join(f"{x:08b}" for x in b)
    bits += "0" * (-len(bits) % 15)
    return "".join(chr(19968 + int(bits[i:i + 15], 2)) for i in range(0, len(bits), 15))


def bn_decode(s):  # leader's BN: 2 chars of byte length, then one big integer in base 2**15
    acc = 0
    for ch in s[2:]:
        acc = acc << 15 | (ord(ch) - 19968)
    return acc.to_bytes((ord(s[0]) - 19968) << 15 | (ord(s[1]) - 19968), "big")


def bn_encode(b):
    n, acc, digits = len(b), int.from_bytes(b, "big"), []
    while True:
        digits.append(acc & 32767)
        acc >>= 15
        if not acc:
            break
    return chr(19968 + (n >> 15)) + chr(19968 + (n & 32767)) + "".join(chr(19968 + d) for d in reversed(digits))


def repack(src, name, dec, enc, lc, lp, pb):
    """Re-compress a blob's payload as raw LZMA2 (no .xz container) with tuned literal/position bits; payload unchanged."""
    old = re.search(rf"^{name}='([^']*)'", src, re.M).group(1)
    raw = lzma.decompress(dec(old))
    new = enc(lzma.compress(raw, format=lzma.FORMAT_RAW, filters=[{"id": lzma.FILTER_LZMA2, "preset": 9 | lzma.PRESET_EXTREME, "lc": lc, "lp": lp, "pb": pb}]))
    assert lzma.decompress(dec(new), lzma.FORMAT_RAW, None, [{"id": lzma.FILTER_LZMA2}]) == raw
    return src.replace(f"{name}='{old}'", f"{name}='{new}'"), len(old) - len(new)

FZ_EV = '''def FZ(Z,k):
\td,x=Q(n_neighbors=k,metric='cosine').fit(Z).kneighbors(Z);n=B(Z);g=A.maximum(d[:,1:]-d[:,1:2],0);lo=A.zeros(n);hi=A.full(n,A.inf);m=A.ones(n);t=A.log2(k)
\tfor _ in K(64):s=A.exp(-g/m[:,None]).sum(1)>t;hi=A.where(s,m,hi);lo=A.where(s,lo,m);m=A.where(A.isinf(hi),m*2,(lo+hi)/2)
\tm=A.maximum(m,1e-3*d[:,1:].mean(1));P=U((A.exp(-g/m[:,None]).ravel(),(A.repeat(A.arange(n),k-1),x[:,1:].ravel())),shape=(n,n));return P+P.T-P.multiply(P.T)
def EV(P,dim,n,**kw):G=A.asarray(P.sum(1)).ravel();G[G==0]=1.;H=1./A.sqrt(G);return eigsh(U(P.multiply(H[:,None]).multiply(H[None,:])),k=D(dim,n-2),which='LA',**kw)[1]
def SF(Z,dim=16,k=10):n=B(Z);return C(EV(FZ(Z,k)+U((A.ones(n),(A.arange(n),A.arange(n))),shape=(n,n)),dim,n,tol=.001,maxiter=300,v0=A.full(n,1./A.sqrt(n),dtype=A.float32)).astype(A.float32))
'''

UM = '''def UM(Z,dim=10,ep=200):
\tn=B(Z);P=FZ(Z,15).tocoo();r=A.random.default_rng(0);V=EV(P,dim+1,n+3,tol=1e-4,maxiter=n*5,v0=r.random(n));Y=V[:,:-1][:,::-1];Y=Y-Y.min(0);Y=(10*Y/A.maximum(Y.max(0),1e-12)+r.normal(scale=1e-4,size=Y.shape)).astype(A.float32)
\tr=A.random.default_rng(0);k=P.data>=P.data.max()/ep;h,t,w=P.row[k],P.col[k],P.data[k];e=w.max()/w;x=e.copy();a,b=1.8956,.8006
\tfor i in K(ep):
\t\tf=x<=i+1
\t\tif not f.any():continue
\t\tx[f]+=e[f];hh,tt=h[f],t[f];q=Y[hh]-Y[tt];s=(q*q).sum(1);c=A.maximum(s,1e-12);c=A.where(s>0,-2*a*b*c**(b-1)/(1+a*c**b),0);g=A.clip(c[:,None]*q,-4,4)*(1-i/ep);dY=A.zeros_like(Y);A.add.at(dY,hh,g);A.add.at(dY,tt,-g)
\t\thn=A.repeat(hh,5);tn=r.integers(0,n,hn.size);q=Y[hn]-Y[tn];s=(q*q).sum(1);g=A.clip((2*b/((.001+s)*(1+a*s**b)))[:,None]*q,-4,4)*(1-i/ep);g[tn==hn]=0;A.add.at(dY,hn,g);Y+=dY.astype(A.float32)
\treturn Y
'''


def sub(src, old, new):
    n = src.count(old)
    if n != 1:
        raise SystemExit(f"{old[:70]!r} matches {n} times")
    return src.replace(old, new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("out")
    ap.add_argument("--umap-dim", type=int, default=10)
    ap.add_argument("--umap-w", default=".5")
    ap.add_argument("--spec-w", default=".5")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--no-spec", action="store_true")
    ap.add_argument("--spec-dim", type=int, default=8)
    ap.add_argument("--spec-k", type=int, default=15)
    ap.add_argument("--umap-min-len", type=int, default=210, help="UMAP block only when the median text length exceeds this (0 = always)")
    a = ap.parse_args()
    src = Path(a.base).read_text(encoding="utf-8")
    base_len = len(src)
    lines = src.split("\n")
    # 3a. drop the >15,000-text branch (original L350-L360)
    i = lines.index("\t\tif J>Aj:")
    assert lines[i + 10] == "\t\t\treturn X", lines[i + 10]
    del lines[i:i + 11]
    src = "\n".join(lines)
    # 3d. raw-LZMA2 re-pack of both weight blobs (decoded weights are byte-identical; asserted in repack)
    src, saved_ak = repack(src, "Ak", b2_decode, b2_encode, 0, 0, 0)
    src, saved_bd = repack(src, "BD", bn_decode, bn_encode, 0, 0, 1)
    src = sub(src, "H=l.decompress(B2(G));", f"H=l.decompress(B2(G),{RAW});")
    src = sub(src, "C=l.decompress(BN(BD));", f"C=l.decompress(BN(BD),{RAW});")
    print(f"blob re-pack saved {saved_ak} + {saved_bd} chars")
    # 3b. dead names, unused import, no-ops
    for old, new in [
        ("from sklearn.neighbors import NearestNeighbors as Q,kneighbors_graph", "from sklearn.neighbors import NearestNeighbors as Q"),
        ("\nAf=1\n", "\n"), ("\nAj=15000\n", "\n"),
        ("A6,A7,BW=8192,4096,24", "A6,A7=8192,4096"), ("B5,BX,BY=4.,.5,13e1", "B5=4."),
        ("A8,B6,BZ=6000,4000,32", "A8,B6=6000,4000"), ("Ba,B7,Bb=.4,.35,0", "B7=.35"),
        ("A9,AA,Bc,Bd=1.,1.,1.,1.", "A9,AA=1.,1."),
        ("A9=AO if N else(AN if 230<AI<330 else 1.4)", "A9=AO if N else AN"),
        ("Y.append(C(k)*(1. if N else 1.))", "Y.append(C(k))"),
        # title= flag has identical branches (B9,BA == BB,BC): drop the plumbing
        ("B9,BA,BB,BC=.35,8000,.35,8000", "B9,BA=.35,8000"), ("def BL(P,title=False):\n\tK,L=(B9,BA)if title else(BB,BC);E=[]", "def BL(P):\n\tK,L=B9,BA;E=[]"),
        ("X=BL(e,True)", "X=BL(e)"), ("def BO(title=False):", "def BO():"), ("def BP(texts,title=False):", "def BP(texts):"),
        ("G=BO(title);", "G=BO();"), ("N=BP(L,True)if", "N=BP(L)if"),
        # RED: per-point moves are independent, so the inner loop vectorises exactly
        ("\t\tm=lab==i;s=Z[m]@M.T;bb,v=s.argmax(1),s.max(1)\n\t\tfor jj,k in enumerate(A.where(m)[0]):\n\t\t\tif v[jj]>=t:lab[k]=Cc[bb[jj]]\n",
         "\t\tm=A.where(lab==i)[0];s=Z[m]@M.T;o=s.max(1)>=t;lab[m[o]]=Cc[s.argmax(1)[o]]\n"),
        # web service: same endpoints, flatter
        ("def make_app():\n service=m(title='Apex v14')\n @service.get('/health')\n async def health():return {'status':'healthy'}\n @service.post('/cluster',response_model=ClusterResponse)\n async def cluster(request:ClusterRequest):\n  if not request.texts:raise AH(status_code=400,detail='No texts provided')\n  return {'cluster_ids':cluster_texts(request.texts)}\n return service\napp=make_app()\n",
         "app=m()\n@app.get('/health')\nasync def health():return{'status':'healthy'}\n@app.post('/cluster',response_model=ClusterResponse)\nasync def cluster(request:ClusterRequest):\n if not request.texts:raise AH(status_code=400,detail='No texts provided')\n return{'cluster_ids':cluster_texts(request.texts)}\n"),
        # 3c. leader's spectral block A_ now uses the shared eigen helper (same arithmetic, same eigsh arguments)
        ("G=A.asarray(F.sum(axis=1)).ravel();G[G==0]=1.;H=1./A.sqrt(G);I=U(F.multiply(H[:,None]).multiply(H[None,:]));K,J=eigsh(I,k=D(B,E-2),which='LA',tol=.001,maxiter=300,v0=A.full(E,1./A.sqrt(E),dtype=A.float32));return C(J.astype(A.float32))",
         "return C(EV(F,B,E,tol=.001,maxiter=300,v0=A.full(E,1./A.sqrt(E),dtype=A.float32)).astype(A.float32))"),
    ]:
        src = sub(src, old, new)
    if '"""Public Apex180663 adaptation; v14."""' in src:
        src = sub(src, '"""Public Apex180663 adaptation; v14."""', "#Apex180663 v14 lineage")
    # helpers go right after B1 (module level, before first use at runtime)
    anchor = "def B1(rel):"
    j = src.index(anchor)
    j = src.index("\n", j) + 1
    src = src[:j] + FZ_EV + (UM if not a.no_umap else "") + src[j:]
    # 1. short path: spectral block after BV's smoothing
    if not a.no_spec:
        src = sub(src, "\t\tif P.time()-S<Z*.62:O=BT(O,alpha=B7,k=40)\n",
                  f"\t\tif P.time()-S<Z*.62:O=BT(O,alpha=B7,k=40);O=C(A.hstack([O,SF(O,{a.spec_dim},{a.spec_k})*{a.spec_w}]))\n")
    # 2. long path: UMAP block next to the leader's spectral block
    if not a.no_umap:
        src = sub(src, "\t\t\tif AC is not None:G=C(A.hstack([G,AC]))\n",
                  "\t\t\tif AC is not None:\n\t\t\t\tY0=[G,AC]\n"
                  f"\t\t\t\tif {f'AI>{a.umap_min_len} and ' if a.umap_min_len else ''}P.perf_counter()-u<r*.5:Yu=UM(G,{a.umap_dim});Y0.append(C(Yu-Yu.mean(0))*{a.umap_w})\n"
                  "\t\t\t\tG=C(A.hstack(Y0))\n")
    Path(a.out).write_text(src, encoding="utf-8")
    print(f"{a.out}: {len(src)} chars (base {base_len}, limit 50,000, headroom {50000 - len(src)})")
    if len(src) >= 50000:
        raise SystemExit("over the 50,000-character limit")


if __name__ == "__main__":
    main()
