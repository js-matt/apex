"""Text clustering solution for Apex (Subnet 1).

Starts from the official baseline (TF-IDF + KMeans) with a few cheap upgrades:
  - LSA (TruncatedSVD) to densify TF-IDF before clustering
  - MiniBatchKMeans so 50K texts fit comfortably in the 90s / 1.5GB budget
  - Exact-duplicate texts share a label (social media is full of reposts)

Contract (see apex/shared/competition/src/competition/text_clustering/README.md):
  GET  /health  -> {"status": "healthy"}
  POST /cluster {"texts": [...]} -> {"cluster_ids": [...]}   (-1 = noise)

Constraints: CPU only, no internet, 1.5GB RAM, 90s, file < 50,000 chars.

Usage:
    python solution.py --port 8001
"""

import argparse
import re

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

SEED = 42


class ClusterRequest(BaseModel):
    texts: list[str]


class ClusterResponse(BaseModel):
    cluster_ids: list[int]


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def estimate_num_clusters(n_samples: int) -> int:
    k = int(np.sqrt(n_samples / 2))
    return max(2, min(k, 100))


def cluster_texts(texts: list[str]) -> list[int]:
    n = len(texts)
    if n < 2:
        return [0] * n

    processed = [preprocess_text(t) for t in texts]

    # Cluster unique non-empty texts only, then broadcast back.
    uniq: dict[str, int] = {}
    for t in processed:
        if t and t not in uniq:
            uniq[t] = len(uniq)
    docs = list(uniq)
    if len(docs) < 2:
        return [0 if t else -1 for t in processed]

    try:
        X = TfidfVectorizer(
            max_features=50_000,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        ).fit_transform(docs)
    except ValueError:  # vocabulary empty after min_df etc.
        return [0 if t else -1 for t in processed]

    n_comp = min(128, X.shape[1] - 1, len(docs) - 1)
    if n_comp >= 2:
        X = normalize(TruncatedSVD(n_components=n_comp, random_state=SEED).fit_transform(X))

    k = min(estimate_num_clusters(len(docs)), len(docs))
    labels = MiniBatchKMeans(
        n_clusters=k, random_state=SEED, n_init=3, batch_size=2048
    ).fit_predict(X)

    return [int(labels[uniq[t]]) if t else -1 for t in processed]


def make_app() -> FastAPI:
    app = FastAPI(title="Text Clustering Miner")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/cluster", response_model=ClusterResponse)
    def cluster(request: ClusterRequest) -> ClusterResponse:
        if not request.texts:
            raise HTTPException(status_code=400, detail="No texts provided")
        return ClusterResponse(cluster_ids=cluster_texts(request.texts))

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="info")
