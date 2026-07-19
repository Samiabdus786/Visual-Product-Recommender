"""
baseline_similarity.py

Cosine-similarity based retrieval over precomputed embeddings.
Also provides an optional FAISS index for fast approximate/exact search
over larger embedding sets.

Usage (standalone demo):
    python src/baseline_similarity.py --query_id 1163 --k 10
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


class SimilaritySearcher:
    """
    Wraps a set of embeddings + id index and exposes:
      - retrieve_by_embedding(query_vec, k)
      - retrieve_by_id(product_id, k)
    Uses scikit-learn cosine_similarity by default; use_faiss=True switches
    to a FAISS flat index (exact search, much faster for large N).
    """

    def __init__(self, embeddings_path, index_csv=None, use_faiss=False):
        self.embeddings = np.load(embeddings_path).astype("float32")
        index_csv = index_csv or config.EMBEDDINGS_INDEX_PATH
        self.index_df = pd.read_csv(index_csv)
        self.id_to_row = dict(zip(self.index_df["id"], self.index_df["row"]))
        self.row_to_id = dict(zip(self.index_df["row"], self.index_df["id"]))

        # L2-normalize once so cosine similarity == dot product
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        self.normed = self.embeddings / norms

        self.use_faiss = use_faiss
        self._faiss_index = None
        if use_faiss:
            self._build_faiss_index()

    def _build_faiss_index(self):
        import faiss
        dim = self.normed.shape[1]
        index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine sim
        index.add(self.normed)
        self._faiss_index = index

    def retrieve_by_embedding(self, query_vec, k=None, exclude_row=None):
        k = k or config.TOP_K
        q = query_vec.astype("float32").reshape(1, -1)
        q_norm = q / max(np.linalg.norm(q), 1e-8)

        if self.use_faiss:
            scores, rows = self._faiss_index.search(q_norm, k + (1 if exclude_row is not None else 0))
            rows, scores = rows[0], scores[0]
        else:
            sims = cosine_similarity(q_norm, self.normed)[0]
            order = np.argsort(-sims)
            take = k + (1 if exclude_row is not None else 0)
            rows = order[:take]
            scores = sims[rows]

        results = []
        for r, s in zip(rows, scores):
            if exclude_row is not None and r == exclude_row:
                continue
            results.append({"row": int(r), "id": self.row_to_id[int(r)], "score": float(s)})
        return results[:k]

    def retrieve_by_id(self, product_id, k=None):
        row = self.id_to_row[product_id]
        query_vec = self.embeddings[row]
        return self.retrieve_by_embedding(query_vec, k=k, exclude_row=row)


def demo(query_id, k):
    searcher = SimilaritySearcher(config.BASELINE_EMBEDDINGS_PATH)
    results = searcher.retrieve_by_id(query_id, k=k)
    print(f"Top-{k} similar products for id={query_id}:")
    for r in results:
        print(f"  id={r['id']}  cosine_sim={r['score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_id", type=int, required=True)
    parser.add_argument("--k", type=int, default=config.TOP_K)
    args = parser.parse_args()
    demo(args.query_id, args.k)
