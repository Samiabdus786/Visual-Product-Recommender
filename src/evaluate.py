"""
evaluate.py

Evaluates and compares the three embedding spaces:
  1. Baseline (raw pretrained CNN features)
  2. Fine-tuned (transfer learning, last layers fine-tuned)
  3. Siamese (triplet-loss trained embedding)

Metrics:
  - Precision@K, Recall@K   (relevance = same articleType as the query)
  - Inference time per query
  - Embedding generation time (reported separately per stage's log)

Also produces a side-by-side visual grid (matplotlib figure) of top-K
retrievals for a handful of sample queries, saved to outputs/.

Usage:
    python src/evaluate.py
"""

import os
import sys
import time
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.baseline_similarity import SimilaritySearcher  # noqa: E402


def precision_recall_at_k(searcher, df, id_to_category, query_ids, k):
    precisions, recalls = [], []
    for qid in query_ids:
        category = id_to_category[qid]
        total_relevant = sum(1 for v in id_to_category.values() if v == category) - 1  # exclude query itself
        if total_relevant <= 0:
            continue

        results = searcher.retrieve_by_id(qid, k=k)
        retrieved_ids = [r["id"] for r in results]
        relevant_retrieved = sum(1 for rid in retrieved_ids if id_to_category.get(rid) == category)

        precisions.append(relevant_retrieved / k)
        recalls.append(relevant_retrieved / total_relevant)

    return float(np.mean(precisions)), float(np.mean(recalls))


def measure_query_latency(searcher, query_ids, k, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        for qid in query_ids[:20]:
            start = time.perf_counter()
            searcher.retrieve_by_id(qid, k=k)
            times.append(time.perf_counter() - start)
    return float(np.mean(times)) * 1000  # ms


def evaluate_all():
    df = pd.read_csv(config.SUBSET_METADATA_CSV)
    id_to_category = dict(zip(df["id"], df["articleType"]))

    rng = np.random.RandomState(config.RANDOM_SEED)
    all_ids = df["id"].tolist()
    n_queries = max(1, int(len(all_ids) * (1 - config.TRAIN_TEST_SPLIT)))
    query_ids = list(rng.choice(all_ids, size=n_queries, replace=False))

    embedding_sets = {
        "baseline": config.BASELINE_EMBEDDINGS_PATH,
        "finetuned": config.FINETUNED_EMBEDDINGS_PATH,
        "siamese": config.SIAMESE_EMBEDDINGS_PATH,
    }

    results = {}
    searchers = {}
    for name, path in embedding_sets.items():
        if not os.path.exists(path):
            print(f"[SKIP] {name}: embeddings not found at {path} (run the corresponding stage first)")
            continue

        searcher = SimilaritySearcher(path)
        searchers[name] = searcher

        model_metrics = {"precision_at_k": {}, "recall_at_k": {}}
        for k in config.EVAL_K_VALUES:
            p, r = precision_recall_at_k(searcher, df, id_to_category, query_ids, k)
            model_metrics["precision_at_k"][k] = round(p, 4)
            model_metrics["recall_at_k"][k] = round(r, 4)

        model_metrics["avg_query_latency_ms"] = round(
            measure_query_latency(searcher, query_ids, k=config.TOP_K), 3
        )

        results[name] = model_metrics
        print(f"\n[{name.upper()}]")
        print(json.dumps(model_metrics, indent=2))

    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    with open(os.path.join(config.OUTPUTS_DIR, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics -> {os.path.join(config.OUTPUTS_DIR, 'evaluation_results.json')}")

    if len(searchers) >= 2:
        plot_comparison(searchers, df, query_ids[:3], k=5)

    return results


def plot_comparison(searchers, df, sample_query_ids, k=5):
    """Visual grid: rows = models, columns = query + top-k retrieved images."""
    id_to_path = dict(zip(df["id"], df["image_path"]))
    model_names = list(searchers.keys())

    n_rows = len(model_names) * len(sample_query_ids)
    n_cols = k + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    row = 0
    for qid in sample_query_ids:
        for model_name in model_names:
            searcher = searchers[model_name]
            results = searcher.retrieve_by_id(qid, k=k)

            ax = axes[row, 0]
            ax.imshow(Image.open(id_to_path[qid]))
            ax.set_title(f"QUERY\n({model_name})", fontsize=8)
            ax.axis("off")

            for col, r in enumerate(results, start=1):
                ax = axes[row, col]
                ax.imshow(Image.open(id_to_path[r["id"]]))
                ax.set_title(f"{r['score']:.2f}", fontsize=8)
                ax.axis("off")

            row += 1

    plt.tight_layout()
    out_path = os.path.join(config.OUTPUTS_DIR, "retrieval_comparison.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visual comparison -> {out_path}")


if __name__ == "__main__":
    evaluate_all()
