#Live Link:
https://visual-appuct-recommender-vueqmanj.streamlit.app

# Visual Product Recommender

An image-based recommendation system that retrieves visually similar
products using deep learning embeddings. It combines **transfer learning**
(pretrained CNN feature extraction + fine-tuning) with a **Siamese network**
trained with **triplet loss** on a subset of the
[Fashion Product Images dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset),
providing fast, interactive, and explainable (cosine-distance-based) visual
search.

---

## 1. Problem Statement

Traditional keyword search fails to capture visual similarity — style,
texture, silhouette, color palette — that a shopper actually cares about.
This project retrieves visually similar products directly from an uploaded
image, using deep visual embeddings instead of text metadata.

## 2. System Overview

```
 ┌───────────────┐     ┌───────────────────┐     ┌───────────────────┐     ┌──────────────────┐
 │ Image Upload  │ --> │ Feature Extraction │ --> │ Similarity Search │ --> │ Top-K Retrieval  │
 │  (Streamlit)  │     │  (CNN embedding)   │     │ (cosine / FAISS)  │     │  + explanation   │
 └───────────────┘     └───────────────────┘     └───────────────────┘     └──────────────────┘
                                 ▲
                                 │ optionally replaced by
                        ┌────────────────────┐
                        │  Siamese Network    │
                        │ (triplet-loss tuned)│
                        └────────────────────┘
```

Three embedding spaces are built and compared:

| Stage | Description | Script |
|---|---|---|
| **Baseline** | Frozen pretrained ResNet50/EfficientNet, GAP pooled features | `src/feature_extraction.py` |
| **Fine-tuned** | Last N layers of the backbone fine-tuned on the subset (classification objective) | `src/transfer_learning.py` |
| **Siamese** | Shared-weight network trained with (Anchor, Positive, Negative) triplets + triplet loss | `src/train_siamese.py` |

## 3. Project Structure

```
visual_product_recommender/
├── config.py                   # all paths & hyperparameters
├── requirements.txt
├── run_pipeline.py             # runs the full pipeline end-to-end
├── data/
│   └── prepare_subset.py       # builds the curated dataset subset
├── src/
│   ├── preprocessing.py        # resize/normalize/augment
│   ├── feature_extraction.py   # baseline pretrained-CNN embeddings
│   ├── baseline_similarity.py  # cosine similarity / FAISS retrieval
│   ├── transfer_learning.py    # fine-tune last layers, extract embeddings
│   ├── siamese_dataset.py      # triplet generator
│   ├── siamese_model.py        # Siamese architecture + triplet loss
│   ├── train_siamese.py        # trains Siamese net, extracts embeddings
│   └── evaluate.py             # Precision@K, Recall@K, latency, visual comparison
├── app/
│   └── streamlit_app.py        # interactive UI
├── embeddings/                 # generated .npy embeddings + id index
├── models/                     # saved Keras models (.h5)
└── outputs/                    # evaluation_results.json, comparison plots
```

## 4. Setup

### 4.1 Install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4.2 Download the dataset
The original Fashion Product Images dataset on Kaggle is about 25GB, which is a lot to
download and store just for a subset of ~2000 images. So I used the **small** variant
instead — same products, same `styles.csv` metadata, just lower-res images (~80x60
instead of 2400x1600) and only a few hundred MB. Since the pipeline resizes everything
to 224x224 for the CNN anyway, the lower source resolution doesn't cost anything here.

1. Download **Fashion Product Images (Small)** from Kaggle:
   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
2. Extract it so that you have:
   ```
   data/raw/fashion-product-images-small/styles.csv
   data/raw/fashion-product-images-small/images/<id>.jpg
   ```
   This matches the default paths already set in `config.py` — no changes needed.

> If you specifically need the full high-res dataset for something else, it's here:
> [fashion-product-images-dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)
> (~25GB) — just repoint `RAW_DATASET_DIR` in `config.py` at it. Wasn't necessary for this project.

## 5. Running the Pipeline

Run everything end-to-end:
```bash
python run_pipeline.py --stage all
```

Or run stage-by-stage (recommended the first time, so you can inspect
intermediate outputs):

```bash
# 1. Build the curated subset (5-8 categories, ~200-300 imgs/category)
python run_pipeline.py --stage subset

# 2. Baseline: pretrained CNN embeddings + cosine similarity
python run_pipeline.py --stage baseline

# 3. Transfer learning: fine-tune last layers, extract fine-tuned embeddings
python run_pipeline.py --stage finetune

# 4. Siamese network: build triplets, train with triplet loss, extract embeddings
python run_pipeline.py --stage siamese

# 5. Evaluate & compare all three (Precision@K, Recall@K, latency, visual grid)
python run_pipeline.py --stage evaluate
```

Each stage writes its outputs so later stages (and the Streamlit app) can
find them:
- `embeddings/*.npy` + `embeddings/embeddings_index.csv`
- `models/finetuned_backbone.h5`, `models/siamese_embedding_model.h5`
- `outputs/evaluation_results.json`, `outputs/retrieval_comparison.png`

## 6. Launch the Interactive App

```bash
streamlit run app/streamlit_app.py
```

Features:
- Upload any product image
- Switch between Baseline / Fine-tuned / Siamese embedding models
- Adjustable Top-K
- Cosine-similarity score shown per result (explainability)
- Query latency displayed live

## 7. Methodology Recap

1. **Dataset subsetting** — 5–8 categories, ~200–300 images each (~1500–2500 total), for tractable training within limited compute. Used the small variant of the Kaggle dataset for this (full dataset is ~25GB — see Section 4.2 for why).
2. **Preprocessing** — resize to 224×224, normalize with ImageNet mean/std.
3. **Feature extraction** — pretrained ResNet50/EfficientNet with the classification head removed → global embedding via GlobalAveragePooling.
4. **Baseline similarity** — cosine similarity (scikit-learn) or FAISS flat index over the embeddings.
5. **Transfer learning** — freeze most backbone layers, fine-tune the last N on the subset with a classification objective, then re-extract embeddings from the fine-tuned backbone.
6. **Siamese network** — construct (Anchor, Positive, Negative) triplets (Positive = same category, Negative = different category), train a shared-weight embedding network with **triplet loss** so the embedding space is explicitly optimized for similarity rather than classification.
7. **Evaluation** — Precision@K / Recall@K (relevance = same category), qualitative before/after visual comparisons, and latency benchmarks — comparing Baseline vs Fine-tuned vs Siamese.

## 8. Evaluation Metrics

- **Precision@K** = (# relevant items in top-K) / K
- **Recall@K** = (# relevant items in top-K) / (total relevant items in the subset)
- **Qualitative** — side-by-side visual grids (`outputs/retrieval_comparison.png`)
- **Performance** — average query latency (embedding extraction + retrieval), reported in `outputs/evaluation_results.json`

Relevance is defined as sharing the same `articleType` category as the
query image (a standard, reproducible proxy for "visually/semantically
similar" on this dataset). This is intentionally simple to keep the project
self-contained; a stronger notion of relevance (e.g. human-labeled pairs)
could replace it in a production setting.

## 9. Tech Stack

- **TensorFlow / Keras** — pretrained backbones, transfer learning, Siamese network, triplet loss
- **NumPy / scikit-learn** — cosine similarity, Precision/Recall metrics
- **FAISS** (optional) — fast exact/approximate nearest-neighbor search over embeddings
- **Streamlit** — interactive upload-and-search UI
- **OpenCV / Pillow** — image I/O and augmentation
- **Pandas** — metadata handling

## 10. Results

Ran all three models on the 2000-image subset (8 categories, 250 images each) and evaluated them. Here's what came out:

| Model | Precision@10 | Recall@10 | Avg query latency |
|---|---|---|---|
| Baseline (pretrained ResNet50) | 0.720 | 0.029 | 23.2 ms |
| Fine-tuned (last layers retrained) | **0.911** | 0.037 | 23.9 ms |
| Siamese (triplet loss) | 0.827 | 0.033 | **1.3 ms** |

Both fine-tuning and the Siamese network clearly beat the raw pretrained baseline, which was the main point. Fine-tuning actually edged out Siamese on precision — that tracks, since it's directly trained to classify the same categories the evaluation checks against, so it has a bit of an unfair advantage on this specific metric compared to Siamese's more general similarity objective.

Where Siamese wins is speed: about 18x faster per query than the other two (1.3ms vs ~24ms). That's because its embeddings are only 128 dimensions instead of 2048, so there's way less to compare during retrieval. Doesn't matter much at 2000 images, but if this were scaled to a real catalog with millions of products, that difference adds up fast.

Recall@K looks low across the board (under 0.08 even at K=20) — that's not the models failing, it's just math. Each category only has ~250 images in the gallery, so pulling the top 20 can never recall more than 20/250 ≈ 8% of what's relevant, no matter how good the ranking is. Would need bigger per-category pools to push that number up.

Didn't get a chance to try bumping the Siamese training epochs or embedding dimension to see if it'd close the gap with fine-tuning — that'd be the next thing to test with more time.

## 11. Enhanced Edition — What's New

**Crash-resilient training.** `transfer_learning.py` and `train_siamese.py` now
checkpoint after every epoch (`models/checkpoints/`). If training is interrupted
(power loss, laptop sleep, etc.), simply re-run the same command — it automatically
resumes from the last completed epoch instead of starting over. The checkpoint is
deleted automatically once training finishes successfully.

**Upgraded Streamlit app** (`app/streamlit_app.py`), now a 4-tab experience:

1. **🔍 Visual Search** — custom-styled product cards, color-coded similarity badges
   (green/amber/red), and automatic **out-of-domain detection**: if the best match
   scores below a user-adjustable confidence threshold, the app surfaces a clear
   warning explaining the image likely doesn't belong to any trained category,
   instead of silently returning a misleading "best guess".
2. **📊 Model Analytics** — live Precision@K / Recall@K line charts and a latency
   bar chart comparing Baseline vs Fine-tuned vs Siamese, read directly from
   `outputs/evaluation_results.json` (auto-updates whenever you re-run `--stage evaluate`).
3. **🌌 Embedding Space** — an interactive t-SNE 2D projection (Plotly) of any
   embedding set, colored by category and hoverable by product name. This is the
   single most convincing visual for a demo/report: tight, well-separated clusters
   are direct visual evidence that the Siamese network organizes the embedding
   space better than the raw baseline.
4. **ℹ️ About** — auto-generated project/methodology summary from your actual
   gallery composition (category names, image counts) so it stays accurate as you
   change `config.CATEGORIES`.

To use the Embedding Space and Analytics tabs, run at least two of the three
pipeline stages (baseline/finetune/siamese) plus `--stage evaluate`.

## 12. Advanced Additions

Added three things on top of the base project to push it further:

**Grad-CAM explainability.** The Search tab now has a "🔥 Explain top match"
toggle. It shows a heatmap over your uploaded image highlighting which regions
actually drove the similarity score to the #1 result — not just a number, but
a visual answer to "why did it think these look alike." This works a bit
differently from textbook Grad-CAM (which explains a classifier's predicted
class) since these models don't have a classification head — instead it
computes gradients of the *similarity score itself* with respect to the last
conv layer, so it's explaining the retrieval decision directly. Implementation
is in `src/gradcam.py`.

**FAISS wired in properly.** `SimilaritySearcher` always supported a
`use_faiss` flag for exact nearest-neighbor search via a FAISS flat index
instead of scikit-learn's `cosine_similarity`, but the app never actually
used it — it always ran on the sklearn path by default. Added a toggle in the
sidebar so it's now genuinely switchable and demonstrable, not just dead code
sitting in a file.

**FastAPI backend.** `api/main.py` exposes the same recommender as a REST API
(`POST /search`, plus `/health` and `/models`), independent of the Streamlit
UI. Run it with:
```bash
uvicorn api.main:app --reload --port 8000
```
then open `http://localhost:8000/docs` for an interactive Swagger UI to try
it. This is the same underlying model-loading logic as the Streamlit app
(including the same Keras deserialization fixes), just exposed as a normal
HTTP service instead of a page — the kind of separation that'd actually make
sense if this were deployed as part of a bigger system rather than a
standalone demo.

**Live deployment.** See `DEPLOYMENT.md` for the full walkthrough of getting
this running on Streamlit Community Cloud with a public URL instead of only
running locally.

## 13. Possible Extensions

- Swap category-based relevance for attribute-based relevance (color, pattern) for a richer evaluation signal
- Add a quadruplet or online hard-negative mining strategy to the Siamese training loop
- Approximate nearest neighbor indexing (FAISS IVF/HNSW) for scaling to the full ~44K-image dataset
- Docker + a small pytest suite for the core retrieval/preprocessing functions

