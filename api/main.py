"""
api/main.py

FastAPI backend for the Visual Product Recommender. Wraps the same
embeddings/models used by the Streamlit app behind a REST API, so the
recommender can be called from any client (a JS frontend, a mobile app,
another service) instead of only through the Streamlit UI.

Run with:
    uvicorn api.main:app --reload --port 8000

Then try:
    http://localhost:8000/docs   (interactive Swagger UI)
"""

import os
import sys
import io
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocessing import preprocess_pil  # noqa: E402
from src.baseline_similarity import SimilaritySearcher  # noqa: E402

MODEL_OPTIONS = {
    "baseline": config.BASELINE_EMBEDDINGS_PATH,
    "finetuned": config.FINETUNED_EMBEDDINGS_PATH,
    "siamese": config.SIAMESE_EMBEDDINGS_PATH,
}
MODEL_TO_MODELFILE = {
    "baseline": None,
    "finetuned": config.FINETUNED_MODEL_PATH,
    "siamese": config.SIAMESE_MODEL_PATH,
}

# ---------------------------------------------------------------------------
# In-memory caches (simple dicts — this is a small demo API, not a
# production model-serving setup with proper cache eviction/warmup)
# ---------------------------------------------------------------------------
_metadata_df = None
_searchers = {}
_embedding_models = {}
_baseline_model = None
_initializer_patched = False


def _patch_keras_initializer_bug():
    """Same fix as the Streamlit app: works around a Keras 3 bug deserializing
    VarianceScaling-family initializers (GlorotUniform etc.) from legacy .h5 files."""
    global _initializer_patched
    if _initializer_patched:
        return
    from tensorflow.keras import initializers as _init
    classes_to_patch = [
        _init.GlorotUniform, _init.GlorotNormal, _init.HeUniform, _init.HeNormal,
        _init.LecunUniform, _init.LecunNormal, _init.VarianceScaling,
        _init.RandomNormal, _init.RandomUniform, _init.TruncatedNormal, _init.Orthogonal,
    ]

    def _make_patched(orig_init):
        def patched(self, *args, **kwargs):
            kwargs.pop("input_axes", None)
            kwargs.pop("output_axes", None)
            return orig_init(self, *args, **kwargs)
        return patched

    for cls in classes_to_patch:
        cls.__init__ = _make_patched(cls.__init__)
    _initializer_patched = True


def get_metadata():
    global _metadata_df
    if _metadata_df is None:
        _metadata_df = pd.read_csv(config.SUBSET_METADATA_CSV)
    return _metadata_df


def get_searcher(model_name):
    if model_name not in _searchers:
        path = MODEL_OPTIONS[model_name]
        if not os.path.exists(path):
            raise HTTPException(404, f"Embeddings for '{model_name}' not found. Run the pipeline stage first.")
        _searchers[model_name] = SimilaritySearcher(path)
    return _searchers[model_name]


def get_embedding_model(model_name):
    import tensorflow as tf

    global _baseline_model
    if model_name == "baseline":
        if _baseline_model is None:
            from src.feature_extraction import build_backbone
            _baseline_model = build_backbone()
        return _baseline_model

    if model_name not in _embedding_models:
        _patch_keras_initializer_bug()
        model_path = MODEL_TO_MODELFILE[model_name]
        if not os.path.exists(model_path):
            raise HTTPException(404, f"Model file for '{model_name}' not found. Run the pipeline stage first.")

        class _L2NormalizeLayer(tf.keras.layers.Layer):
            def call(self, inputs, mask=None):
                return tf.math.l2_normalize(inputs, axis=1)

            def compute_output_shape(self, input_shape):
                return input_shape

            @classmethod
            def from_config(cls, cfg):
                base_config = {k: v for k, v in cfg.items() if k in ("name", "trainable", "dtype")}
                return cls(**base_config)

        _embedding_models[model_name] = tf.keras.models.load_model(
            model_path, compile=False, safe_mode=False,
            custom_objects={"Lambda": _L2NormalizeLayer},
        )
    return _embedding_models[model_name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up metadata on startup; models load lazily on first request per model
    if os.path.exists(config.SUBSET_METADATA_CSV):
        get_metadata()
    yield


app = FastAPI(
    title="Visual Product Recommender API",
    description="Image-based product retrieval via deep embeddings (Baseline / Fine-tuned / Siamese).",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SearchResult(BaseModel):
    id: int
    product_name: str
    category: str
    similarity: float


class SearchResponse(BaseModel):
    model: str
    query_latency_ms: float
    top_k: int
    results: list[SearchResult]


@app.get("/health")
def health():
    return {"status": "ok", "gallery_ready": os.path.exists(config.SUBSET_METADATA_CSV)}


@app.get("/models")
def list_models():
    """Which of the three embedding sets are actually trained and available."""
    return {
        name: {
            "embeddings_available": os.path.exists(path),
            "model_file_available": (
                True if name == "baseline" else os.path.exists(MODEL_TO_MODELFILE[name])
            ),
        }
        for name, path in MODEL_OPTIONS.items()
    }


@app.post("/search", response_model=SearchResponse)
async def search(
    file: UploadFile = File(..., description="Product image to search with"),
    model: str = Query("baseline", enum=["baseline", "finetuned", "siamese"]),
    k: int = Query(10, ge=1, le=50),
):
    import time

    if not os.path.exists(config.SUBSET_METADATA_CSV):
        raise HTTPException(400, "Gallery not built yet — run the pipeline first.")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(400, "Could not read the uploaded file as an image.")

    df = get_metadata()
    id_to_name = dict(zip(df["id"], df.get("productDisplayName", df["id"])))
    id_to_category = dict(zip(df["id"], df["articleType"]))

    searcher = get_searcher(model)
    embedding_model = get_embedding_model(model)

    t0 = time.perf_counter()
    preprocessed = preprocess_pil(image)
    batch = np.expand_dims(preprocessed, axis=0)
    query_embedding = embedding_model.predict(batch, verbose=0)[0]
    raw_results = searcher.retrieve_by_embedding(query_embedding, k=k)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    results = [
        SearchResult(
            id=int(r["id"]),
            product_name=str(id_to_name.get(r["id"], r["id"])),
            category=str(id_to_category.get(r["id"], "")),
            similarity=round(r["score"], 4),
        )
        for r in raw_results
    ]

    return SearchResponse(model=model, query_latency_ms=round(elapsed_ms, 2), top_k=k, results=results)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
