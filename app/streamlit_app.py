"""
streamlit_app.py

Interactive UI for the Visual Product Recommender — upgraded edition.

Tabs:
  1. Visual Search   — upload an image, pick a model, see explainable top-K results
  2. Model Analytics — Precision@K / Recall@K / latency comparison across models
  3. Embedding Space — interactive 2D (t-SNE) map of the learned embedding space
  4. About           — project & methodology summary

Run with:
    streamlit run app/streamlit_app.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
import config  # noqa: E402
from src.preprocessing import preprocess_pil  # noqa: E402
from src.baseline_similarity import SimilaritySearcher  # noqa: E402

st.set_page_config(
    page_title="Visual Product Recommender",
    page_icon="👗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# STYLING
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&family=Inter:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Poppins', sans-serif !important; }

.main { background: linear-gradient(180deg, #fafaff 0%, #f5f6fa 100%); }

.hero {
    background: linear-gradient(120deg, #6C63FF 0%, #A78BFA 45%, #F472B6 100%);
    padding: 2.1rem 2.4rem;
    border-radius: 20px;
    color: white;
    margin-bottom: 1.6rem;
    box-shadow: 0 12px 32px rgba(108, 99, 255, 0.25);
}
.hero h1 { margin: 0; font-size: 2.1rem; font-weight: 700; color: white; }
.hero p { margin: 0.4rem 0 0 0; opacity: 0.92; font-size: 1.02rem; }

.product-card {
    background: white;
    border-radius: 16px;
    padding: 0.7rem 0.7rem 1rem 0.7rem;
    box-shadow: 0 4px 18px rgba(30, 30, 60, 0.08);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    margin-bottom: 1rem;
    border: 1px solid rgba(108, 99, 255, 0.08);
}
.product-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 28px rgba(108, 99, 255, 0.22);
}
.product-card img { border-radius: 12px; }

.score-badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    color: white;
    margin-top: 6px;
}
.rank-badge {
    display: inline-block;
    background: #6C63FF;
    color: white;
    border-radius: 999px;
    width: 24px; height: 24px;
    text-align: center;
    font-size: 0.75rem;
    font-weight: 700;
    line-height: 24px;
    margin-right: 6px;
}

.metric-card {
    background: white;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    box-shadow: 0 4px 14px rgba(30, 30, 60, 0.06);
    border-left: 4px solid #6C63FF;
}

.warning-box {
    background: #FFF7ED;
    border-left: 4px solid #F59E0B;
    padding: 0.9rem 1.1rem;
    border-radius: 10px;
    color: #92400E;
    margin: 0.6rem 0 1rem 0;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1F1B3A 0%, #2D2755 100%);
}
section[data-testid="stSidebar"] * { color: #F1F0FF !important; }
</style>
""", unsafe_allow_html=True)


def score_color(score):
    """Green for strong match, amber for weak, red for very weak."""
    if score >= 0.75:
        return "#16A34A"
    elif score >= 0.5:
        return "#D97706"
    return "#DC2626"


# ---------------------------------------------------------------------------
# CACHED LOADERS
# ---------------------------------------------------------------------------
@st.cache_resource
def load_metadata():
    return pd.read_csv(config.SUBSET_METADATA_CSV)


@st.cache_resource
def load_searcher(embeddings_path, use_faiss=False):
    return SimilaritySearcher(embeddings_path, use_faiss=use_faiss)


@st.cache_resource
def load_embedding_model(model_path):
    import tensorflow as tf
    _patch_keras_initializer_bug()

    class _L2NormalizeLayer(tf.keras.layers.Layer):
        """
        Stand-in for the Lambda(l2_normalize) layer used in the Siamese model.
        Keras 3's reconstruction of pickled Python lambdas from .h5 is fragile
        (loses access to the 'tf' module, can't infer output shape). This class
        performs the identical L2-normalization directly in real Python code
        with no pickled bytecode involved, and is substituted in via
        custom_objects={"Lambda": _L2NormalizeLayer} — this works because the
        layer has no trainable weights, so there's nothing to mismatch when
        swapping in a different (but functionally identical) layer class.
        """
        def call(self, inputs, mask=None):
            return tf.math.l2_normalize(inputs, axis=1)

        def compute_output_shape(self, input_shape):
            return input_shape

        @classmethod
        def from_config(cls, config):
            base_config = {k: v for k, v in config.items() if k in ("name", "trainable", "dtype")}
            return cls(**base_config)

    return tf.keras.models.load_model(
        model_path, compile=False, safe_mode=False,
        custom_objects={"Lambda": _L2NormalizeLayer},
    )


_initializer_patched = False


def _patch_keras_initializer_bug():
    """
    Works around a known Keras 3 round-trip bug: saving a model to legacy
    .h5 format writes VarianceScaling-family initializer configs (used by
    GlorotUniform/HeUniform/LecunUniform/etc, the default kernel
    initializers for Conv2D/Dense) with extra 'input_axes'/'output_axes'
    keys. Each of these classes defines its OWN __init__ (doesn't just
    inherit VarianceScaling's), so we patch every affected class directly
    to ignore the two extra keys, letting existing trained .h5 files load
    without needing to retrain anything.
    """
    global _initializer_patched
    if _initializer_patched:
        return
    from tensorflow.keras import initializers as _init

    classes_to_patch = [
        _init.GlorotUniform, _init.GlorotNormal,
        _init.HeUniform, _init.HeNormal,
        _init.LecunUniform, _init.LecunNormal,
        _init.VarianceScaling,
        _init.RandomNormal, _init.RandomUniform, _init.TruncatedNormal,
        _init.Orthogonal,
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


@st.cache_data
def load_evaluation_results():
    path = os.path.join(config.OUTPUTS_DIR, "evaluation_results.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def compute_tsne(embeddings_path, sample_size=1200):
    from sklearn.manifold import TSNE
    embeddings = np.load(embeddings_path)
    index_df = pd.read_csv(config.EMBEDDINGS_INDEX_PATH)

    n = min(sample_size, len(embeddings))
    rng = np.random.RandomState(config.RANDOM_SEED)
    idx = rng.choice(len(embeddings), size=n, replace=False)

    tsne = TSNE(n_components=2, random_state=config.RANDOM_SEED, init="pca", perplexity=30)
    coords = tsne.fit_transform(embeddings[idx])

    sub_index = index_df.iloc[idx].reset_index(drop=True)
    sub_index["x"] = coords[:, 0]
    sub_index["y"] = coords[:, 1]
    return sub_index


MODEL_OPTIONS = {
    "Baseline (pretrained CNN)": config.BASELINE_EMBEDDINGS_PATH,
    "Fine-tuned (transfer learning)": config.FINETUNED_EMBEDDINGS_PATH,
    "Siamese Network (triplet loss)": config.SIAMESE_EMBEDDINGS_PATH,
}
MODEL_TO_MODELFILE = {
    "Baseline (pretrained CNN)": None,
    "Fine-tuned (transfer learning)": config.FINETUNED_MODEL_PATH,
    "Siamese Network (triplet loss)": config.SIAMESE_MODEL_PATH,
}


@st.cache_resource
def load_baseline_backbone():
    from src.feature_extraction import build_backbone
    return build_backbone()


def get_embedding_model(model_choice):
    if model_choice == "Baseline (pretrained CNN)":
        return load_baseline_backbone()
    return load_embedding_model(MODEL_TO_MODELFILE[model_choice])


def get_query_embedding(pil_image, model_choice):
    img_array = preprocess_pil(pil_image)
    batch = np.expand_dims(img_array, axis=0)
    model = get_embedding_model(model_choice)
    return model.predict(batch, verbose=0)[0]


# ---------------------------------------------------------------------------
# HERO
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>👗 Visual Product Recommender</h1>
    <p>Upload a product photo — deep-learning embeddings retrieve visually similar items,
    with explainable cosine-similarity scores.</p>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(config.SUBSET_METADATA_CSV):
    st.error(
        "Subset metadata not found. Run `python data/prepare_subset.py` and the "
        "feature-extraction / training scripts before launching the app."
    )
    st.stop()

df = load_metadata()
# Rebuild paths fresh as ABSOLUTE paths anchored to this script's own location
# (PROJECT_ROOT), rather than a relative path that depends on whatever the
# current working directory happens to be when the app is launched. Relying
# on relative + cwd was the root cause of images not resolving on Streamlit
# Cloud — this sidesteps that ambiguity entirely.
_subset_images_dir_abs = os.path.join(PROJECT_ROOT, config.SUBSET_IMAGES_DIR)
id_to_path = {row_id: os.path.join(_subset_images_dir_abs, f"{row_id}.jpg") for row_id in df["id"]}
id_to_name = dict(zip(df["id"], df.get("productDisplayName", df["id"])))
id_to_category = dict(zip(df["id"], df["articleType"]))

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    compare_mode = st.toggle(
        "🆚 Compare all 3 models",
        value=False,
        help="Run the same query through Baseline, Fine-tuned, and Siamese side-by-side."
    )
    model_choice = st.selectbox("Embedding model", list(MODEL_OPTIONS.keys()), disabled=compare_mode)
    k = st.slider("Top-K results", min_value=3, max_value=20, value=config.TOP_K if not compare_mode else 5)
    confidence_threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="If the best match scores below this, we flag it as an out-of-domain / low-confidence result."
    )
    use_faiss = st.toggle(
        "⚡ Use FAISS index",
        value=False,
        help="Exact nearest-neighbor search via FAISS instead of scikit-learn cosine similarity. "
             "Same results, built for scaling to much larger galleries."
    )
    show_gradcam = st.toggle(
        "🔥 Explain top match (Grad-CAM)",
        value=False,
        help="Show a heatmap of which parts of your query image drove the similarity to the #1 result."
    )
    st.markdown("---")
    st.markdown(
        "**Baseline** — raw pretrained CNN features.\n\n"
        "**Fine-tuned** — last layers fine-tuned on this dataset.\n\n"
        "**Siamese** — embedding space explicitly trained with triplet loss so visually/"
        "semantically similar products sit close together."
    )
    st.markdown("---")
    st.caption(f"Gallery size: {len(df)} products across {df['articleType'].nunique()} categories")

    with st.expander("🐛 Debug info", expanded=False):
        sample_id = df["id"].iloc[0]
        sample_path = id_to_path.get(sample_id, "N/A")
        st.code(
            f"cwd: {os.getcwd()}\n"
            f"PROJECT_ROOT: {PROJECT_ROOT}\n"
            f"SUBSET_IMAGES_DIR (from config): {config.SUBSET_IMAGES_DIR}\n"
            f"Sample resolved path: {sample_path}\n"
            f"Sample path exists: {os.path.exists(sample_path)}",
            language="text",
        )

embeddings_path = MODEL_OPTIONS[model_choice]

# ---------------------------------------------------------------------------
# TABS
# ---------------------------------------------------------------------------
tab_search, tab_analytics, tab_embedding, tab_about = st.tabs(
    ["🔍 Visual Search", "📊 Model Analytics", "🌌 Embedding Space", "ℹ️ About"]
)

# ============================== TAB 1: SEARCH ==============================
def render_product_card(pid, rank, score):
    color = score_color(score)
    crown = "🏆 " if rank == 1 else ""
    st.markdown('<div class="product-card">', unsafe_allow_html=True)
    img_path = id_to_path.get(pid)
    if img_path and os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        # Show WHY, instead of just leaving a silent blank space — makes this
        # diagnosable instead of a mystery if it happens again.
        st.caption(f"⚠️ image not found:\n`{os.path.basename(img_path) if img_path else pid}`")
    st.markdown(
        f'<span class="rank-badge">{rank}</span>'
        f'<b>{crown}{str(id_to_name.get(pid, pid))[:26]}</b>',
        unsafe_allow_html=True,
    )
    st.caption(id_to_category.get(pid, ""))
    st.markdown(
        f'<span class="score-badge" style="background:{color}">sim: {score:.3f}</span>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


with tab_search:
    st.markdown(f"""
    <div style="display:flex; gap:14px; margin-bottom:1.2rem;">
        <div class="metric-card" style="flex:1;">📦 Gallery<br><b>{len(df)} products</b></div>
        <div class="metric-card" style="flex:1;">🏷️ Categories<br><b>{df['articleType'].nunique()}</b></div>
        <div class="metric-card" style="flex:1;">🧠 Mode<br><b>{'Compare all 3' if compare_mode else model_choice.split(' (')[0]}</b></div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a product image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        query_image = Image.open(uploaded_file)

        col_query, col_results = st.columns([1, 4])
        with col_query:
            st.markdown('<div class="product-card">', unsafe_allow_html=True)
            st.image(query_image, caption="Query image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if compare_mode:
            missing = [name for name, path in MODEL_OPTIONS.items() if not os.path.exists(path)]
            if missing:
                st.warning(
                    "Compare mode needs all three embedding sets. Missing: "
                    + ", ".join(missing) + ". Showing whichever are available."
                )

            with col_results:
                st.subheader("Side-by-side: Baseline vs Fine-tuned vs Siamese")
                cmp_cols = st.columns(3)
                for col, (name, path) in zip(cmp_cols, MODEL_OPTIONS.items()):
                    with col:
                        st.markdown(f"#### {name.split(' (')[0]}")
                        if not os.path.exists(path):
                            st.caption("Not trained yet.")
                            continue
                        searcher = load_searcher(path, use_faiss=use_faiss)
                        with st.spinner("Searching..."):
                            t0 = time.perf_counter()
                            q_emb = get_query_embedding(query_image, name)
                            results = searcher.retrieve_by_embedding(q_emb, k=k)
                            elapsed_ms = (time.perf_counter() - t0) * 1000
                        st.caption(f"⏱ {elapsed_ms:.0f} ms")
                        for i, r in enumerate(results):
                            render_product_card(r["id"], i + 1, r["score"])
        else:
            if not os.path.exists(embeddings_path):
                st.warning(
                    f"Embeddings for **{model_choice}** not found at `{embeddings_path}`. "
                    "Run the corresponding pipeline stage first (see README)."
                )
            else:
                searcher = load_searcher(embeddings_path, use_faiss=use_faiss)
                with st.spinner("Extracting features & searching..."):
                    t0 = time.perf_counter()
                    query_embedding = get_query_embedding(query_image, model_choice)
                    results = searcher.retrieve_by_embedding(query_embedding, k=k)
                    elapsed_ms = (time.perf_counter() - t0) * 1000

                with col_results:
                    top_score = results[0]["score"] if results else 0.0
                    st.subheader(f"Top-{k} similar products")
                    st.caption(f"⏱ Query processed in {elapsed_ms:.1f} ms  ·  Model: {model_choice}"
                               + ("  ·  ⚡ FAISS" if use_faiss else ""))

                    if top_score < confidence_threshold:
                        st.markdown(
                            f'<div class="warning-box">⚠️ <b>Low-confidence match</b> — the best result '
                            f'scored only {top_score:.2f} cosine similarity, below your {confidence_threshold:.2f} '
                            f'threshold. This usually means the uploaded image doesn\'t closely resemble any '
                            f'category in the gallery ({", ".join(sorted(df["articleType"].unique()))}). '
                            f'Results below are still the closest available matches, but treat them as low-confidence.</div>',
                            unsafe_allow_html=True,
                        )

                    cols = st.columns(5)
                    for i, r in enumerate(results):
                        with cols[i % 5]:
                            render_product_card(r["id"], i + 1, r["score"])

                    if show_gradcam and results:
                        st.markdown("---")
                        st.markdown("#### 🔥 Why did the top result match?")
                        try:
                            from src.gradcam import compute_similarity_gradcam, overlay_heatmap
                            gradcam_model = get_embedding_model(model_choice)
                            preprocessed = preprocess_pil(query_image)
                            target_row = results[0]["row"]
                            target_embedding = searcher.embeddings[target_row]
                            heatmap = compute_similarity_gradcam(gradcam_model, preprocessed, target_embedding)
                            original_resized = np.array(query_image.convert("RGB").resize(config.IMG_SIZE))
                            overlay = overlay_heatmap(original_resized, heatmap)

                            gc1, gc2 = st.columns(2)
                            with gc1:
                                st.image(original_resized, caption="Your query image", use_container_width=True)
                            with gc2:
                                st.image(
                                    overlay,
                                    caption=f"Regions driving the match to '{id_to_name.get(results[0]['id'], results[0]['id'])}'",
                                    use_container_width=True,
                                )
                            st.caption(
                                "Red/yellow = regions of your image that most increased similarity to the "
                                "#1 result. Blue = regions that mattered least."
                            )
                        except Exception as e:
                            st.caption(f"Grad-CAM explanation unavailable for this model/image ({e}).")
    else:
        st.info("👆 Upload an image to get started. Try the **Compare all 3 models** toggle in the sidebar for the most impressive view.")
        st.markdown("#### Browse example products from the gallery")
        sample_ids = df["id"].sample(min(10, len(df)), random_state=1).tolist()
        cols = st.columns(5)
        for i, pid in enumerate(sample_ids):
            with cols[i % 5]:
                if os.path.exists(id_to_path[pid]):
                    st.markdown('<div class="product-card">', unsafe_allow_html=True)
                    st.image(id_to_path[pid], caption=id_to_category.get(pid, ""), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

# ============================== TAB 2: ANALYTICS ==============================
with tab_analytics:
    st.subheader("Baseline vs Fine-tuned vs Siamese — quantitative comparison")
    results_json = load_evaluation_results()

    if results_json is None:
        st.info(
            "No evaluation results yet. Run `python run_pipeline.py --stage evaluate` "
            "after generating embeddings for at least two models to populate this dashboard."
        )
    else:
        models_present = list(results_json.keys())
        cols = st.columns(len(models_present))
        for col, name in zip(cols, models_present):
            latency = results_json[name].get("avg_query_latency_ms", None)
            p_at_10 = results_json[name]["precision_at_k"].get("10", results_json[name]["precision_at_k"].get(10))
            with col:
                st.markdown(
                    f'<div class="metric-card"><b>{name.title()}</b><br>'
                    f'Precision@10: <b>{p_at_10:.3f}</b><br>'
                    f'Latency: <b>{latency:.1f} ms</b></div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### Precision@K")
        prec_rows = []
        for name, metrics in results_json.items():
            for k_val, v in metrics["precision_at_k"].items():
                prec_rows.append({"model": name, "K": int(k_val), "precision": v})
        prec_df = pd.DataFrame(prec_rows)
        fig = px.line(prec_df, x="K", y="precision", color="model", markers=True,
                      color_discrete_sequence=["#6C63FF", "#F472B6", "#16A34A"])
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Recall@K")
        rec_rows = []
        for name, metrics in results_json.items():
            for k_val, v in metrics["recall_at_k"].items():
                rec_rows.append({"model": name, "K": int(k_val), "recall": v})
        rec_df = pd.DataFrame(rec_rows)
        fig2 = px.line(rec_df, x="K", y="recall", color="model", markers=True,
                       color_discrete_sequence=["#6C63FF", "#F472B6", "#16A34A"])
        fig2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Query latency")
        lat_df = pd.DataFrame([
            {"model": name, "latency_ms": metrics.get("avg_query_latency_ms", 0)}
            for name, metrics in results_json.items()
        ])
        fig3 = px.bar(lat_df, x="model", y="latency_ms", color="model",
                      color_discrete_sequence=["#6C63FF", "#F472B6", "#16A34A"])
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

        comparison_img = os.path.join(config.OUTPUTS_DIR, "retrieval_comparison.png")
        if os.path.exists(comparison_img):
            st.markdown("### Qualitative comparison (sample queries)")
            st.image(comparison_img, use_container_width=True)

# ============================== TAB 3: EMBEDDING SPACE ==============================
with tab_embedding:
    st.subheader("2D map of the learned embedding space (t-SNE)")
    st.caption(
        "Each point is a product. Tight, well-separated clusters by color (category) "
        "indicate a more semantically organized embedding space — this is the visual "
        "evidence that the Siamese network improves on the raw baseline."
    )

    available_models = {name: path for name, path in MODEL_OPTIONS.items() if os.path.exists(path)}
    if not available_models:
        st.info("No embeddings available yet. Run the pipeline first.")
    else:
        tsne_model_choice = st.selectbox("Embedding space to visualize", list(available_models.keys()), key="tsne_model")
        with st.spinner("Computing t-SNE projection (cached after first run)..."):
            proj_df = compute_tsne(available_models[tsne_model_choice])
            proj_df["category"] = proj_df["id"].map(id_to_category)
            proj_df["name"] = proj_df["id"].map(id_to_name)

        fig4 = px.scatter(
            proj_df, x="x", y="y", color="category", hover_name="name",
            opacity=0.75, height=650,
            color_discrete_sequence=px.colors.qualitative.Bold,
        )
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white", legend_title_text="Category")
        fig4.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
        st.plotly_chart(fig4, use_container_width=True)

# ============================== TAB 4: ABOUT ==============================
with tab_about:
    st.subheader("About this project")
    st.markdown(f"""
An image-based recommendation system that retrieves visually similar products using
deep-learning embeddings, transfer learning, and a Siamese network trained with triplet
loss on a subset of the **Fashion Product Images** dataset.

**Pipeline**
1. Curated subset — {df['articleType'].nunique()} categories, {len(df)} images
2. Preprocessing — resize to 224×224, ImageNet normalization
3. Baseline embeddings — frozen pretrained CNN ({config.BACKBONE})
4. Transfer learning — fine-tune the last layers on the subset
5. Siamese network — triplet loss (anchor / positive / negative) for a semantically organized embedding space
6. Retrieval — cosine similarity (optionally FAISS) over precomputed embeddings
7. Evaluation — Precision@K, Recall@K, latency, qualitative comparison

**Categories in this gallery:** {", ".join(sorted(df['articleType'].unique()))}

**Note on scope:** this system only knows about the categories above. Uploading an
image outside that domain (e.g. a random object) will still return a "closest match"
by construction — that's expected retrieval-system behavior, flagged in the Search tab
via the confidence threshold.
    """)
