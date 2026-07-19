"""
config.py
Central configuration for the Visual Product Recommender project.
Edit the paths below to match where you've extracted the Kaggle
'Fashion Product Images' dataset.
"""

import os

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
# Root of the extracted Kaggle dataset. It should contain styles.csv and an
# "images/" folder with files like "1163.jpg".
#
# DEFAULT: the SMALL dataset variant (recommended) —
#   https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
# Same styles.csv / same product IDs as the full dataset, just lower-res
# images (~80x60 instead of 2400x1600) and only a few hundred MB instead of
# ~25GB. Since this pipeline resizes everything to 224x224 anyway, nothing
# is lost for this project's purposes.
RAW_DATASET_DIR = "data/raw/fashion-product-images-small"
RAW_STYLES_CSV = os.path.join(RAW_DATASET_DIR, "styles.csv")
RAW_IMAGES_DIR = os.path.join(RAW_DATASET_DIR, "images")

# ---------------------------------------------------------------------------
# ADVANCED / OPTIONAL: if you ever want the full high-resolution dataset
# instead, download the "fashion-product-images-dataset" (~25GB), extract it,
# and just repoint the three lines above at that folder — no other code
# changes needed. Not required or recommended for this project.

# Where the curated SUBSET (images + metadata) will be written to.
SUBSET_DIR = "data/subset"
SUBSET_IMAGES_DIR = os.path.join(SUBSET_DIR, "images")
SUBSET_METADATA_CSV = os.path.join(SUBSET_DIR, "subset_metadata.csv")

# Where computed embeddings / indices / trained models are stored.
EMBEDDINGS_DIR = "embeddings"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

BASELINE_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "baseline_embeddings.npy")
FINETUNED_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "finetuned_embeddings.npy")
SIAMESE_EMBEDDINGS_PATH = os.path.join(EMBEDDINGS_DIR, "siamese_embeddings.npy")
EMBEDDINGS_INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "embeddings_index.csv")  # row -> product id

SIAMESE_MODEL_PATH = os.path.join(MODELS_DIR, "siamese_embedding_model.h5")
FINETUNED_MODEL_PATH = os.path.join(MODELS_DIR, "finetuned_backbone.h5")

# ---------------------------------------------------------------------------
# DATASET SUBSETTING
# ---------------------------------------------------------------------------
# Chosen from the `articleType` column of styles.csv. Feel free to change —
# just make sure the values exist in your styles.csv.
CATEGORIES = [
    "Tshirts",
    "Shirts",
    "Casual Shoes",
    "Sports Shoes",
    "Dresses",
    "Handbags",
    "Watches",
    "Sunglasses",
]
IMAGES_PER_CATEGORY = 250      # ~200-300 as per spec
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------------------------
IMG_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# FEATURE EXTRACTION / BACKBONE
# ---------------------------------------------------------------------------
BACKBONE = "resnet50"          # "resnet50" or "efficientnet"
EMBEDDING_DIM = 128             # dim of the projection head used for Siamese training

# ---------------------------------------------------------------------------
# TRANSFER LEARNING (fine-tuning the backbone with a classification head)
# ---------------------------------------------------------------------------
FT_UNFREEZE_LAST_N_LAYERS = 15
FT_EPOCHS = 8
FT_BATCH_SIZE = 32
FT_LEARNING_RATE = 1e-4

# ---------------------------------------------------------------------------
# SIAMESE NETWORK (TRIPLET LOSS)
# ---------------------------------------------------------------------------
TRIPLETS_PER_EPOCH = 3000
SIAMESE_EPOCHS = 8
SIAMESE_BATCH_SIZE = 32
SIAMESE_LEARNING_RATE = 1e-4
TRIPLET_MARGIN = 0.3

# ---------------------------------------------------------------------------
# RETRIEVAL / EVALUATION
# ---------------------------------------------------------------------------
TOP_K = 10
EVAL_K_VALUES = [1, 5, 10, 20]
TRAIN_TEST_SPLIT = 0.85   # fraction of subset used as the "gallery" (search index);
                           # remainder used as held-out queries for evaluation
