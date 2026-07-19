"""
feature_extraction.py

Loads a pretrained CNN (ResNet50 or EfficientNetB0), strips the
classification head, and extracts a global embedding vector for every
image in the subset. These are the BASELINE embeddings (no fine-tuning).

Usage:
    python src/feature_extraction.py
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocessing import preprocess_image  # noqa: E402


def build_backbone(name=None):
    """
    Returns a Keras Model that maps a (224,224,3) image to a flat embedding
    vector, using a pretrained ImageNet backbone with the classification
    head removed (transfer learning feature extractor).
    """
    name = (name or config.BACKBONE).lower()
    input_shape = config.IMG_SIZE + (3,)

    if name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone '{name}'")

    base.trainable = False  # frozen for baseline extraction
    x = GlobalAveragePooling2D()(base.output)
    model = Model(inputs=base.input, outputs=x, name=f"{name}_feature_extractor")
    return model


def extract_embeddings(metadata_csv=None, output_path=None, backbone_name=None, batch_size=32):
    metadata_csv = metadata_csv or config.SUBSET_METADATA_CSV
    output_path = output_path or config.BASELINE_EMBEDDINGS_PATH

    df = pd.read_csv(metadata_csv)
    print(f"Extracting embeddings for {len(df)} images using backbone='{backbone_name or config.BACKBONE}'")

    model = build_backbone(backbone_name)

    embeddings = []
    ids = []

    paths = df["image_path"].tolist()
    id_list = df["id"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting features"):
        batch_paths = paths[i:i + batch_size]
        batch_ids = id_list[i:i + batch_size]
        batch_imgs = np.stack([preprocess_image(p) for p in batch_paths], axis=0)
        batch_embeds = model.predict(batch_imgs, verbose=0)
        embeddings.append(batch_embeds)
        ids.extend(batch_ids)

    embeddings = np.concatenate(embeddings, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embeddings)

    index_df = pd.DataFrame({"row": range(len(ids)), "id": ids})
    index_df.to_csv(config.EMBEDDINGS_INDEX_PATH, index=False)

    print(f"Saved embeddings {embeddings.shape} -> {output_path}")
    print(f"Saved id index -> {config.EMBEDDINGS_INDEX_PATH}")
    return embeddings, ids


if __name__ == "__main__":
    extract_embeddings()
