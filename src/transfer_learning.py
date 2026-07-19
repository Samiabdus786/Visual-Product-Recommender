"""
transfer_learning.py

Step 5 of the methodology: fine-tune the last layers of the pretrained
backbone on the subset dataset using a classification objective
(predict articleType). Most layers stay frozen; only the last N layers
of the backbone + a new classification head are trained. After training,
we again strip the head to produce FINE-TUNED embeddings, which serve as
the "transfer learning" baseline to compare against the raw pretrained
baseline and the Siamese network.

Usage:
    python src/transfer_learning.py
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocessing import preprocess_image, augment_image, load_image, normalize_imagenet  # noqa: E402


def build_classifier_backbone(num_classes, backbone_name=None):
    name = (backbone_name or config.BACKBONE).lower()
    input_shape = config.IMG_SIZE + (3,)

    if name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone '{name}'")

    # Freeze all layers, then unfreeze the last N
    base.trainable = True
    for layer in base.layers[:-config.FT_UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

    x = GlobalAveragePooling2D(name="gap")(base.output)
    embedding_layer = x  # this is what we'll reuse for embeddings later
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax", name="classifier")(x)

    full_model = Model(inputs=base.input, outputs=outputs, name="finetuned_classifier")
    embedding_model = Model(inputs=base.input, outputs=embedding_layer, name="finetuned_embedding")
    return full_model, embedding_model


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, paths, labels, batch_size, augment=False):
        self.paths = paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.paths[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs = []
        for p in batch_paths:
            img = load_image(p)
            if self.augment:
                img = augment_image(img)
            imgs.append(normalize_imagenet(img))
        return np.stack(imgs), np.array(batch_labels)


def run_transfer_learning():
    df = pd.read_csv(config.SUBSET_METADATA_CSV)

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["articleType"])
    num_classes = len(le.classes_)

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=config.RANDOM_SEED, stratify=df["label"]
    )

    train_gen = DataGenerator(train_df["image_path"].tolist(), train_df["label"].tolist(),
                               config.FT_BATCH_SIZE, augment=True)
    val_gen = DataGenerator(val_df["image_path"].tolist(), val_df["label"].tolist(),
                             config.FT_BATCH_SIZE, augment=False)

    full_model, embedding_model = build_classifier_backbone(num_classes)
    full_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.FT_LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print(full_model.summary())

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ]

    full_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.FT_EPOCHS,
        callbacks=callbacks,
    )

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    embedding_model.save(config.FINETUNED_MODEL_PATH)
    print(f"Saved fine-tuned embedding model -> {config.FINETUNED_MODEL_PATH}")

    # Extract fine-tuned embeddings for the whole subset
    extract_finetuned_embeddings(embedding_model, df)


def extract_finetuned_embeddings(embedding_model, df, batch_size=32):
    embeddings = []
    ids = []
    paths = df["image_path"].tolist()
    id_list = df["id"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting fine-tuned embeddings"):
        batch_paths = paths[i:i + batch_size]
        batch_ids = id_list[i:i + batch_size]
        batch_imgs = np.stack([preprocess_image(p) for p in batch_paths], axis=0)
        batch_embeds = embedding_model.predict(batch_imgs, verbose=0)
        embeddings.append(batch_embeds)
        ids.extend(batch_ids)

    embeddings = np.concatenate(embeddings, axis=0)
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    np.save(config.FINETUNED_EMBEDDINGS_PATH, embeddings)
    print(f"Saved fine-tuned embeddings {embeddings.shape} -> {config.FINETUNED_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    run_transfer_learning()
