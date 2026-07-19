"""
train_siamese.py

Trains the Siamese network with triplet loss on triplets generated from
the subset (same-category positive, different-category negative), then
saves the shared embedding network and extracts Siamese embeddings for
the entire subset.

Usage:
    python src/train_siamese.py
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.siamese_dataset import build_triplets, TripletDataGenerator  # noqa: E402
from src.siamese_model import build_embedding_network, build_siamese_training_model, triplet_loss  # noqa: E402
from src.preprocessing import preprocess_image  # noqa: E402


def train():
    df = pd.read_csv(config.SUBSET_METADATA_CSV)

    print("Building triplets...")
    triplets = build_triplets(df, n_triplets=config.TRIPLETS_PER_EPOCH, seed=config.RANDOM_SEED)
    print(f"  Generated {len(triplets)} triplets")

    train_gen = TripletDataGenerator(triplets, batch_size=config.SIAMESE_BATCH_SIZE, augment=True)

    embedding_network = build_embedding_network()
    training_model, embedding_network = build_siamese_training_model(embedding_network)

    training_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.SIAMESE_LEARNING_RATE),
        loss=triplet_loss(config.TRIPLET_MARGIN),
    )

    print(training_model.summary())

    # --- Crash resilience: resume from checkpoint if one exists ---
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    ckpt_dir = os.path.join(config.MODELS_DIR, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "siamese_ckpt.weights.h5")
    initial_epoch = 0

    if os.path.exists(ckpt_path):
        print(f"Found existing checkpoint at {ckpt_path} — resuming instead of retraining from scratch.")
        training_model.load_weights(ckpt_path)
        log_path = os.path.join(ckpt_dir, "siamese_epoch.txt")
        if os.path.exists(log_path):
            with open(log_path) as f:
                initial_epoch = int(f.read().strip())
            print(f"Resuming from epoch {initial_epoch}/{config.SIAMESE_EPOCHS}")

    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            with open(os.path.join(ckpt_dir, "siamese_epoch.txt"), "w") as f:
                f.write(str(epoch + 1))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.5, patience=2),
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, save_weights_only=True, save_freq="epoch", verbose=0
        ),
        EpochLogger(),
    ]

    training_model.fit(
        train_gen,
        epochs=config.SIAMESE_EPOCHS,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
    )

    # Training finished successfully — clean up checkpoint so future runs start fresh
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
    log_path = os.path.join(ckpt_dir, "siamese_epoch.txt")
    if os.path.exists(log_path):
        os.remove(log_path)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    embedding_network.save(config.SIAMESE_MODEL_PATH)
    print(f"Saved Siamese embedding network -> {config.SIAMESE_MODEL_PATH}")

    extract_siamese_embeddings(embedding_network, df)


def extract_siamese_embeddings(embedding_network, df, batch_size=32):
    embeddings = []
    ids = []
    paths = df["image_path"].tolist()
    id_list = df["id"].tolist()

    for i in tqdm(range(0, len(paths), batch_size), desc="Extracting Siamese embeddings"):
        batch_paths = paths[i:i + batch_size]
        batch_ids = id_list[i:i + batch_size]
        batch_imgs = np.stack([preprocess_image(p) for p in batch_paths], axis=0)
        batch_embeds = embedding_network.predict(batch_imgs, verbose=0)
        embeddings.append(batch_embeds)
        ids.extend(batch_ids)

    embeddings = np.concatenate(embeddings, axis=0)
    os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
    np.save(config.SIAMESE_EMBEDDINGS_PATH, embeddings)
    print(f"Saved Siamese embeddings {embeddings.shape} -> {config.SIAMESE_EMBEDDINGS_PATH}")


if __name__ == "__main__":
    train()