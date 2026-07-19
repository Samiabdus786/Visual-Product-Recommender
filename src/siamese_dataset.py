"""
siamese_dataset.py

Builds (Anchor, Positive, Negative) triplets from the subset metadata:
  - Anchor & Positive: same articleType (category) -> should be pulled together
  - Negative: different articleType -> should be pushed apart

A tf.keras.utils.Sequence yields batches of preprocessed triplet images.
"""

import os
import sys
import random

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.preprocessing import load_image, augment_image, normalize_imagenet  # noqa: E402


def build_triplets(df, n_triplets, seed=None):
    """
    df must contain columns: id, image_path, articleType
    Returns a list of (anchor_path, positive_path, negative_path) tuples.
    """
    rng = random.Random(seed)
    by_category = df.groupby("articleType")["image_path"].apply(list).to_dict()
    categories = list(by_category.keys())

    triplets = []
    for _ in range(n_triplets):
        pos_cat = rng.choice(categories)
        neg_cat = rng.choice([c for c in categories if c != pos_cat])

        pos_pool = by_category[pos_cat]
        neg_pool = by_category[neg_cat]

        if len(pos_pool) < 2:
            continue

        anchor_path, positive_path = rng.sample(pos_pool, 2)
        negative_path = rng.choice(neg_pool)

        triplets.append((anchor_path, positive_path, negative_path))

    return triplets


class TripletDataGenerator(tf.keras.utils.Sequence):
    """
    Yields batches shaped as:
      ([anchor_batch, positive_batch, negative_batch], dummy_labels)
    dummy_labels are zeros since the triplet loss is computed from the
    embeddings themselves (not from a label), but Keras' fit() API expects
    a `y` argument.
    """

    def __init__(self, triplets, batch_size=32, augment=True):
        self.triplets = triplets
        self.batch_size = batch_size
        self.augment = augment

    def __len__(self):
        return int(np.ceil(len(self.triplets) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.triplets[idx * self.batch_size:(idx + 1) * self.batch_size]
        anchors, positives, negatives = [], [], []

        for a_path, p_path, n_path in batch:
            a_img = load_image(a_path)
            p_img = load_image(p_path)
            n_img = load_image(n_path)

            if self.augment:
                a_img = augment_image(a_img)
                p_img = augment_image(p_img)
                n_img = augment_image(n_img)

            anchors.append(normalize_imagenet(a_img))
            positives.append(normalize_imagenet(p_img))
            negatives.append(normalize_imagenet(n_img))

        anchors = np.stack(anchors)
        positives = np.stack(positives)
        negatives = np.stack(negatives)
        dummy_labels = np.zeros((len(batch),))

        return (anchors, positives, negatives), dummy_labels

    def on_epoch_end(self):
        random.shuffle(self.triplets)
