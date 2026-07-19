"""
siamese_model.py

Defines:
  - The shared embedding network (pretrained CNN backbone + projection head
    onto an L2-normalized EMBEDDING_DIM-dim space).
  - A Siamese wrapper that takes (anchor, positive, negative) inputs,
    computes their embeddings with the SHARED network, and outputs the
    triplet loss.

Core idea: same weights are applied to all three inputs ("Siamese" =
identical twins), and training pulls anchor/positive embeddings together
while pushing anchor/negative embeddings apart -- this is what makes the
learned embedding space "semantically" organized by visual similarity
rather than raw pixel similarity.
"""

import os
import sys

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50, EfficientNetB0

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def build_embedding_network(backbone_name=None, embedding_dim=None, unfreeze_last_n=None):
    """
    Pretrained CNN backbone (transfer learning) -> GAP -> Dense projection
    -> L2 normalize. This single network is reused (shared weights) for
    anchor, positive and negative inputs.
    """
    name = (backbone_name or config.BACKBONE).lower()
    embedding_dim = embedding_dim or config.EMBEDDING_DIM
    unfreeze_last_n = unfreeze_last_n if unfreeze_last_n is not None else config.FT_UNFREEZE_LAST_N_LAYERS
    input_shape = config.IMG_SIZE + (3,)

    if name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    elif name == "efficientnet":
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone '{name}'")

    # Freeze most of the backbone; fine-tune only the last N layers
    base.trainable = True
    for layer in base.layers[:-unfreeze_last_n]:
        layer.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(embedding_dim)(x)
    outputs = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="l2_normalize")(x)

    model = Model(inputs=base.input, outputs=outputs, name=f"{name}_embedding_network")
    return model


def triplet_loss(margin=None):
    margin = margin if margin is not None else config.TRIPLET_MARGIN

    def loss_fn(y_true, y_pred):
        """
        y_pred is expected to be a concatenation of [anchor_emb, positive_emb,
        negative_emb] along axis=-1, each of size embedding_dim (this is how
        SiameseTripletModel packages its output below). y_true is unused
        (dummy labels) but required by the Keras fit() signature.
        """
        emb_dim = y_pred.shape[-1] // 3
        anchor = y_pred[:, :emb_dim]
        positive = y_pred[:, emb_dim:2 * emb_dim]
        negative = y_pred[:, 2 * emb_dim:]

        pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
        neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)

        basic_loss = pos_dist - neg_dist + margin
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
        return loss

    return loss_fn


def build_siamese_training_model(embedding_network=None):
    """
    Wraps the shared embedding_network into a 3-input training model whose
    output is the concatenation of the three embeddings, so that
    `triplet_loss` above can unpack and compute the triplet objective.
    """
    embedding_network = embedding_network or build_embedding_network()
    input_shape = config.IMG_SIZE + (3,)

    anchor_input = layers.Input(shape=input_shape, name="anchor")
    positive_input = layers.Input(shape=input_shape, name="positive")
    negative_input = layers.Input(shape=input_shape, name="negative")

    anchor_emb = embedding_network(anchor_input)
    positive_emb = embedding_network(positive_input)
    negative_emb = embedding_network(negative_input)

    concatenated = layers.Concatenate(axis=-1)([anchor_emb, positive_emb, negative_emb])

    training_model = Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=concatenated,
        name="siamese_triplet_training_model",
    )
    return training_model, embedding_network
