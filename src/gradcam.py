"""
gradcam.py

Similarity-based Grad-CAM: explains *why* a query image matched a
particular retrieved product, by highlighting which regions of the query
image contributed most to the similarity score.

Standard Grad-CAM computes gradients of a class score w.r.t. the last
conv layer. Our models don't have a classification head (they output raw
embeddings), so instead we compute gradients of the SIMILARITY between
the query embedding and a specific retrieved item's embedding. This shows
which pixels pushed the two embeddings closer together in the model's
learned space — directly answering "why did the model think these look
alike?"
"""

import numpy as np
import tensorflow as tf
import cv2


def find_last_conv_layer(model):
    """Walk backwards through the model and return the name of the last
    layer whose output is 4D (spatial feature map) — works for any CNN
    backbone (ResNet, EfficientNet, etc.) without hardcoding layer names."""
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("Could not find a 4D (conv-like) layer in this model.")


def compute_similarity_gradcam(model, preprocessed_image, target_embedding, last_conv_layer_name=None):
    """
    preprocessed_image: (224, 224, 3) float32 array, already resized/normalized
                         (i.e. output of preprocessing.preprocess_pil / preprocess_image)
    target_embedding:   (embedding_dim,) float32 array — the retrieved product's
                         precomputed embedding we're explaining the match against
    Returns a (224, 224) float32 heatmap in [0, 1], resized to the input image size.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    img_batch = np.expand_dims(preprocessed_image, axis=0)
    target_vec = tf.constant(target_embedding.reshape(1, -1).astype("float32"))

    with tf.GradientTape() as tape:
        conv_output, embedding = grad_model(img_batch)
        # Similarity = dot product between query embedding and target embedding.
        # For L2-normalized embeddings (Siamese) this equals cosine similarity;
        # for raw GAP embeddings (Baseline/Fine-tuned) it's an unnormalized dot
        # product, which still correctly identifies which regions drive the score.
        similarity = tf.reduce_sum(embedding * target_vec, axis=1)

    grads = tape.gradient(similarity, conv_output)
    if grads is None:
        raise RuntimeError("Gradient computation failed — check that the layer name is correct.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = tf.reduce_sum(conv_output * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_resized = cv2.resize(heatmap, (preprocessed_image.shape[1], preprocessed_image.shape[0]))
    return heatmap_resized


def overlay_heatmap(original_rgb_uint8, heatmap, alpha=0.45):
    """
    original_rgb_uint8: (H, W, 3) uint8 RGB image (the actual displayed query image,
                         NOT the ImageNet-normalized array used for the model)
    heatmap:            (H, W) float32 in [0, 1], from compute_similarity_gradcam
    Returns a (H, W, 3) uint8 RGB image with a jet-colormap heatmap overlaid.
    """
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = (heatmap_color.astype("float32") * alpha +
               original_rgb_uint8.astype("float32") * (1 - alpha))
    return np.clip(overlay, 0, 255).astype("uint8")
