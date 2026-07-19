"""
preprocessing.py

Image loading & preprocessing utilities:
  - resize to 224x224
  - normalize using ImageNet mean/std
  - simple augmentation for Siamese/fine-tuning training
"""

import os
import sys

import numpy as np
from PIL import Image
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def load_image(path, target_size=None):
    """Load an image from disk as an RGB numpy array (uint8)."""
    target_size = target_size or config.IMG_SIZE
    img = Image.open(path).convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    return np.array(img)


def normalize_imagenet(img_array):
    """
    img_array: float32 array in [0, 255], shape (H, W, 3)
    Returns a normalized array using ImageNet mean/std, ready for the CNN backbone.
    """
    img = img_array.astype("float32") / 255.0
    mean = np.array(config.IMAGENET_MEAN, dtype="float32")
    std = np.array(config.IMAGENET_STD, dtype="float32")
    img = (img - mean) / std
    return img


def preprocess_image(path, target_size=None):
    """Full pipeline: load -> resize -> normalize. Returns float32 (H, W, 3)."""
    img = load_image(path, target_size)
    return normalize_imagenet(img)


def preprocess_pil(pil_image, target_size=None):
    """Same pipeline but for an already-loaded PIL image (used by the Streamlit app)."""
    target_size = target_size or config.IMG_SIZE
    img = pil_image.convert("RGB").resize(target_size, Image.BILINEAR)
    img = np.array(img)
    return normalize_imagenet(img)


def augment_image(img_array):
    """
    Light augmentation used only during Siamese / fine-tuning training
    (random horizontal flip + small brightness jitter). Operates on a
    uint8 RGB array BEFORE normalization.
    """
    img = img_array.copy()

    # Random horizontal flip
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)

    # Random brightness jitter
    if np.random.rand() < 0.5:
        factor = np.random.uniform(0.8, 1.2)
        img = np.clip(img.astype("float32") * factor, 0, 255).astype("uint8")

    return img


def batch_preprocess(paths, target_size=None, augment=False):
    """Load & preprocess a list of image paths into a single batch array."""
    target_size = target_size or config.IMG_SIZE
    batch = []
    for p in paths:
        img = load_image(p, target_size)
        if augment:
            img = augment_image(img)
        batch.append(normalize_imagenet(img))
    return np.stack(batch, axis=0)
