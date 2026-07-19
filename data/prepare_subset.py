"""
prepare_subset.py

Builds the curated training/evaluation subset from the raw
'Fashion Product Images' Kaggle dataset.

Steps:
  1. Load styles.csv
  2. Filter to the categories defined in config.CATEGORIES
  3. Randomly sample ~IMAGES_PER_CATEGORY images per category
  4. Verify the corresponding image file exists
  5. Copy selected images into data/subset/images/
  6. Write data/subset/subset_metadata.csv with columns:
       id, gender, masterCategory, subCategory, articleType,
       baseColour, season, usage, productDisplayName, image_path

Usage:
    python data/prepare_subset.py
"""

import os
import sys
import shutil
import random

import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402


def load_styles():
    if not os.path.exists(config.RAW_STYLES_CSV):
        raise FileNotFoundError(
            f"Could not find {config.RAW_STYLES_CSV}.\n"
            "Download the dataset from Kaggle "
            "(https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset), "
            "extract it, and point config.RAW_DATASET_DIR to the extracted folder."
        )
    # The Kaggle styles.csv occasionally has a malformed row or two -> skip bad lines.
    df = pd.read_csv(config.RAW_STYLES_CSV, on_bad_lines="skip")
    return df


def build_subset():
    random.seed(config.RANDOM_SEED)
    df = load_styles()

    print(f"Loaded {len(df)} rows from styles.csv")
    print(f"Target categories: {config.CATEGORIES}")

    os.makedirs(config.SUBSET_IMAGES_DIR, exist_ok=True)

    selected_rows = []
    for category in config.CATEGORIES:
        cat_df = df[df["articleType"] == category]
        if cat_df.empty:
            print(f"  [WARN] No rows found for category '{category}', skipping.")
            continue

        n = min(config.IMAGES_PER_CATEGORY, len(cat_df))
        sampled = cat_df.sample(n=n, random_state=config.RANDOM_SEED)
        print(f"  {category}: sampled {len(sampled)} / {len(cat_df)} available")
        selected_rows.append(sampled)

    if not selected_rows:
        raise RuntimeError("No categories matched — check config.CATEGORIES against your styles.csv values.")

    subset_df = pd.concat(selected_rows, ignore_index=True)

    # Verify image exists & copy it into the subset folder
    kept_records = []
    for _, row in tqdm(subset_df.iterrows(), total=len(subset_df), desc="Copying images"):
        img_id = row["id"]
        src_path = os.path.join(config.RAW_IMAGES_DIR, f"{img_id}.jpg")
        if not os.path.exists(src_path):
            continue
        dst_path = os.path.join(config.SUBSET_IMAGES_DIR, f"{img_id}.jpg")
        shutil.copyfile(src_path, dst_path)

        record = row.to_dict()
        record["image_path"] = dst_path
        kept_records.append(record)

    final_df = pd.DataFrame(kept_records)
    final_df.to_csv(config.SUBSET_METADATA_CSV, index=False)

    print("\nSubset build complete.")
    print(f"  Total images copied : {len(final_df)}")
    print(f"  Categories           : {final_df['articleType'].nunique()}")
    print(f"  Metadata written to  : {config.SUBSET_METADATA_CSV}")
    print(final_df["articleType"].value_counts())


if __name__ == "__main__":
    build_subset()
