"""
pipeline/dataset.py
────────────────────
Dataset loader for PlantVillage / plant_leaves datasets.

Folder layout expected (mirrors PlantVillage naming):
  dataset_imgs/
    ├── Tomato___healthy/
    ├── Tomato___Early_blight/
    ├── Pepper__bell___Bacterial_spot/
    └── ...

Label mapping: each folder name is mapped to one of the 5 canonical
stress classes via keyword matching.

Usage
─────
  from pipeline.dataset import load_dataset, LABEL_MAP
  X, y, paths = load_dataset("dataset_imgs/")

  from pipeline.classifier import train_knn, loo_evaluate
  info = train_knn(X, y)
  eval_result = loo_evaluate(X, y)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple

from pipeline.preprocessing import preprocess
from pipeline.segmentation import segment_leaf, segment_stress
from pipeline.features import extract_features, feature_vector


# ──────────────────────────────────────────────
# Label mapping
# ──────────────────────────────────────────────

# Keywords in folder names → canonical class
LABEL_MAP_RULES = [
    (["healthy"],                                         "healthy"),
    (["yellow", "chloro", "mosaic", "miner"],            "chlorosis_yellowing"),
    (["blight", "scorch", "burn", "dry", "leaf_curl"],   "dryness_dehydration"),
    (["spot", "necrotic", "rust", "septoria",
      "bacterial", "cercospora", "target"],               "spot_necrotic"),
]


def folder_to_label(folder_name: str) -> str:
    name_lower = folder_name.lower()
    for keywords, label in LABEL_MAP_RULES:
        if any(kw in name_lower for kw in keywords):
            return label
    return "unknown_abnormality"


IMG_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".bmp"}


# ──────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────

def load_dataset(dataset_root: str,
                 max_per_class: int = 200,
                 verbose: bool = True) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Walk dataset_root, extract feature vectors, return (X, y, paths).

    Parameters
    ----------
    dataset_root    : path to the folder containing per-class sub-folders
    max_per_class   : cap images per class (avoids memory issues on large datasets)
    verbose         : print progress

    Returns
    -------
    X     : (n, n_features) float64 array
    y     : list of class-name strings
    paths : list of image file paths (same order as X, y)
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset not found: {root}")

    X_list, y_list, p_list = [], [], []

    subdirs = sorted([d for d in root.iterdir() if d.is_dir()])
    if not subdirs:
        # flat folder — treat all images as a single unlabelled batch
        subdirs = [root]

    for subdir in subdirs:
        label = folder_to_label(subdir.name)
        imgs  = [f for f in subdir.iterdir()
                 if f.suffix.lower() in IMG_EXTS][:max_per_class]

        if verbose:
            print(f"  {subdir.name[:40]:<40} → {label:<25} ({len(imgs)} imgs)")

        for img_path in imgs:
            try:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb = preprocess(rgb)

                leaf_mask = segment_leaf(rgb)
                stress_info = segment_stress(rgb, leaf_mask)
                stress_mask = stress_info["stress_mask"]

                feats = extract_features(rgb, leaf_mask, stress_mask)
                fvec  = feature_vector(feats)

                X_list.append(fvec)
                y_list.append(label)
                p_list.append(str(img_path))
            except Exception as e:
                if verbose:
                    print(f"    [skip] {img_path.name}: {e}")

    if not X_list:
        raise ValueError("No images could be processed. Check dataset path and format.")

    X = np.vstack(X_list)
    return X, y_list, p_list


# ──────────────────────────────────────────────
# Convenience: train from folder
# ──────────────────────────────────────────────

def train_from_folder(dataset_root: str,
                      max_per_class: int = 200,
                      run_loo: bool = True,
                      verbose: bool = True) -> dict:
    """
    Full train pipeline:
      1. Load dataset
      2. Train k-NN
      3. (Optionally) run LOO cross-validation
    Returns combined result dict.
    """
    from pipeline.classifier import train_knn, loo_evaluate

    if verbose:
        print(f"\n=== LeafGuard Dataset Loader ===")
        print(f"Root: {dataset_root}\n")

    X, y, paths = load_dataset(dataset_root, max_per_class=max_per_class, verbose=verbose)

    if verbose:
        from collections import Counter
        dist = Counter(y)
        print(f"\nClass distribution: {dict(dist)}")
        print(f"Total samples: {len(y)}\n")

    train_info = train_knn(X, y)
    if verbose:
        print(f"Training accuracy: {train_info['train_accuracy']*100:.1f}%")

    result = {"train": train_info, "n_samples": len(y)}

    if run_loo and len(y) >= 4:
        if verbose:
            print("\nRunning Leave-One-Out cross-validation…")
        loo_info = loo_evaluate(X, y)
        result["loo"] = loo_info
        if verbose:
            print(f"LOO accuracy: {loo_info['loo_accuracy']*100:.1f}%")
            print("\nConfusion Matrix:")
            labels = loo_info["labels"]
            cm     = loo_info["confusion_matrix"]
            header = f"{'':>22}" + "".join(f"{l[:8]:>10}" for l in labels)
            print(header)
            for row_label, row in zip(labels, cm):
                print(f"{row_label[:22]:>22}" + "".join(f"{v:>10}" for v in row))

    return result
