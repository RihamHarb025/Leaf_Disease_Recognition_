"""
pipeline/features.py
─────────────────────
Full feature vector extraction from preprocessed image + masks.
Features cover: morphology, colour, texture (LBP + GLCM), and edge density.
Returns a flat dict and a numpy array in a fixed canonical order.
"""

import cv2
import numpy as np
from typing import Dict, Tuple

try:
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


# ──────────────────────────────────────────────
# Canonical feature order (used for the vector)
# ──────────────────────────────────────────────
FEATURE_NAMES = [
    # morphology
    "leaf_area", "stress_area", "severity_pct",
    "num_regions", "largest_region_area",
    "stress_perimeter", "compactness", "aspect_ratio",
    "extent", "solidity",
    # colour
    "color_diff",
    "healthy_mean_r", "healthy_mean_g", "healthy_mean_b",
    "stress_mean_r",  "stress_mean_g",  "stress_mean_b",
    "stress_mean_h",  "stress_mean_s",  "stress_mean_v",
    # texture
    "gray_stress_mean", "gray_stress_std",
    "lbp_mean",  "lbp_std",
    "glcm_contrast", "glcm_homogeneity",
    # edges
    "local_variance", "edge_density",
]


# ──────────────────────────────────────────────
# Sub-extractors
# ──────────────────────────────────────────────

def _morphology(leaf_mask: np.ndarray, stress_mask: np.ndarray) -> dict:
    leaf_area   = int(np.count_nonzero(leaf_mask))
    stress_area = int(np.count_nonzero(stress_mask))
    sev_pct     = (stress_area / leaf_area * 100.0) if leaf_area > 0 else 0.0

    n, labels, stats, _ = cv2.connectedComponentsWithStats(stress_mask, connectivity=8)
    num_regions = max(0, n - 1)
    largest_area = int(np.max(stats[1:, cv2.CC_STAT_AREA])) if num_regions > 0 else 0

    # perimeter and shape descriptors from the largest stress contour
    contours, _ = cv2.findContours(stress_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter   = 0.0
    compactness = 0.0
    aspect_ratio = 1.0
    extent      = 0.0
    solidity    = 0.0

    if contours:
        largest_c = max(contours, key=cv2.contourArea)
        area_c    = cv2.contourArea(largest_c)
        perimeter = float(cv2.arcLength(largest_c, True))

        if area_c > 0 and perimeter > 0:
            compactness = float((perimeter ** 2) / (4 * np.pi * area_c))

        x, y, w, h = cv2.boundingRect(largest_c)
        aspect_ratio = float(w / h) if h > 0 else 1.0
        rect_area = w * h
        extent  = float(area_c / rect_area) if rect_area > 0 else 0.0

        hull = cv2.convexHull(largest_c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_c / hull_area) if hull_area > 0 else 0.0

    return dict(
        leaf_area=leaf_area, stress_area=stress_area, severity_pct=sev_pct,
        num_regions=num_regions, largest_region_area=largest_area,
        stress_perimeter=perimeter, compactness=compactness,
        aspect_ratio=aspect_ratio, extent=extent, solidity=solidity,
    )


def _colour(image_rgb: np.ndarray,
            leaf_mask: np.ndarray,
            stress_mask: np.ndarray) -> dict:
    healthy_mask = (leaf_mask > 0) & (stress_mask == 0)
    stress_bool  = stress_mask > 0

    def _mean_rgb(m):
        if np.any(m):
            return image_rgb[m].mean(axis=0).tolist()
        return [0.0, 0.0, 0.0]

    hr, hg, hb = _mean_rgb(healthy_mask)
    sr, sg, sb = _mean_rgb(stress_bool)

    color_diff = float(np.linalg.norm(
        np.array([sr, sg, sb]) - np.array([hr, hg, hb])
    ))

    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    if np.any(stress_bool):
        sh, ss, sv = hsv[stress_bool].mean(axis=0).tolist()
    else:
        sh, ss, sv = 0.0, 0.0, 0.0

    return dict(
        color_diff=color_diff,
        healthy_mean_r=hr, healthy_mean_g=hg, healthy_mean_b=hb,
        stress_mean_r=sr,  stress_mean_g=sg,  stress_mean_b=sb,
        stress_mean_h=sh,  stress_mean_s=ss,  stress_mean_v=sv,
    )


def _texture(image_rgb: np.ndarray, stress_mask: np.ndarray) -> dict:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    stress_pixels = gray[stress_mask > 0]
    if len(stress_pixels) > 0:
        gm = float(stress_pixels.mean())
        gs = float(stress_pixels.std())
    else:
        gm, gs = 0.0, 0.0

    lbp_mean, lbp_std = 0.0, 0.0
    glcm_contrast, glcm_homogeneity = 0.0, 0.0

    if _SKIMAGE:
        try:
            lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
            if np.any(stress_mask > 0):
                lv = lbp[stress_mask > 0]
                lbp_mean = float(lv.mean())
                lbp_std  = float(lv.std())

            gray_q = (gray // 16).astype(np.uint8)   # 16-level quantisation
            glcm   = graycomatrix(gray_q, distances=[1], angles=[0],
                                  levels=16, symmetric=True, normed=True)
            glcm_contrast    = float(graycoprops(glcm, "contrast")[0, 0])
            glcm_homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
        except Exception:
            pass

    return dict(
        gray_stress_mean=gm, gray_stress_std=gs,
        lbp_mean=lbp_mean, lbp_std=lbp_std,
        glcm_contrast=glcm_contrast, glcm_homogeneity=glcm_homogeneity,
    )


def _edges(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> dict:
    gray   = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    lap    = cv2.Laplacian(gray, cv2.CV_64F)
    lv     = float(np.var(lap))

    edges  = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 100)
    leaf_px = int(np.count_nonzero(leaf_mask))
    edge_density = float(np.count_nonzero(
        cv2.bitwise_and(edges, leaf_mask)
    ) / leaf_px) if leaf_px > 0 else 0.0

    return dict(local_variance=lv, edge_density=edge_density)


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def extract_features(image_rgb: np.ndarray,
                     leaf_mask: np.ndarray,
                     stress_mask: np.ndarray) -> Dict[str, float]:
    """Return the full feature dict (key → scalar)."""
    feats = {}
    feats.update(_morphology(leaf_mask, stress_mask))
    feats.update(_colour(image_rgb, leaf_mask, stress_mask))
    feats.update(_texture(image_rgb, stress_mask))
    feats.update(_edges(image_rgb, leaf_mask))
    return feats


def feature_vector(feats: Dict[str, float]) -> np.ndarray:
    """Convert feature dict → fixed-length numpy array in FEATURE_NAMES order."""
    return np.array([feats.get(k, 0.0) for k in FEATURE_NAMES], dtype=np.float64)
