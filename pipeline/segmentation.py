"""
pipeline/segmentation.py
────────────────────────
Leaf mask and stress-region mask extraction.
Uses HSV thresholding with Otsu fallback for the leaf,
and multi-cue HSV + LAB thresholding for stress regions.
"""

import cv2
import numpy as np


# ──────────────────────────────────────────────
# Low-level morphology helpers
# ──────────────────────────────────────────────

def _clean_mask(mask: np.ndarray,
                kernel_size: int = 5,
                open_iter: int = 1,
                close_iter: int = 1) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    m = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=open_iter)
    m = cv2.morphologyEx(m,    cv2.MORPH_CLOSE, k, iterations=close_iter)
    return m


def _keep_largest(mask: np.ndarray) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = np.zeros_like(mask)
    out[labels == largest] = 255
    return out


def _remove_small(mask: np.ndarray, min_area: int = 80) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


# ──────────────────────────────────────────────
# Leaf segmentation
# ──────────────────────────────────────────────

def segment_leaf(image_rgb: np.ndarray) -> np.ndarray:
    """
    Returns a binary uint8 mask (255 = leaf, 0 = background).
    Primary method: HSV green thresholding.
    Fallback: Otsu on grayscale when green coverage is too low.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)

    # broad green range
    lower = np.array([22, 35, 35])
    upper = np.array([100, 255, 255])
    raw = cv2.inRange(hsv, lower, upper)
    cleaned = _clean_mask(raw, kernel_size=7, open_iter=1, close_iter=3)
    leaf_mask = _keep_largest(cleaned)

    # Otsu fallback when green coverage < 5 %
    h, w = image_rgb.shape[:2]
    if np.count_nonzero(leaf_mask) < 0.05 * h * w:
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        leaf_mask = _keep_largest(otsu)

    # Fill internal holes
    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(leaf_mask)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(filled, [largest], -1, 255, thickness=cv2.FILLED)
    return filled


# ──────────────────────────────────────────────
# Stress / disease region segmentation
# ──────────────────────────────────────────────

def segment_stress(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> dict:
    """
    Returns a dict:
        stress_mask  : uint8 binary mask of stressed pixels
        severity_pct : float percentage of leaf that is stressed
    Combines HSV (yellow/brown/dark) and LAB (a* channel) cues.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    _, a_ch, _ = cv2.split(lab)

    # Yellow / orange / brown lesions  (hue 8–35, decent sat & val)
    yellow_brown = (
        ((h >= 8)  & (h <= 35))  & (s >= 55) & (v >= 50)
    ).astype(np.uint8) * 255

    # Dark necrotic spots (low val, low sat)
    dark_necrotic = (
        (v < 60) & (s < 80) & (v > 15)
    ).astype(np.uint8) * 255

    # LAB a* > 135 → reddish / brown
    lab_stress = (a_ch.astype(np.int32) > 135).astype(np.uint8) * 255

    # Union and restrict to leaf area
    combined = cv2.bitwise_or(yellow_brown, dark_necrotic)
    combined = cv2.bitwise_or(combined, lab_stress)
    combined = cv2.bitwise_and(combined, leaf_mask)

    # Morphological cleanup
    stress_mask = _clean_mask(combined, kernel_size=3, open_iter=1, close_iter=2)
    stress_mask = _remove_small(stress_mask, min_area=80)

    leaf_px   = int(np.count_nonzero(leaf_mask))
    stress_px = int(np.count_nonzero(stress_mask))
    sev_pct   = (stress_px / leaf_px * 100.0) if leaf_px > 0 else 0.0

    return {"stress_mask": stress_mask, "severity_pct": sev_pct}


# ──────────────────────────────────────────────
# Edge map (used as aux feature)
# ──────────────────────────────────────────────

def detect_stress_edges(image_rgb: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    gray    = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges   = cv2.Canny(blurred, threshold1=30, threshold2=100)
    edges   = cv2.bitwise_and(edges, leaf_mask)
    return _remove_small(edges, min_area=30)


# ──────────────────────────────────────────────
# Overlay helper (for display)
# ──────────────────────────────────────────────

def highlight_stress(image_rgb: np.ndarray, stress_mask: np.ndarray,
                     color=(255, 60, 60), alpha: float = 0.35) -> np.ndarray:
    overlay = image_rgb.copy()
    overlay[stress_mask > 0] = color
    return cv2.addWeighted(image_rgb, 1 - alpha, overlay, alpha, 0)
