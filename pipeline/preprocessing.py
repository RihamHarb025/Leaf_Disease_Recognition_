"""
pipeline/preprocessing.py
─────────────────────────
Image quality checks + preprocessing transformations.
Called first in the pipeline before segmentation.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List


# ──────────────────────────────────────────────
# Quality report
# ──────────────────────────────────────────────

@dataclass
class QualityReport:
    brightness: float = 0.0
    contrast: float = 0.0
    blur_score: float = 0.0
    blur_label: str = "Unknown"
    high_freq_energy: float = 0.0
    shadow_ratio: float = 0.0
    overexposure_ratio: float = 0.0
    passed: bool = False
    issues: List[str] = field(default_factory=list)


# Thresholds
_MIN_BRIGHTNESS    = 40.0
_MAX_BRIGHTNESS    = 220.0
_MIN_CONTRAST      = 15.0
_MIN_BLUR          = 60.0          # Laplacian variance
_MIN_HF_ENERGY     = 2.0


def _blur_label(score: float) -> str:
    if score >= 300:  return "Very Sharp"
    if score >= 150:  return "Sharp"
    if score >= 60:   return "Acceptable"
    return "Blurry"


def check_quality(image_rgb: np.ndarray) -> QualityReport:
    """Run all quality checks and return a QualityReport."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    brightness       = float(np.mean(gray))
    contrast         = float(np.std(gray))
    blur_score       = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_lbl         = _blur_label(blur_score)

    blurred          = cv2.GaussianBlur(gray.astype(np.float32), (9, 9), 0)
    hf_energy        = float(np.mean(np.abs(gray.astype(np.float32) - blurred)))

    shadow_ratio     = float(np.mean(gray < 30))
    overexposure_ratio = float(np.mean(gray > 240))

    issues = []
    if brightness < _MIN_BRIGHTNESS:
        issues.append(f"Image too dark (brightness={brightness:.1f}, min={_MIN_BRIGHTNESS})")
    if brightness > _MAX_BRIGHTNESS:
        issues.append(f"Image overexposed (brightness={brightness:.1f}, max={_MAX_BRIGHTNESS})")
    if contrast < _MIN_CONTRAST:
        issues.append(f"Low contrast (std={contrast:.1f}, min={_MIN_CONTRAST})")
    if blur_score < _MIN_BLUR:
        issues.append(f"Image too blurry (Laplacian var={blur_score:.1f}, min={_MIN_BLUR})")
    if hf_energy < _MIN_HF_ENERGY:
        issues.append(f"Low high-frequency energy ({hf_energy:.2f})")

    return QualityReport(
        brightness=brightness,
        contrast=contrast,
        blur_score=blur_score,
        blur_label=blur_lbl,
        high_freq_energy=hf_energy,
        shadow_ratio=shadow_ratio,
        overexposure_ratio=overexposure_ratio,
        passed=len(issues) == 0,
        issues=issues,
    )


# ──────────────────────────────────────────────
# Preprocessing transforms
# ──────────────────────────────────────────────

def resize(image: np.ndarray, size: tuple = (512, 512)) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def denoise(image: np.ndarray) -> np.ndarray:
    """Light bilateral filter — preserves edges."""
    return cv2.bilateralFilter(image, d=7, sigmaColor=50, sigmaSpace=50)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """CLAHE on the L channel in LAB space."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = cv2.merge((clahe.apply(l), a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)


def preprocess(image_rgb: np.ndarray) -> np.ndarray:
    """Full preprocessing chain: resize → denoise → CLAHE contrast."""
    img = resize(image_rgb)
    img = denoise(img)
    img = enhance_contrast(img)
    return img
