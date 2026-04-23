"""
helpers.py
──────────
Single-file helpers used by app.py.
Covers: image I/O, session logging, model metadata, and all matplotlib charts.
"""

import cv2
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict


# ── paths ──────────────────────────────────────────────────────────────────────
MODEL_DIR = Path(__file__).parent / "models"
LOG_PATH  = Path(__file__).parent / "data" / "logs" / "diagnosis_log.csv"


# ══════════════════════════════════════════════════════════════════════════════
# IMAGE I/O
# ══════════════════════════════════════════════════════════════════════════════

def load_image_rgb(file_bytes: bytes) -> Optional[np.ndarray]:
    arr = np.frombuffer(file_bytes, np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


# ══════════════════════════════════════════════════════════════════════════════
# SESSION LOG
# ══════════════════════════════════════════════════════════════════════════════

def append_log(decision: dict, sensors: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "label":          decision["label"],
        "label_display":  decision["label_display"],
        "confidence":     round(decision["confidence"], 4),
        "severity_pct":   round(decision["severity_pct"], 2),
        "severity_class": decision["severity_class"],
        "risk_level":     decision["risk_level"],
        "recommendation": decision["recommendation"],
        **{f"sensor_{k}": v for k, v in sensors.items()},
        **{f"act_{k}": v   for k, v in decision["actuators"].items()},
    }
    df = pd.DataFrame([row])
    if LOG_PATH.exists():
        df.to_csv(LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)


def load_log(severity_filter=None, risk_filter=None) -> pd.DataFrame:
    if not LOG_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LOG_PATH)
    if severity_filter:
        df = df[df["severity_class"].isin(severity_filter)]
    if risk_filter:
        df = df[df["risk_level"].isin(risk_filter)]
    return df


def clear_log() -> None:
    if LOG_PATH.exists():
        LOG_PATH.unlink()


# ══════════════════════════════════════════════════════════════════════════════
# MODEL METADATA
# ══════════════════════════════════════════════════════════════════════════════

def model_is_trained() -> bool:
    return (MODEL_DIR / "knn_model.pkl").exists()


def model_summary_text() -> str:
    if not model_is_trained():
        return "No trained model — rule-based fallback active"
    meta_path = MODEL_DIR / "knn_meta.json"
    if not meta_path.exists():
        return "Model present (no metadata)"
    with open(meta_path) as f:
        m = json.load(f)
    return (f"k-NN  k={m.get('k','?')}  "
            f"n={m.get('n_samples','?')}  "
            f"train_acc={m.get('train_accuracy',0)*100:.1f}%")


def load_model_meta() -> dict:
    p = MODEL_DIR / "knn_meta.json"
    return json.load(open(p)) if p.exists() else {}


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def timestamp_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{ext}"


# ══════════════════════════════════════════════════════════════════════════════
# CHARTS  (all return matplotlib Figure — caller does st.pyplot(fig))
# ══════════════════════════════════════════════════════════════════════════════

# colour palette matching the app theme
_G1 = "#4a6b2a"   # dark green
_G2 = "#6b9a3a"   # mid green
_G3 = "#a8c878"   # light green
_BG = "#f4f0e6"   # page background
_WH = "#ffffff"

_CLASS_COLORS = {
    "healthy":             "#4caf50",
    "chlorosis_yellowing": "#ff9800",
    "dryness_dehydration": "#f59e0b",
    "spot_necrotic":       "#ef4444",
    "unknown_abnormality": "#9e9e9e",
}


def _base(w=9, h=4):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_WH)
    ax.set_facecolor(_BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#ccc")
    ax.tick_params(colors="#555")
    return fig, ax


def plot_severity_donut(severity_pct: float, label: str = "") -> plt.Figure:
    healthy = max(0.0, 100.0 - severity_pct)
    color = ("#4caf50" if severity_pct < 3 else
             "#ff9800" if severity_pct < 15 else
             "#f44336" if severity_pct < 35 else "#7b1fa2")
    fig, ax = plt.subplots(figsize=(3.8, 3.8))
    fig.patch.set_facecolor(_WH)
    ax.set_facecolor(_WH)
    ax.pie([severity_pct, healthy],
           colors=[color, "#e8e2d6"],
           startangle=90,
           wedgeprops=dict(width=0.42, edgecolor=_WH, linewidth=2))
    ax.text(0, 0.08, f"{severity_pct:.1f}%",
            ha="center", va="center", fontsize=20, fontweight="bold", color=color)
    ax.text(0, -0.22, "stressed",
            ha="center", va="center", fontsize=9, color="#999")
    if label:
        ax.set_title(label[:30], fontsize=9, color="#555", pad=6)
    plt.tight_layout()
    return fig


def plot_calibration_scores(names: List[str], scores: List[float],
                             best_idx: int = -1) -> plt.Figure:
    fig, ax = _base(w=max(6, len(names) * 1.2), h=3.5)
    colors = [_G1 if i == best_idx else _G3 for i in range(len(names))]
    ax.bar(names, scores, color=colors, edgecolor=_WH, linewidth=0.6)
    ax.set_ylabel("Quality Score", fontsize=10, color="#555")
    ax.set_title("Calibration Image Quality Scores",
                 fontsize=11, fontweight="bold", color=_G1, pad=10)
    ax.set_xticklabels([n[:18] for n in names], rotation=20, ha="right", fontsize=8)
    if best_idx >= 0:
        ax.legend(handles=[mpatches.Patch(color=_G1, label="Best window")], fontsize=9)
    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: List[List[int]], labels: List[str]) -> plt.Figure:
    cm_arr   = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.where(row_sums > 0, cm_arr / row_sums, 0)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))
    fig.patch.set_facecolor(_WH)
    im = ax.imshow(cm_norm, cmap=plt.cm.Greens, vmin=0, vmax=1)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Normalised", fontsize=9, color="#555")
    cb.ax.tick_params(labelsize=8, colors="#555")
    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_xticklabels([l[:14] for l in labels],
                                               rotation=30, ha="right", fontsize=9)
    ax.set_yticks(ticks); ax.set_yticklabels([l[:14] for l in labels], fontsize=9)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{int(cm_arr[i,j])}\n({cm_norm[i,j]*100:.0f}%)",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "#333",
                    fontweight="bold" if i == j else "normal")
    ax.set_xlabel("Predicted", fontsize=10, color="#555", labelpad=8)
    ax.set_ylabel("True",      fontsize=10, color="#555", labelpad=8)
    ax.set_title("Confusion Matrix (LOO)", fontsize=12,
                 fontweight="bold", color=_G1, pad=14)
    plt.tight_layout()
    return fig


def plot_per_class_metrics(report: dict, labels: List[str]) -> plt.Figure:
    metrics = ["precision", "recall", "f1-score"]
    x, w   = np.arange(len(labels)), 0.25
    fig, ax = _base(w=max(8, len(labels) * 1.8), h=4.5)
    for i, (metric, color) in enumerate(zip(metrics, [_G1, _G2, _G3])):
        vals = [report.get(l, {}).get(metric, 0.0) for l in labels]
        bars = ax.bar(x + i * w, vals, w, label=metric.capitalize(),
                      color=color, edgecolor=_WH, linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0.04:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, color="#444")
    ax.set_xticks(x + w)
    ax.set_xticklabels([l[:16] for l in labels], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.18); ax.set_ylabel("Score", fontsize=10, color="#555")
    ax.set_title("Per-Class Precision / Recall / F1",
                 fontsize=12, fontweight="bold", color=_G1, pad=12)
    ax.legend(fontsize=9, framealpha=0.7)
    plt.tight_layout()
    return fig


def plot_class_distribution(counts: Dict[str, int]) -> plt.Figure:
    labels = sorted(counts.keys())
    vals   = [counts[l] for l in labels]
    colors = [_CLASS_COLORS.get(l, _G2) for l in labels]
    fig, ax = _base(w=max(6, len(labels) * 1.5), h=4)
    bars = ax.bar(labels, vals, color=colors, edgecolor=_WH, linewidth=0.8)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(vals) * 0.01,
                str(v), ha="center", va="bottom", fontsize=9,
                color="#444", fontweight="600")
    ax.set_xticklabels([l[:16] for l in labels], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Images", fontsize=10, color="#555")
    ax.set_title("Class Distribution", fontsize=12,
                 fontweight="bold", color=_G1, pad=12)
    plt.tight_layout()
    return fig


def plot_feature_importance(X: np.ndarray, feature_names: List[str],
                             top_n: int = 15) -> plt.Figure:
    variances = np.var(X, axis=0)
    idx    = np.argsort(variances)[::-1][:top_n]
    names  = [feature_names[i][:22] for i in idx]
    vals   = variances[idx] / (variances[idx].max() + 1e-9)
    colors = [_G1 if v > 0.5 else _G2 if v > 0.25 else _G3 for v in vals]
    fig, ax = _base(w=9, h=max(4, top_n * 0.35))
    ax.barh(names[::-1], vals[::-1], color=colors[::-1],
            edgecolor=_WH, linewidth=0.5)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Normalised Variance", fontsize=10, color="#555")
    ax.set_title("Feature Importance (Variance Proxy)",
                 fontsize=12, fontweight="bold", color=_G1, pad=12)
    for v, bar in zip(vals[::-1],
                      ax.patches):
        ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.2f}", va="center", fontsize=8, color="#444")
    plt.tight_layout()
    return fig
