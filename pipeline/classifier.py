"""
pipeline/classifier.py
──────────────────────
k-NN classifier for leaf stress classification.

• Training   : train_knn(X, y)  — fits and saves model + scaler
• Prediction : predict(fvec)    — loads model, returns label + confidence
• Evaluation : loo_evaluate(X, y) — Leave-One-Out cross-validation with
                                    confusion matrix and per-class accuracy
• Fallback   : rule_based_classify(feats) — pure rule engine when no model
               is trained yet.

Supported classes
─────────────────
  healthy              — no detectable stress
  chlorosis_yellowing  — nutrient deficiency / pH imbalance
  dryness_dehydration  — brown/dry leaf margins
  spot_necrotic        — discrete dark necrotic lesions
  unknown_abnormality  — stress present but type unclear
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import LeaveOneOut

MODEL_DIR  = Path(__file__).parent.parent / "models"
MODEL_PATH = MODEL_DIR / "knn_model.pkl"
META_PATH  = MODEL_DIR / "knn_meta.json"

CLASSES = [
    "healthy",
    "chlorosis_yellowing",
    "dryness_dehydration",
    "spot_necrotic",
    "unknown_abnormality",
]

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────

def train_knn(X: np.ndarray, y: List[str],
              k: int = 5,
              weights: str = "distance") -> dict:
    """
    Fit a k-NN on (X, y) with Z-score normalisation.
    Persists model + scaler to models/.
    Returns a training-set accuracy report.

    Parameters
    ----------
    X : (n_samples, n_features) float array
    y : list of class-name strings (must be in CLASSES)
    k : number of neighbours
    weights : 'uniform' | 'distance'
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric="euclidean")
    clf.fit(X_scaled, y)

    # training accuracy
    y_pred_train = clf.predict(X_scaled)
    train_acc = float(np.mean(np.array(y_pred_train) == np.array(y)))

    # persist
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)

    meta = {"k": k, "weights": weights, "n_samples": len(y),
            "classes": list(set(y)), "train_accuracy": round(train_acc, 4)}
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)

    return {"train_accuracy": train_acc, "n_samples": len(y), "k": k}


# ──────────────────────────────────────────────
# Leave-One-Out cross-validation
# ──────────────────────────────────────────────

def loo_evaluate(X: np.ndarray, y: List[str]) -> dict:
    """
    Run Leave-One-Out cross-validation.
    Refits a k=5 distance-weighted k-NN on every split.
    Returns accuracy, per-class precision/recall/f1,
    and a raw confusion matrix (rows=true, cols=pred).
    """
    labels = sorted(set(y))
    loo    = LeaveOneOut()
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = [y[i] for i in train_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        k = min(5, len(X_tr))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
        clf.fit(X_tr_s, y_tr)

        pred = clf.predict(X_te_s)[0]
        y_true_all.append(y[test_idx[0]])
        y_pred_all.append(pred)

    accuracy = float(np.mean(np.array(y_true_all) == np.array(y_pred_all)))
    cm       = confusion_matrix(y_true_all, y_pred_all, labels=labels).tolist()
    report   = classification_report(y_true_all, y_pred_all,
                                     labels=labels, output_dict=True,
                                     zero_division=0)

    return {
        "loo_accuracy": round(accuracy, 4),
        "confusion_matrix": cm,
        "labels": labels,
        "classification_report": report,
        "y_true": y_true_all,
        "y_pred": y_pred_all,
    }


# ──────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────

def predict(feature_vec: np.ndarray) -> dict:
    """
    Load saved model and predict a single sample.
    Returns label, confidence, all_probs, model_available.
    Falls back to uniform probs if model not found.
    """
    if not MODEL_PATH.exists():
        return _no_model_result()

    try:
        with open(MODEL_PATH, "rb") as f:
            bundle = pickle.load(f)
        clf: KNeighborsClassifier = bundle["clf"]
        scaler: StandardScaler   = bundle["scaler"]

        x = scaler.transform(feature_vec.reshape(1, -1))
        label = clf.predict(x)[0]
        proba = clf.predict_proba(x)[0]
        classes = list(clf.classes_)

        all_probs = {c: float(p) for c, p in zip(classes, proba)}
        confidence = float(max(proba))

        return dict(label=label, confidence=confidence,
                    all_probs=all_probs, model_available=True)
    except Exception as e:
        return _no_model_result(str(e))


def _no_model_result(reason: str = "") -> dict:
    uniform = 1.0 / len(CLASSES)
    return dict(
        label="unknown_abnormality",
        confidence=uniform,
        all_probs={c: uniform for c in CLASSES},
        model_available=False,
        reason=reason or "Model not trained yet.",
    )


# ──────────────────────────────────────────────
# Rule-based fallback classifier
# ──────────────────────────────────────────────

def rule_based_classify(feats: dict) -> dict:
    """
    Pure heuristic classifier — no model required.
    Uses colour means, severity, and shape features.
    """
    sev    = feats.get("severity_pct", 0.0)
    sh_h   = feats.get("stress_mean_h", 0.0)   # hue of stress pixels
    sh_s   = feats.get("stress_mean_s", 0.0)
    n_reg  = feats.get("num_regions", 0)
    compact= feats.get("compactness", 0.0)

    scores: Dict[str, float] = {c: 0.0 for c in CLASSES}

    if sev < 3.0:
        scores["healthy"] += 1.0

    # Yellowing / chlorosis: yellow-ish hue (10–35), lower sat
    if 10 <= sh_h <= 40 and sh_s < 160:
        scores["chlorosis_yellowing"] += 1.0

    # Dryness: brownish hue (5–20), higher perimeter-to-area ratio
    if 5 <= sh_h <= 22 and compact > 15:
        scores["dryness_dehydration"] += 0.8

    # Necrosis: many distinct compact spots
    if n_reg >= 3 and compact < 20:
        scores["spot_necrotic"] += 0.9

    if sev > 5.0:
        # reduce healthy weight when stress is clearly present
        scores["healthy"] *= max(0.0, 1.0 - sev / 50.0)

    total = sum(scores.values()) or 1.0
    probs = {k: v / total for k, v in scores.items()}
    label = max(probs, key=probs.get)

    return dict(label=label, confidence=probs[label],
                all_probs=probs, model_available=False)
