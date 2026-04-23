"""
tests/test_pipeline.py
──────────────────────
Unit tests for each pipeline stage.
Run with: python -m pytest tests/
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


def _make_green_leaf(h=256, w=256):
    """Synthetic green leaf image (RGB)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :, 1] = 120  # green channel
    img[:, :, 0] = 40
    img[:, :, 2] = 40
    # add a yellow stress patch
    img[80:120, 80:140, 0] = 200
    img[80:120, 80:140, 1] = 180
    img[80:120, 80:140, 2] = 20
    return img


# ─────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────

def test_preprocess_shape():
    from pipeline.preprocessing import preprocess
    img = _make_green_leaf(300, 400)
    out = preprocess(img)
    assert out.shape == (512, 512, 3), "preprocess must output 512×512×3"


def test_quality_pass():
    from pipeline.preprocessing import check_quality
    img = _make_green_leaf(512, 512)
    # make it not too dark and not blurry (add noise)
    noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(np.int32) + noise, 0, 255).astype(np.uint8)
    report = check_quality(img)
    assert hasattr(report, "passed")
    assert isinstance(report.brightness, float)


def test_quality_dark_fails():
    from pipeline.preprocessing import check_quality
    dark = np.zeros((256, 256, 3), dtype=np.uint8)  # pure black
    report = check_quality(dark)
    assert not report.passed
    assert len(report.issues) > 0


# ─────────────────────────────────────────────
# Segmentation
# ─────────────────────────────────────────────

def test_segment_leaf_returns_mask():
    from pipeline.preprocessing import preprocess
    from pipeline.segmentation import segment_leaf
    img = preprocess(_make_green_leaf())
    mask = segment_leaf(img)
    assert mask.shape == img.shape[:2]
    assert mask.dtype == np.uint8
    assert np.count_nonzero(mask) > 0, "Leaf mask should not be empty on green image"


def test_segment_stress_dict():
    from pipeline.preprocessing import preprocess
    from pipeline.segmentation import segment_leaf, segment_stress
    img = preprocess(_make_green_leaf())
    leaf = segment_leaf(img)
    result = segment_stress(img, leaf)
    assert "stress_mask" in result
    assert "severity_pct" in result
    assert 0.0 <= result["severity_pct"] <= 100.0


# ─────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────

def test_feature_dict_keys():
    from pipeline.preprocessing import preprocess
    from pipeline.segmentation import segment_leaf, segment_stress
    from pipeline.features import extract_features, FEATURE_NAMES
    img = preprocess(_make_green_leaf())
    leaf = segment_leaf(img)
    stress = segment_stress(img, leaf)["stress_mask"]
    feats = extract_features(img, leaf, stress)
    for name in FEATURE_NAMES:
        assert name in feats, f"Missing feature: {name}"


def test_feature_vector_shape():
    from pipeline.preprocessing import preprocess
    from pipeline.segmentation import segment_leaf, segment_stress
    from pipeline.features import extract_features, feature_vector, FEATURE_NAMES
    img = preprocess(_make_green_leaf())
    leaf = segment_leaf(img)
    stress = segment_stress(img, leaf)["stress_mask"]
    feats = extract_features(img, leaf, stress)
    fvec = feature_vector(feats)
    assert fvec.shape == (len(FEATURE_NAMES),)
    assert fvec.dtype == np.float64


# ─────────────────────────────────────────────
# Classifier — rule-based (no model needed)
# ─────────────────────────────────────────────

def test_rule_based_classify():
    from pipeline.classifier import rule_based_classify, CLASSES
    feats = {
        "severity_pct": 20.0, "stress_mean_h": 25.0, "stress_mean_s": 100.0,
        "num_regions": 5, "compactness": 10.0,
    }
    result = rule_based_classify(feats)
    assert result["label"] in CLASSES
    assert 0.0 <= result["confidence"] <= 1.0
    assert set(result["all_probs"].keys()) == set(CLASSES)


def test_predict_no_model():
    from pipeline.classifier import predict
    import numpy as np
    fvec = np.zeros(28)
    result = predict(fvec)
    # should return a valid result even without trained model
    assert "label" in result
    assert "model_available" in result


# ─────────────────────────────────────────────
# Training (tiny synthetic dataset)
# ─────────────────────────────────────────────

def test_train_and_predict():
    from pipeline.classifier import train_knn, predict, CLASSES
    import numpy as np

    rng = np.random.default_rng(42)
    n = 20
    X = rng.random((n, 28))
    y = [CLASSES[i % len(CLASSES)] for i in range(n)]

    info = train_knn(X, y, k=3)
    assert 0.0 <= info["train_accuracy"] <= 1.0

    result = predict(X[0])
    assert result["model_available"] is True
    assert result["label"] in CLASSES


def test_loo_evaluate():
    from pipeline.classifier import loo_evaluate, CLASSES
    import numpy as np

    rng = np.random.default_rng(0)
    n = 15
    X = rng.random((n, 28))
    y = [CLASSES[i % len(CLASSES)] for i in range(n)]

    result = loo_evaluate(X, y)
    assert "loo_accuracy" in result
    assert "confusion_matrix" in result
    assert 0.0 <= result["loo_accuracy"] <= 1.0


# ─────────────────────────────────────────────
# Decision engine
# ─────────────────────────────────────────────

def test_fuse_output_keys():
    from pipeline.decision_engine import fuse

    pred = {"label": "chlorosis_yellowing", "confidence": 0.8,
            "all_probs": {}, "model_available": True}
    feats = {"severity_pct": 20.0, "leaf_area": 50000, "stress_area": 10000}
    sensors = {"ph": 6.8, "ec": 1.0, "temperature": 25.0,
               "humidity": 60.0, "water_level": "normal", "light": "medium"}

    d = fuse(pred, feats, sensors)
    for key in ["label","label_display","confidence","severity_pct",
                "severity_class","severity_color","sensor_issues",
                "interpretation","risk_level","actuators","recommendation"]:
        assert key in d, f"Missing key in decision: {key}"


def test_fuse_actuator_nutrient_pump_on_low_ec():
    from pipeline.decision_engine import fuse
    pred = {"label": "healthy", "confidence": 0.9, "all_probs": {}, "model_available": True}
    feats = {"severity_pct": 1.0, "leaf_area": 50000, "stress_area": 500}
    sensors = {"ph": 6.0, "ec": 0.8, "temperature": 22.0,
               "humidity": 65.0, "water_level": "normal", "light": "medium"}
    d = fuse(pred, feats, sensors)
    assert d["actuators"]["nutrient_pump"] is True


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


# ─────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────

def test_plot_confusion_matrix():
    import helpers
    import matplotlib.pyplot as plt
    cm = [[10, 2, 0], [1, 8, 1], [0, 2, 9]]
    labels = ["healthy", "chlorosis_yellowing", "spot_necrotic"]
    fig = helpers.plot_confusion_matrix(cm, labels)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_severity_donut():
    import helpers
    import matplotlib.pyplot as plt
    fig = helpers.plot_severity_donut(18.5, "Test")
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_feature_importance():
    import helpers
    from pipeline.features import FEATURE_NAMES
    import matplotlib.pyplot as plt
    import numpy as np
    X = np.random.rand(20, len(FEATURE_NAMES))
    fig = helpers.plot_feature_importance(X, FEATURE_NAMES)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_model_summary_no_model(tmp_path, monkeypatch):
    """model_summary_text returns fallback string when no model exists."""
    import helpers
    monkeypatch.setattr(helpers, "MODEL_DIR", tmp_path)
    result = helpers.model_is_trained()
    assert result is False
    text = helpers.model_summary_text()
    assert "rule-based" in text.lower() or "no trained" in text.lower()


def test_append_and_load_log(tmp_path, monkeypatch):
    import helpers
    monkeypatch.setattr(helpers, "LOG_PATH", tmp_path / "log.csv")

    decision = {
        "label": "healthy", "label_display": "Healthy",
        "confidence": 0.9, "severity_pct": 1.0,
        "severity_class": "none", "risk_level": "low",
        "recommendation": "All good.",
        "actuators": {
            "main_water_pump": False, "nutrient_pump": False,
            "warning_buzzer": False, "visual_alert": False,
            "maintenance_flag": False,
        },
    }
    sensors = {"ph": 6.0, "ec": 1.5, "water_level": "normal",
               "temperature": 22.0, "humidity": 65.0, "light": "medium"}

    helpers.append_log(decision, sensors)
    df = helpers.load_log()
    assert len(df) == 1
    assert df.iloc[0]["label"] == "healthy"
