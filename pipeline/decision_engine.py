"""
pipeline/decision_engine.py
───────────────────────────
Fuses classifier output with hydroponic sensor readings to produce:
  • Final label (possibly overriding the classifier)
  • Severity classification tier
  • Sensor anomaly list
  • Actuator state recommendations
  • Human-readable interpretation and recommendation
"""

from typing import Dict, Any


# ──────────────────────────────────────────────
# Thresholds
# ──────────────────────────────────────────────

SENSOR_RANGES = {
    "ph":          (5.5,  6.5),
    "ec":          (1.2,  2.4),
    "temperature": (18.0, 28.0),
    "humidity":    (50.0, 80.0),
}

SEVERITY_TIERS = [
    (0.0,  3.0,  "none",     "#4caf50"),
    (3.0,  15.0, "mild",     "#ff9800"),
    (15.0, 35.0, "moderate", "#f44336"),
    (35.0, 100.0,"severe",   "#7b1fa2"),
]

LABEL_DISPLAY = {
    "healthy":             "Healthy — No Stress",
    "chlorosis_yellowing": "Chlorosis / Yellowing Stress",
    "dryness_dehydration": "Dryness / Dehydration Stress",
    "spot_necrotic":       "Spot / Necrotic Lesions",
    "unknown_abnormality": "Unknown Abnormality",
}

RISK_MAP = {
    ("none",     "low"):      "low",
    ("none",     "moderate"): "low",
    ("none",     "high"):     "moderate",
    ("mild",     "low"):      "low",
    ("mild",     "moderate"): "moderate",
    ("mild",     "high"):     "moderate",
    ("moderate", "low"):      "moderate",
    ("moderate", "moderate"): "moderate",
    ("moderate", "high"):     "high",
    ("severe",   "low"):      "high",
    ("severe",   "moderate"): "high",
    ("severe",   "high"):     "high",
}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _severity_tier(pct: float):
    for lo, hi, cls, color in SEVERITY_TIERS:
        if lo <= pct < hi:
            return cls, color
    return "severe", "#7b1fa2"


def _sensor_pressure(sensors: dict) -> tuple:
    """Returns (issues list, sensor_stress_level: 'low'|'moderate'|'high')."""
    issues = []
    for key, (lo, hi) in SENSOR_RANGES.items():
        val = sensors.get(key)
        if val is None:
            continue
        v = float(val)
        if not (lo <= v <= hi):
            direction = "low" if v < lo else "high"
            unit_map = {"ph": "", "ec": " mS/cm", "temperature": " °C", "humidity": "%"}
            issues.append(f"{key.upper()} {direction} ({v}{unit_map.get(key,'')}; optimal {lo}–{hi})")

    wl = sensors.get("water_level", "normal")
    if wl in ("low", "empty"):
        issues.append(f"Water level {wl.upper()} — refill required")

    n = len(issues)
    level = "low" if n == 0 else ("moderate" if n <= 1 else "high")
    return issues, level


def _actuators(label: str, sev_class: str, sensor_issues: list,
               sensors: dict) -> dict:
    ec = float(sensors.get("ec", 1.5))
    wl = sensors.get("water_level", "normal")

    water_pump      = wl in ("low", "empty")
    nutrient_pump   = (ec < 1.2) or (label == "chlorosis_yellowing")
    warning_buzzer  = sev_class in ("severe",)
    visual_alert    = sev_class in ("moderate", "severe") or len(sensor_issues) >= 2
    maintenance_flag = sev_class == "severe" or wl == "empty"

    return dict(
        main_water_pump  = water_pump,
        nutrient_pump    = nutrient_pump,
        warning_buzzer   = warning_buzzer,
        visual_alert     = visual_alert,
        maintenance_flag = maintenance_flag,
    )


def _interpret(label: str, sev_class: str, sensor_issues: list,
               confidence: float) -> str:
    label_d = LABEL_DISPLAY.get(label, label)
    conf_s  = f"{confidence*100:.0f}%"

    base = f"{label_d} detected (model confidence {conf_s})."
    if sev_class == "none":
        base = "Leaf appears healthy with no significant stress detected."

    if sensor_issues:
        sensor_note = "Sensor readings also indicate: " + "; ".join(sensor_issues[:2]) + "."
    else:
        sensor_note = "All sensor readings are within normal range."

    return f"{base} {sensor_note}"


def _recommendation(label: str, sev_class: str,
                    sensor_issues: list, sensors: dict) -> str:
    ec  = float(sensors.get("ec",  1.5))
    ph  = float(sensors.get("ph",  6.5))
    wl  = sensors.get("water_level", "normal")

    recs = []
    if label == "chlorosis_yellowing":
        recs.append("Increase nutrient concentration — possible iron or nitrogen deficiency.")
        if ec < 1.2:
            recs.append(f"EC is low ({ec:.1f} mS/cm); target 1.5–2.0 mS/cm.")
    elif label == "dryness_dehydration":
        recs.append("Increase irrigation frequency or check if water delivery is blocked.")
    elif label == "spot_necrotic":
        recs.append("Isolate affected plants; inspect for fungal or bacterial pathogens.")

    if wl in ("low", "empty"):
        recs.append("Refill reservoir immediately.")
    if not (5.5 <= ph <= 6.5):
        dir_ = "lower" if ph > 6.5 else "raise"
        recs.append(f"Adjust pH ({ph:.1f}); {dir_} to 5.5–6.5 range.")

    if sev_class == "severe":
        recs.append("Severe stress detected — manual inspection strongly recommended.")
    elif sev_class == "none":
        recs = ["No action required. Continue routine monitoring."]

    return " ".join(recs) if recs else "Monitor closely and repeat diagnosis in 24 hours."


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

def fuse(prediction: dict, features: dict, sensors: dict) -> dict:
    """
    Combine classifier prediction + sensor state into a full decision dict.

    Parameters
    ----------
    prediction : output of classifier.predict() or rule_based_classify()
    features   : output of features.extract_features()
    sensors    : dict from session_state.sensors

    Returns
    -------
    Full decision dict consumed by app.py pages (Dashboard, Diagnosis).
    """
    label      = prediction.get("label", "unknown_abnormality")
    confidence = float(prediction.get("confidence", 0.5))
    sev_pct    = float(features.get("severity_pct", 0.0))

    sev_class, sev_color = _severity_tier(sev_pct)
    sensor_issues, sensor_stress = _sensor_pressure(sensors)

    # Sensor context can upgrade a 'healthy' classification
    if label == "healthy" and sensor_stress == "high":
        label = "unknown_abnormality"
        confidence = min(confidence, 0.55)

    risk_level  = RISK_MAP.get((sev_class, sensor_stress), "moderate")
    acts        = _actuators(label, sev_class, sensor_issues, sensors)
    interp      = _interpret(label, sev_class, sensor_issues, confidence)
    rec         = _recommendation(label, sev_class, sensor_issues, sensors)

    return dict(
        label          = label,
        label_display  = LABEL_DISPLAY.get(label, label),
        confidence     = confidence,
        severity_pct   = sev_pct,
        severity_class = sev_class,
        severity_color = sev_color,
        sensor_issues  = sensor_issues,
        interpretation = interp,
        risk_level     = risk_level,
        actuators      = acts,
        recommendation = rec,
    )
