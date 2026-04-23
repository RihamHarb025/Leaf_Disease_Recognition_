"""
LeafGuard — Leaf Disease Recognition Platform
Run: streamlit run app.py
"""

import io, zipfile, tempfile
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from pipeline.preprocessing   import check_quality, preprocess
from pipeline.segmentation    import segment_leaf, segment_stress, highlight_stress
from pipeline.features        import extract_features, feature_vector, FEATURE_NAMES
from pipeline.classifier      import predict, rule_based_classify, train_knn, loo_evaluate
from pipeline.decision_engine import fuse
from pipeline.dataset         import load_dataset

from helpers import (
    load_image_rgb, append_log, load_log, clear_log,
    model_is_trained, model_summary_text, load_model_meta,
    df_to_csv_bytes, timestamp_filename,
    plot_severity_donut, plot_calibration_scores,
    plot_confusion_matrix, plot_per_class_metrics,
    plot_class_distribution, plot_feature_importance,
)

st.set_page_config(page_title="LeafGuard", layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

[data-testid="stAppViewContainer"] {
    background: #f0ece0;
    font-family: 'DM Sans', sans-serif;
    color: #1e2e10;
}
[data-testid="stHeader"],
[data-testid="stDecoration"] { display: none !important; }
.block-container { padding: 0 2.5rem 4rem !important; max-width: 1360px; }

/* Fix all native Streamlit widget labels so they read on the warm background */
.stSlider label p,
.stSelectbox label p,
.stFileUploader label p,
.stCheckbox label p,
.stMultiSelect label p,
label[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] p {
    color: #2d4a14 !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
}
[data-testid="stMetricLabel"] p  { color: #2d4a14 !important; font-weight: 700 !important; }
[data-testid="stMetricValue"]    { color: #1e2e10 !important; }
[data-testid="stFileUploadDropzone"] { background: #fff !important; border-color: #b0c890 !important; }
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small { color: #3a5020 !important; font-weight: 500 !important; }
.stTabs [data-baseweb="tab"] p  { color: #4a5e3a !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] p { color: #1e2e10 !important; }
.stExpander summary p            { color: #2d4a14 !important; font-weight: 600 !important; }
[data-testid="stMarkdownContainer"] p { color: #1e2e10; }
[data-testid="stMarkdownContainer"] strong { color: #2d4a14; }
.stProgress > div > div          { background: #4a6b2a !important; }
[data-testid="stSelectbox"] > div > div { background: #fff !important; }
[data-testid="stSelectbox"] > div > div span { color: #1e2e10 !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #4a5e3a !important; }

/* NAV BAR */
div[data-testid="stRadio"] {
    background: #2d4a14 !important;
    padding: 0 2.5rem !important;
    margin-bottom: 2rem !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 999 !important;
}
div[data-testid="stRadio"] > div:first-child { display: none !important; }
div[data-testid="stRadio"] > div:last-child {
    display: flex !important;
    flex-direction: row !important;
    align-items: center !important;
    gap: 2px !important;
    padding: 8px 0 !important;
}
div[data-testid="stRadio"] > div:last-child > label {
    display: flex !important;
    align-items: center !important;
    padding: 8px 16px !important;
    border-radius: 8px !important;
    cursor: pointer !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #b8d98a !important;
    white-space: nowrap !important;
    transition: background 0.15s !important;
    background: transparent !important;
}
div[data-testid="stRadio"] > div:last-child > label > div:first-child { display: none !important; }
div[data-testid="stRadio"] > div:last-child > label > div:last-child p {
    font-size: 13px !important;
    font-weight: 600 !important;
    margin: 0 !important;
    line-height: 1 !important;
    color: inherit !important;
}
div[data-testid="stRadio"] > div:last-child > label:hover {
    background: rgba(255,255,255,0.1) !important;
    color: #fff !important;
}
div[data-testid="stRadio"] > div:last-child label:has(input:checked) {
    background: #6b9a3a !important;
    color: #fff !important;
}
div[data-testid="stRadio"] > div:last-child label:has(input:checked) p { color: #fff !important; }

/* HERO */
.hero {
    background: linear-gradient(135deg, #1e3209 0%, #2d4a14 45%, #4a6b2a 100%);
    padding: 60px 56px; border-radius: 24px; color: #fff;
    margin-bottom: 2rem; box-shadow: 0 14px 40px rgba(20,40,5,.28);
    position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; right: -40px; top: -40px;
    width: 300px; height: 300px; background: rgba(255,255,255,.04); border-radius: 50%;
}
.hero-eye   { font-size: 11px; letter-spacing: 2.5px; text-transform: uppercase; color: #a8d478; margin-bottom: 12px; font-weight: 700; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 46px; line-height: 1.1; margin-bottom: 14px; max-width: 600px; }
.hero-sub   { font-size: 15.5px; color: #c8e6a0; line-height: 1.75; max-width: 560px; margin-bottom: 30px; }
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; }
.hero-pill  { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); border-radius: 999px; padding: 6px 15px; font-size: 12px; font-weight: 600; color: #e8f4d0; }

/* STATUS STRIP */
.sstrip { display: flex; gap: 10px; margin-bottom: 2rem; flex-wrap: wrap; }
.spill  { background: #fff; border: 1.5px solid #c8c4b8; border-radius: 999px; padding: 6px 14px; font-size: 12px; font-weight: 600; color: #3a4e2a; display: flex; align-items: center; gap: 6px; }
.dot    { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dg { background: #4caf50; } .dy { background: #ff9800; } .dx { background: #9e9e9e; }

/* SECTION HEADER */
.sh       { margin-bottom: 1.4rem; }
.sh-eye   { font-size: 10.5px; letter-spacing: 2px; text-transform: uppercase; color: #5a8a28; font-weight: 700; margin-bottom: 4px; }
.sh-title { font-family: 'DM Serif Display', serif; font-size: 30px; color: #1e2e10; line-height: 1.15; }
.sh-desc  { font-size: 14px; color: #3a5020; max-width: 560px; margin-top: 5px; }

/* CARDS */
.card {
    background: #fff; border: 1.5px solid #ccc8bc; border-radius: 18px;
    padding: 22px; box-shadow: 0 2px 10px rgba(0,0,0,.05); margin-bottom: 14px;
}
.card-lbl { font-size: 10.5px; text-transform: uppercase; letter-spacing: 1.5px; color: #5a6b50; font-weight: 700; margin-bottom: 6px; }
.card-val { font-family: 'DM Serif Display', serif; font-size: 22px; color: #1e2e10; }
.card-sub { font-size: 12px; color: #4a5e3a; margin-top: 3px; }
.card.cg  { border-left: 5px solid #22c55e; border-radius: 0 18px 18px 0; }
.card.cy  { border-left: 5px solid #f59e0b; border-radius: 0 18px 18px 0; }
.card.cr  { border-left: 5px solid #ef4444; border-radius: 0 18px 18px 0; }

/* HOME FEATURE CARDS */
.fc       { background: #fff; border: 1.5px solid #ccc8bc; border-radius: 18px; padding: 22px; min-height: 170px; box-shadow: 0 2px 10px rgba(0,0,0,.04); margin-bottom: 14px; }
.fc-icon  { width: 46px; height: 46px; border-radius: 13px; background: linear-gradient(135deg,#6b9a3a,#2d4a14); display: flex; align-items: center; justify-content: center; font-size: 21px; margin-bottom: 12px; }
.fc-title { font-family: 'DM Serif Display', serif; font-size: 19px; color: #1e2e10; margin-bottom: 5px; }
.fc-text  { font-size: 13px; color: #3a5020; line-height: 1.65; }

/* SENSOR BADGE */
.sb    { font-size: 10.5px; font-weight: 700; padding: 2px 9px; border-radius: 999px; margin-left: 6px; }
.b-ok  { background: #e8f5e9; color: #1b5e20; }
.b-wa  { background: #fff3e0; color: #bf360c; }
.b-ba  { background: #ffebee; color: #b71c1c; }

/* SENSOR TABLE */
.stbl { background: #fff; border-radius: 16px; padding: 20px; border: 1.5px solid #ccc8bc; margin-top: 14px; }
.srow { display: flex; justify-content: space-between; align-items: center; padding: 9px 0; border-bottom: 1px solid #e8e2d6; font-size: 13.5px; }
.srow:last-child { border-bottom: none; }
.srow-k { color: #3a5020; font-weight: 600; }
.srow-v { font-weight: 700; color: #1e2e10; }

/* ACTUATOR */
.act    { background: #fff; border-radius: 13px; padding: 14px; text-align: center; border: 2px solid #ccc8bc; margin-bottom: 10px; }
.act.on { border-color: #4caf50; background: #f0faf0; }
.act-icon  { font-size: 23px; margin-bottom: 5px; }
.act-name  { font-size: 10.5px; color: #3a5020; font-weight: 700; text-transform: uppercase; letter-spacing: .7px; }
.act-state { font-size: 15px; font-weight: 800; margin-top: 3px; }
.act.on  .act-state { color: #2e7d32; }
.act.off .act-state { color: #9e9e9e; }

/* RECOMMENDATION BOX */
.rec     { background: #eaf4d6; border: 1.5px solid #a0c068; border-radius: 16px; padding: 22px; margin-top: 16px; }
.rec-lbl { font-size: 10.5px; letter-spacing: 2px; text-transform: uppercase; color: #2d5010; font-weight: 700; margin-bottom: 7px; }
.rec-txt { font-family: 'DM Serif Display', serif; font-size: 16.5px; color: #1e2e10; line-height: 1.6; }

/* IMAGE CARD */
.imgcard     { background: #fff; border-radius: 14px; padding: 10px; border: 1.5px solid #ccc8bc; text-align: center; }
.imgcard-lbl { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #3a5020; margin-top: 7px; }

/* SEVERITY BAR */
.sevwrap { background: #d8d2c8; border-radius: 999px; height: 9px; overflow: hidden; margin: 7px 0; }
.sevbar  { height: 100%; border-radius: 999px; }

.divider { border: none; border-top: 1px solid #ccc8bc; margin: 22px 0; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

for k, v in dict(
    last_diagnosis=None, last_features=None, last_pred=None,
    best_cal_time=None, train_result=None, train_X=None, train_y=None,
    sensors=dict(ph=6.5, ec=1.5, temperature=22.0,
                 humidity=60.0, water_level="normal", light="medium"),
).items():
    if k not in st.session_state:
        st.session_state[k] = v

PAGES = ["🏠 Home", "⚙️ Calibration", "🔬 Diagnosis",
         "🧪 Sensors", "🧠 Dashboard", "📋 Logs", "🎓 Train"]

selected = st.radio("nav", PAGES, horizontal=True,
                    label_visibility="collapsed", key="nav_radio",
                    index=PAGES.index(st.session_state.get("_nav", PAGES[0])))
st.session_state["_nav"] = selected
page = selected.split(" ", 1)[-1]


def _sev_color(cls):
    return {"none": "#4caf50", "mild": "#ff9800",
            "moderate": "#f44336", "severe": "#7b1fa2"}.get(cls, "#aaa")

def _risk_icon(r):
    return {"low": "🟢", "moderate": "🟡", "high": "🔴"}.get(r, "⚪")

def _run_pipeline(img_rgb):
    quality = check_quality(img_rgb)
    if not quality.passed:
        return None, quality, None, None, None
    proc        = preprocess(img_rgb)
    leaf_mask   = segment_leaf(proc)
    s_info      = segment_stress(proc, leaf_mask)
    stress_mask = s_info["stress_mask"]
    highlighted = highlight_stress(proc, stress_mask)
    feats       = extract_features(proc, leaf_mask, stress_mask)
    fvec        = feature_vector(feats)
    pred        = predict(fvec)
    if not pred["model_available"]:
        pred    = rule_based_classify(feats)
    decision    = fuse(pred, feats, st.session_state.sensors)
    masks = dict(proc=proc, leaf=leaf_mask, stress=stress_mask, hi=highlighted)
    return decision, quality, feats, pred, masks


# HOME
if page == "Home":
    last = st.session_state.last_diagnosis
    st.markdown(f"""
    <div class="hero">
        <div class="hero-eye">Hydroponic Intelligence Platform</div>
        <div class="hero-title">Smart Leaf Stress Diagnosis and Decision Support</div>
        <div class="hero-sub">
            Upload leaf images, detect stress patterns, simulate hydroponic sensor
            conditions, and receive actionable treatment decisions.
        </div>
        <div class="hero-pills">
            <div class="hero-pill">Vision-Based Diagnosis</div>
            <div class="hero-pill">Sensor Fusion</div>
            <div class="hero-pill">Actuator Simulation</div>
            <div class="hero-pill">Session Logging</div>
        </div>
    </div>
    <div class="sstrip">
        <div class="spill"><span class="dot dg"></span>System Ready</div>
        <div class="spill">
            <span class="dot {'dg' if model_is_trained() else 'dy'}"></span>
            {model_summary_text()}
        </div>
        <div class="spill">
            <span class="dot {'dg' if st.session_state.best_cal_time else 'dx'}"></span>
            Calibration: {st.session_state.best_cal_time or 'Not configured'}
        </div>
        <div class="spill">
            <span class="dot {'dy' if last else 'dx'}"></span>
            Last result: {last['label_display'] if last else 'None'}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""<div class="sh">
        <div class="sh-eye">Platform Overview</div>
        <div class="sh-title">What This System Does</div>
        <div class="sh-desc">A complete end-to-end pipeline from raw image to actionable treatment decision.</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    for col, icon, title, text in [
        (c1, "🍃", "Diagnose Leaf",      "Segmentation and stress classification from a single leaf photo."),
        (c2, "🩺", "Severity Analysis",  "Infected area percentage with a four-tier severity classification system."),
        (c1, "🧪", "Sensor Simulation",  "Adjust pH, EC, water level, temperature and humidity to match your system."),
        (c2, "📊", "Decision Dashboard", "Image and sensor fusion with actuator state recommendations."),
        (c1, "⚙️", "Calibration",        "Find the optimal daily image capture window for your grow environment."),
        (c2, "🎓", "Train Model",         "Upload your dataset, train the k-NN classifier, and review LOO accuracy."),
    ]:
        col.markdown(f"""<div class="fc">
            <div class="fc-icon">{icon}</div>
            <div class="fc-title">{title}</div>
            <div class="fc-text">{text}</div>
        </div>""", unsafe_allow_html=True)


# CALIBRATION
elif page == "Calibration":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Setup</div>
        <div class="sh-title">Image Calibration</div>
        <div class="sh-desc">Upload images taken at different times of day to identify the best capture window for consistent diagnosis results.</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([7, 3], gap="large")
    with left:
        files = st.file_uploader(
            "Upload images (include the hour in the filename, e.g. plant_08.jpg)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        if files:
            st.markdown(f"**{len(files)} image(s) uploaded**")
            prev = st.columns(min(3, len(files)))
            for i, f in enumerate(files[:6]):
                img = load_image_rgb(f.read())
                if img is not None:
                    prev[i % 3].markdown('<div class="imgcard">', unsafe_allow_html=True)
                    prev[i % 3].image(img, use_column_width=True)
                    prev[i % 3].markdown(
                        f'<div class="imgcard-lbl">{f.name[:20]}</div></div>',
                        unsafe_allow_html=True)

            if st.button("Run Calibration Analysis", type="primary"):
                rows = []
                for f in files:
                    f.seek(0)
                    img = load_image_rgb(f.read())
                    if img is None:
                        continue
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    br = float(np.mean(gray))
                    co = float(np.std(gray))
                    bl = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                    t = "Unknown"
                    for part in f.name.replace("-", "_").replace(".", "_").split("_"):
                        if part.isdigit() and 0 <= int(part) <= 23:
                            t = f"{int(part):02d}:00"
                            break
                    q = round(0.3 * min(br / 120, 1) + 0.3 * min(co / 50, 1) + 0.4 * min(bl / 300, 1), 3)
                    rows.append({"Image": f.name, "Time": t,
                                 "Brightness": round(br, 1), "Contrast": round(co, 1),
                                 "Blur Score": round(bl, 1), "Quality Score": q})
                if rows:
                    df = pd.DataFrame(rows)
                    best_i = int(df["Quality Score"].idxmax())
                    st.session_state.best_cal_time = df.iloc[best_i]["Time"]
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    fig = plot_calibration_scores(df["Image"].tolist(),
                                                  df["Quality Score"].tolist(), best_i)
                    st.pyplot(fig, use_container_width=True)

    with right:
        bt = st.session_state.best_cal_time
        if bt:
            st.markdown(f"""<div class="card cg" style="margin-top: 0">
                <div class="card-lbl">Best Capture Window</div>
                <div class="card-val" style="font-size: 38px">{bt}</div>
                <div class="card-sub">Highest combined image quality score</div>
            </div>
            <div class="rec">
                <div class="rec-lbl">Recommendation</div>
                <div class="rec-txt">Schedule your daily capture at <strong>{bt}</strong> for the most reliable diagnosis results.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="card">
                <div class="card-lbl">Awaiting Calibration</div>
                <div class="card-val" style="color: #9e9e9e; font-size: 18px">Not configured</div>
                <div class="card-sub">Upload images and run the analysis to find your optimal window.</div>
            </div>""", unsafe_allow_html=True)


# DIAGNOSIS
elif page == "Diagnosis":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Core Pipeline</div>
        <div class="sh-title">Leaf Diagnosis</div>
        <div class="sh-desc">Upload a leaf photo to run the full quality check, segmentation, feature extraction, and stress classification pipeline.</div>
    </div>""", unsafe_allow_html=True)

    left, right = st.columns([6, 4], gap="large")
    with left:
        uploaded = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png", "jfif"])
        if uploaded:
            original = load_image_rgb(uploaded.read())
            if original is None:
                st.error("Could not decode the image. Please try a different file.")
            else:
                with st.spinner("Analysing image..."):
                    decision, quality, feats, pred, masks = _run_pipeline(original)

                st.markdown("**Image Quality Report**")
                qc = st.columns(4)
                qc[0].metric("Sharpness",  quality.blur_label,        f"{quality.blur_score:.0f}")
                qc[1].metric("Brightness", f"{quality.brightness:.1f}")
                qc[2].metric("Contrast",   f"{quality.contrast:.1f}")
                qc[3].metric("HF Energy",  f"{quality.high_freq_energy:.2f}")

                if not quality.passed:
                    st.error("This image does not meet the minimum quality requirements for reliable diagnosis.")
                    for iss in quality.issues:
                        st.warning(iss)
                else:
                    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                    st.markdown("**Segmentation Output**")
                    r1a, r1b = st.columns(2)
                    r2a, r2b = st.columns(2)
                    for col, arr, lbl in [
                        (r1a, original,                                           "Original"),
                        (r1b, masks["proc"],                                      "Preprocessed"),
                        (r2a, cv2.cvtColor(masks["leaf"],   cv2.COLOR_GRAY2RGB),  "Leaf Mask"),
                        (r2b, cv2.cvtColor(masks["stress"], cv2.COLOR_GRAY2RGB),  "Stress Mask"),
                    ]:
                        col.markdown('<div class="imgcard">', unsafe_allow_html=True)
                        col.image(arr, use_column_width=True)
                        col.markdown(f'<div class="imgcard-lbl">{lbl}</div></div>',
                                     unsafe_allow_html=True)

                    st.markdown('<div class="imgcard" style="margin-top: 12px">',
                                unsafe_allow_html=True)
                    st.image(masks["hi"], use_column_width=True)
                    st.markdown('<div class="imgcard-lbl">Highlighted Stress Regions</div></div>',
                                unsafe_allow_html=True)

                    st.session_state.last_diagnosis = decision
                    st.session_state.last_features  = feats
                    st.session_state.last_pred       = pred

    with right:
        d = st.session_state.last_diagnosis
        f = st.session_state.last_features or {}
        p = st.session_state.last_pred or {}

        if d:
            sev_pct   = d["severity_pct"]
            sev_color = _sev_color(d["severity_class"])
            col_d, col_s = st.columns([3, 2])
            with col_d:
                st.markdown(f"""<div class="card">
                    <div class="card-lbl">Diagnosis</div>
                    <div class="card-val" style="font-size: 17px">{d['label_display']}</div>
                    <div class="card-sub">Confidence: {d['confidence']*100:.1f}%</div>
                </div>
                <div class="card">
                    <div class="card-lbl">Severity</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:28px;color:{sev_color}">{sev_pct:.1f}%</div>
                    <div class="sevwrap"><div class="sevbar" style="width:{min(sev_pct,100):.1f}%;background:{sev_color}"></div></div>
                    <div style="font-size:11px;color:#3a5020;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:4px">{d['severity_class'].capitalize()}</div>
                </div>""", unsafe_allow_html=True)
            with col_s:
                st.pyplot(plot_severity_donut(sev_pct), use_container_width=True)

            alert_cls = "cr" if d["risk_level"] == "high" else ("cy" if d["risk_level"] == "moderate" else "cg")
            st.markdown(f"""<div class="card {alert_cls}">
                <div class="card-lbl">Risk Level</div>
                <div class="card-val" style="font-size:18px">{_risk_icon(d['risk_level'])} {d['risk_level'].capitalize()}</div>
            </div>
            <div class="card">
                <div class="card-lbl">Leaf Metrics</div>
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:6px">
                    <div>
                        <div style="font-size:10px;color:#3a5020;font-weight:700;text-transform:uppercase">Leaf Area</div>
                        <div style="font-size:16px;font-weight:700;color:#1e2e10">{f.get('leaf_area',0):,}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;color:#3a5020;font-weight:700;text-transform:uppercase">Stress Area</div>
                        <div style="font-size:16px;font-weight:700;color:{sev_color}">{f.get('stress_area',0):,}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;color:#3a5020;font-weight:700;text-transform:uppercase">Regions</div>
                        <div style="font-size:16px;font-weight:700;color:#1e2e10">{f.get('num_regions',0)}</div>
                    </div>
                    <div>
                        <div style="font-size:10px;color:#3a5020;font-weight:700;text-transform:uppercase">Largest</div>
                        <div style="font-size:16px;font-weight:700;color:#1e2e10">{f.get('largest_region_area',0):,}</div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            with st.expander("Class Probabilities"):
                for cls, prob in sorted(p.get("all_probs", {}).items(), key=lambda x: -x[1]):
                    st.markdown(f"`{cls}`")
                    st.progress(float(prob), text=f"{prob*100:.1f}%")
        else:
            st.markdown("""<div class="card">
                <div class="card-lbl">Awaiting Image</div>
                <div class="card-val" style="color:#9e9e9e;font-size:17px">No image uploaded</div>
                <div class="card-sub">Upload a leaf image on the left to see the diagnosis results.</div>
            </div>""", unsafe_allow_html=True)


# SENSORS
elif page == "Sensors":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Environment</div>
        <div class="sh-title">Sensor Panel</div>
        <div class="sh-desc">Configure your hydroponic sensor readings. These values are combined with image analysis to produce the final treatment decision.</div>
    </div>""", unsafe_allow_html=True)

    s = st.session_state.sensors
    l, r = st.columns(2, gap="large")

    with l:
        st.markdown("**Nutrient and Water**")
        ph  = st.slider("pH Level",        4.0, 8.0,  float(s["ph"]),          0.1)
        ec  = st.slider("EC (mS/cm)",       0.5, 3.5,  float(s["ec"]),          0.1)
        wl  = st.selectbox("Water Level",   ["full", "normal", "low", "empty"],
                           index=["full", "normal", "low", "empty"].index(s["water_level"]))
        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("**Climate**")
        tmp = st.slider("Temperature (C)", 10.0, 40.0, float(s["temperature"]), 0.5)
        hum = st.slider("Humidity (%)",    20.0, 95.0, float(s["humidity"]),    1.0)

    with r:
        st.markdown("**Lighting**")
        lgt = st.selectbox("Light Intensity", ["low", "medium", "high"],
                           index=["low", "medium", "high"].index(s["light"]))

    st.session_state.sensors = dict(ph=ph, ec=ec, water_level=wl,
                                    temperature=tmp, humidity=hum, light=lgt)

    def _bc(v, lo, hi): return "b-ok" if lo <= v <= hi else "b-wa"
    def _wc(v): return {"full": "b-ok", "normal": "b-ok", "low": "b-wa", "empty": "b-ba"}.get(v, "b-ok")

    ph_lbl  = "Optimal" if 5.5 <= ph  <= 6.5 else "Out of range"
    ec_lbl  = "Optimal" if 1.2 <= ec  <= 2.4 else ("Low" if ec < 1.2 else "High")
    tmp_lbl = "Optimal" if 18  <= tmp <= 28   else "Out of range"
    hum_lbl = "Optimal" if 50  <= hum <= 80   else "Out of range"

    st.markdown(f"""<div class="stbl">
        <div style="font-family:'DM Serif Display',serif;font-size:19px;color:#1e2e10;margin-bottom:12px">Live Sensor Summary</div>
        <div class="srow">
            <span class="srow-k">pH</span>
            <span class="srow-v">{ph:.1f}<span class="sb {_bc(ph,5.5,6.5)}">{ph_lbl}</span></span>
        </div>
        <div class="srow">
            <span class="srow-k">EC (mS/cm)</span>
            <span class="srow-v">{ec:.1f}<span class="sb {_bc(ec,1.2,2.4)}">{ec_lbl}</span></span>
        </div>
        <div class="srow">
            <span class="srow-k">Water Level</span>
            <span class="srow-v"><span class="sb {_wc(wl)}">{wl.capitalize()}</span></span>
        </div>
        <div class="srow">
            <span class="srow-k">Temperature</span>
            <span class="srow-v">{tmp:.1f} C<span class="sb {_bc(tmp,18,28)}">{tmp_lbl}</span></span>
        </div>
        <div class="srow">
            <span class="srow-k">Humidity</span>
            <span class="srow-v">{hum:.1f}%<span class="sb {_bc(hum,50,80)}">{hum_lbl}</span></span>
        </div>
        <div class="srow">
            <span class="srow-k">Light Intensity</span>
            <span class="srow-v">{lgt.capitalize()}</span>
        </div>
    </div>""", unsafe_allow_html=True)


# DASHBOARD
elif page == "Dashboard":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Fused Decision Engine</div>
        <div class="sh-title">Decision Dashboard</div>
        <div class="sh-desc">Image analysis combined with live sensor readings to produce treatment decisions and actuator states.</div>
    </div>""", unsafe_allow_html=True)

    d = st.session_state.last_diagnosis
    f = st.session_state.last_features or {}
    p = st.session_state.last_pred or {}

    if not d:
        st.info("No diagnosis found. Go to Diagnosis and upload a leaf image first.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        model_src = "ML model" if p.get("model_available") else "Rule-based"
        for col, lbl, val, sub in [
            (c1, "Diagnosis",  d["label_display"],            model_src),
            (c2, "Severity",   f"{d['severity_pct']:.1f}%",   d["severity_class"].capitalize()),
            (c3, "Risk",       f"{_risk_icon(d['risk_level'])} {d['risk_level'].capitalize()}", "Combined score"),
            (c4, "Confidence", f"{d['confidence']*100:.1f}%",  "Classification"),
        ]:
            col.markdown(f"""<div class="card">
                <div class="card-lbl">{lbl}</div>
                <div class="card-val" style="font-size:17px">{val}</div>
                <div class="card-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        ll, rr = st.columns(2, gap="large")

        with ll:
            st.markdown("**System Interpretation**")
            ac = "cr" if d["risk_level"] == "high" else ("cy" if d["risk_level"] == "moderate" else "cg")
            st.markdown(f"""<div class="card {ac}">
                <div class="card-lbl">Analysis</div>
                <div class="card-val" style="font-size:14px;line-height:1.55">{d['interpretation']}</div>
            </div>""", unsafe_allow_html=True)

            for iss in d.get("sensor_issues", []):
                st.markdown(f"""<div class="card cy" style="padding:12px 16px">
                    <div style="font-size:13px;color:#7a3a00">Warning: {iss}</div>
                </div>""", unsafe_allow_html=True)

            sv = st.session_state.sensors
            st.markdown(f"""<div class="stbl" style="margin-top:12px">
                <div style="font-size:11px;font-weight:700;color:#3a5020;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Active Sensor Readings</div>
                <div class="srow"><span class="srow-k">pH</span><span class="srow-v">{sv['ph']:.1f}</span></div>
                <div class="srow"><span class="srow-k">EC</span><span class="srow-v">{sv['ec']:.1f} mS/cm</span></div>
                <div class="srow"><span class="srow-k">Water Level</span><span class="srow-v">{sv['water_level'].capitalize()}</span></div>
                <div class="srow"><span class="srow-k">Temperature</span><span class="srow-v">{sv['temperature']:.1f} C</span></div>
            </div>""", unsafe_allow_html=True)

        with rr:
            st.markdown("**Actuator States**")
            acts = d["actuators"]
            act_defs = [
                ("main_water_pump", "💧", "Water Pump"),
                ("nutrient_pump",   "🧪", "Nutrient Pump"),
                ("warning_buzzer",  "🔔", "Buzzer"),
                ("visual_alert",    "🚨", "Visual Alert"),
                ("maintenance_flag","🔧", "Maintenance"),
            ]
            for chunk in [act_defs[:3], act_defs[3:]]:
                cols = st.columns(len(chunk))
                for col, (key, icon, name) in zip(cols, chunk):
                    on = acts.get(key, False)
                    col.markdown(f"""<div class="act {'on' if on else 'off'}">
                        <div class="act-icon">{icon}</div>
                        <div class="act-name">{name}</div>
                        <div class="act-state">{'ON' if on else 'OFF'}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="rec">
            <div class="rec-lbl">Recommended Action</div>
            <div class="rec-txt">{d['recommendation']}</div>
        </div>""", unsafe_allow_html=True)

        with st.expander("Full Feature Summary"):
            LABELS = {
                "leaf_area": "Leaf Area (px)", "stress_area": "Stress Area (px)",
                "severity_pct": "Severity (%)", "num_regions": "Stress Regions",
                "largest_region_area": "Largest Region",
                "compactness": "Compactness", "aspect_ratio": "Aspect Ratio",
                "extent": "Extent", "solidity": "Solidity",
                "color_diff": "Color Difference", "edge_density": "Edge Density",
                "glcm_contrast": "GLCM Contrast", "glcm_homogeneity": "GLCM Homogeneity",
                "lbp_mean": "LBP Mean", "local_variance": "Local Variance",
            }
            rows = [{"Feature": LABELS.get(k, k),
                     "Value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                    for k, v in f.items() if k in LABELS]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        if st.button("Save to Log", type="primary"):
            append_log(d, st.session_state.sensors)
            st.success("Session saved.")


# LOGS
elif page == "Logs":
    st.markdown("""<div class="sh">
        <div class="sh-eye">History</div>
        <div class="sh-title">Session Logs</div>
        <div class="sh-desc">All saved diagnosis sessions with sensor values and treatment decisions.</div>
    </div>""", unsafe_allow_html=True)

    df_all = load_log()
    if df_all.empty:
        st.info("No sessions saved yet. Complete a diagnosis and use Save to Log from the Dashboard.")
    else:
        fc1, fc2, _ = st.columns(3)
        sev_u  = list(df_all["severity_class"].unique()) if "severity_class" in df_all.columns else []
        risk_u = list(df_all["risk_level"].unique())     if "risk_level"     in df_all.columns else []
        sf = fc1.multiselect("Filter by Severity", ["none", "mild", "moderate", "severe"], default=sev_u)
        rf = fc2.multiselect("Filter by Risk",     ["low", "moderate", "high"],            default=risk_u)
        df = load_log(severity_filter=sf or None, risk_filter=rf or None)
        st.markdown(f"**{len(df)} session(s) shown**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        dl_col, cl_col = st.columns([5, 1])
        dl_col.download_button("Download CSV", data=df_to_csv_bytes(df),
                               file_name=timestamp_filename("leafguard_log", "csv"),
                               mime="text/csv")
        if cl_col.button("Clear All"):
            clear_log()
            st.warning("All logs have been cleared.")
            st.rerun()


# TRAIN
elif page == "Train":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Model Training</div>
        <div class="sh-title">Train and Evaluate</div>
        <div class="sh-desc">Upload your dataset as a ZIP file, train the k-NN classifier, and review Leave-One-Out cross-validation results.</div>
    </div>""", unsafe_allow_html=True)

    meta = load_model_meta()
    st.markdown(f"""<div class="sstrip">
        <div class="spill"><span class="dot {'dg' if model_is_trained() else 'dy'}"></span>{model_summary_text()}</div>
    </div>""", unsafe_allow_html=True)

    tab_tr, tab_ev, tab_mi = st.tabs(["Train", "Evaluation Results", "Model Info"])

    with tab_tr:
        st.info(
            "Upload a ZIP file of your dataset. The folder structure inside should be: "
            "dataset_imgs/Tomato___healthy/, dataset_imgs/Tomato___Early_blight/, etc. "
            "Folder names are mapped automatically to the five canonical stress classes."
        )
        uploaded_zip = st.file_uploader("Dataset ZIP file", type=["zip"], key="tzp")
        ck, cm_, cl_ = st.columns(3)
        k_v    = ck.slider("k (neighbours)", 3, 15, 5)
        mx     = cm_.slider("Max images per class", 20, 500, 100, 10)
        do_loo = cl_.checkbox("Run LOO cross-validation", value=True)

        if uploaded_zip and st.button("Start Training", type="primary"):
            with tempfile.TemporaryDirectory() as td:
                with zipfile.ZipFile(io.BytesIO(uploaded_zip.read())) as zf:
                    zf.extractall(td)
                root = td
                for cand in sorted(Path(td).rglob("*")):
                    if cand.is_dir() and any(c.is_dir() for c in cand.iterdir()):
                        root = str(cand)
                        break

                prog = st.progress(0, "Loading dataset...")
                try:
                    X, y, _ = load_dataset(root, max_per_class=mx, verbose=False)
                    prog.progress(40, "Feature extraction complete")
                    dist = Counter(y)
                    st.session_state.train_X = X
                    st.session_state.train_y = y

                    ti = train_knn(X, y, k=k_v)
                    prog.progress(65, "Classifier trained")

                    loo_res = None
                    if do_loo and len(y) >= 4:
                        prog.progress(68, f"Running LOO on {len(y)} samples...")
                        loo_res = loo_evaluate(X, y)
                        st.session_state.train_result = loo_res
                        prog.progress(98, "Evaluation complete")
                    else:
                        st.session_state.train_result = None

                    prog.progress(100, "Done")

                    r1, r2, r3, r4 = st.columns(4)
                    for col, lbl, val in [
                        (r1, "Total Samples",  str(len(y))),
                        (r2, "Classes Found",  str(len(dist))),
                        (r3, "Train Accuracy", f"{ti['train_accuracy']*100:.1f}%"),
                        (r4, "LOO Accuracy",   f"{loo_res['loo_accuracy']*100:.1f}%" if loo_res else "N/A"),
                    ]:
                        col.markdown(f"""<div class="card cg">
                            <div class="card-lbl">{lbl}</div>
                            <div class="card-val">{val}</div>
                        </div>""", unsafe_allow_html=True)

                    st.pyplot(plot_class_distribution(dict(dist)), use_container_width=True)
                    st.success("Model saved to models/knn_model.pkl")
                except Exception as e:
                    prog.empty()
                    st.error(f"Training failed: {e}")
                    raise

    with tab_ev:
        loo = st.session_state.train_result
        if not loo:
            st.info("Train with LOO enabled to see evaluation results here.")
        else:
            acc    = loo["loo_accuracy"]
            labels = loo["labels"]
            rep    = loo["classification_report"]
            macro  = rep.get("macro avg", {})

            m1, m2, m3 = st.columns(3)
            for col, lbl, val in [
                (m1, "LOO Accuracy",    f"{acc*100:.2f}%"),
                (m2, "Macro F1",        f"{macro.get('f1-score',0)*100:.2f}%"),
                (m3, "Macro Precision", f"{macro.get('precision',0)*100:.2f}%"),
            ]:
                ok = float(val.rstrip("%")) >= 70
                col.markdown(f"""<div class="card {'cg' if ok else 'cy'}">
                    <div class="card-lbl">{lbl}</div>
                    <div class="card-val">{val}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            ev1, ev2 = st.columns([3, 2], gap="large")
            with ev1:
                st.markdown("**Confusion Matrix**")
                st.pyplot(plot_confusion_matrix(loo["confusion_matrix"], labels),
                          use_container_width=True)
            with ev2:
                st.markdown("**Per-Class Breakdown**")
                rows = [{"Class": c,
                         "Precision": f"{rep.get(c,{}).get('precision',0):.3f}",
                         "Recall":    f"{rep.get(c,{}).get('recall',0):.3f}",
                         "F1":        f"{rep.get(c,{}).get('f1-score',0):.3f}",
                         "n":         int(rep.get(c,{}).get('support',0))}
                        for c in labels]
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.pyplot(plot_per_class_metrics(rep, labels), use_container_width=True)

            with st.expander("Misclassification Detail"):
                yt, yp = loo.get("y_true", []), loo.get("y_pred", [])
                wrong  = [(t, p_) for t, p_ in zip(yt, yp) if t != p_]
                if wrong:
                    df_w = pd.DataFrame(wrong, columns=["True", "Predicted"])
                    st.markdown(f"**{len(wrong)} errors** out of {len(yt)} samples")
                    st.dataframe(df_w.value_counts().reset_index().rename(columns={0: "Count"}),
                                 hide_index=True, use_container_width=True)
                else:
                    st.success("Perfect LOO score. No misclassifications.")

    with tab_mi:
        if not model_is_trained():
            st.info("No model has been trained yet.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            for col, lbl, val in [
                (m1, "Algorithm",  "k-NN"),
                (m2, "k",          str(meta.get("k", "?"))),
                (m3, "Weights",    meta.get("weights", "?")),
                (m4, "Trained on", str(meta.get("n_samples", "?")) + " samples"),
            ]:
                col.markdown(f"""<div class="card">
                    <div class="card-lbl">{lbl}</div>
                    <div class="card-val" style="font-size:19px">{val}</div>
                </div>""", unsafe_allow_html=True)

            X = st.session_state.get("train_X")
            if X is not None:
                st.markdown("<hr class='divider'>", unsafe_allow_html=True)
                st.markdown("**Feature Importance (variance proxy)**")
                st.caption("Features with higher variance contribute more to the Euclidean distance in k-NN.")
                st.pyplot(plot_feature_importance(X, FEATURE_NAMES), use_container_width=True)
            else:
                st.info("Run a training session in this window to see feature importance.")
