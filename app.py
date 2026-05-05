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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap');

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

/* FILE UPLOADER — cream bg, green border, override ALL states */
[data-testid="stFileUploadDropzone"],
[data-testid="stFileUploadDropzone"]:hover,
section[data-testid="stFileUploadDropzone"],
div[data-testid="stFileUploadDropzone"] {
    background: #f5f0e4 !important;
    background-color: #f5f0e4 !important;
    border: 2px dashed #7ab84a !important;
    border-radius: 14px !important;
    box-shadow: none !important;
}
[data-testid="stFileUploadDropzone"] *,
[data-testid="stFileUploadDropzone"] p,
[data-testid="stFileUploadDropzone"] small,
[data-testid="stFileUploadDropzone"] span,
[data-testid="stFileUploadDropzone"] div { 
    color: #2d5010 !important;
    background: transparent !important;
}
[data-testid="stFileUploadDropzone"] button {
    background: #eaf4d6 !important;
    border: 1.5px solid #7ab84a !important;
    color: #2d5010 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploadDropzone"] svg { 
    fill: #5a8a28 !important;
    color: #5a8a28 !important;
}
[data-testid="stFileUploaderDeleteBtn"],
[data-testid="stFileUploaderDeleteBtn"] * { 
    color: #3a5020 !important;
    background: transparent !important;
}

/* PAGE PANELS */
.page-panel {
    background: #fff;
    border: 1.5px solid #ccc8bc;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 16px;
}
.page-panel-title {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: #5a8a28;
    font-weight: 700;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.page-panel-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e0ddd4;
}
.info-box {
    background: #eaf4d6;
    border: 1.5px solid #a8d478;
    border-radius: 12px;
    padding: 14px 16px;
    font-size: 13px;
    color: #1e4010;
    line-height: 1.6;
    margin-bottom: 14px;
}
.info-box strong { color: #2d5010; }
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
.metric-chip {
    background: #f5f0e4;
    border: 1.5px solid #d0ccbf;
    border-radius: 12px;
    padding: 12px 14px;
}
.metric-chip-lbl { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #5a6b50; margin-bottom: 4px; }
.metric-chip-val { font-family: 'DM Serif Display', serif; font-size: 20px; color: #1e2e10; }
.metric-chip-sub { font-size: 11px; color: #6b7b5a; margin-top: 2px; }
.stTabs [data-baseweb="tab"] p  { color: #4a5e3a !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] p { color: #1e2e10 !important; }
.stExpander summary p            { color: #2d4a14 !important; font-weight: 600 !important; }
[data-testid="stMarkdownContainer"] p { color: #1e2e10; }
[data-testid="stMarkdownContainer"] strong { color: #2d4a14; }
.stProgress > div > div          { background: #4a6b2a !important; }
[data-testid="stSelectbox"] > div > div { background: #fff !important; }
[data-testid="stSelectbox"] > div > div span { color: #1e2e10 !important; }
/* SLIDERS — green throughout */
[data-testid="stSlider"] [role="slider"] {
    background: #4a7a1e !important;
    border-color: #4a7a1e !important;
}
[data-testid="stSlider"] > div > div > div > div {
    background: #4a7a1e !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] [data-testid="stSliderTrackFill"],
[data-testid="stSlider"] [data-baseweb="slider"] div[class*="Track"] > div:first-child {
    background: #4a7a1e !important;
}
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #4a5e3a !important; }

/* ── NAV BAR ── */
/* hide radio */
div[data-testid="stRadio"] {
    display: none !important;
}

/* NAV CONTAINER */
div[data-testid="stHorizontalBlock"]:first-of-type {
    background: #eee8d8 !important;
    border-radius: 16px !important;
    border: 1.5px solid #d6cfb8 !important;
    box-shadow: 0 2px 8px rgba(30,46,16,0.06) !important;

    padding: 6px 10px !important;
    margin-bottom: 18px !important;

    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

/* NAV BUTTONS */
div[data-testid="stHorizontalBlock"]:first-of-type button {
    background: #f5f0e4 !important;
    color: #2d4a14 !important;

    border: 1px solid #b8d090 !important;
    border-radius: 10px !important;

    font-size: 11px !important;
    font-weight: 600 !important;

    padding: 6px 10px !important;
    height: 32px !important;
    min-height: 32px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 6px !important;

    box-shadow: none !important;
    transition: all 0.15s ease !important;

    white-space: nowrap !important;
}

/* hover */
div[data-testid="stHorizontalBlock"]:first-of-type button:hover {
    background: #e3d9b6 !important;
    border-color: #8aac58 !important;
    transform: translateY(-1px);
}

/* active page */
div[data-testid="stHorizontalBlock"]:first-of-type button[data-testid="baseButton-primary"] {
    background: #2d4a14 !important;
    color: #e8f4d0 !important;
    border-color: #2d4a14 !important;
}

/* REMOVE ugly Streamlit inner padding */
div[data-testid="stHorizontalBlock"] button p {
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}
/* HERO */
.hero {
    background: linear-gradient(135deg, #1e3209 0%, #2d4a14 45%, #4a6b2a 100%);
    padding: 60px 56px; border-radius: 24px; color: #fff;
    margin-bottom: 2rem; box-shadow: 0 14px 40px rgba(20,40,5,.28);
    position: relative; overflow: hidden;
}
.hero-grid-bg {
    position: absolute; inset: 0;
    background-image: linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
                      linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 36px 36px;
    border-radius: 24px;
}
.hero::after {
    content: ''; position: absolute; right: -40px; top: -40px;
    width: 300px; height: 300px; background: rgba(255,255,255,.04); border-radius: 50%;
}
.hero-eye   { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #a8d478; margin-bottom: 12px; font-weight: 700; font-family: 'Space Mono', monospace; position: relative; }
.hero-title { font-family: 'DM Serif Display', serif; font-size: 46px; line-height: 1.1; margin-bottom: 14px; max-width: 600px; position: relative; }
.hero-sub   { font-size: 15.5px; color: #c8e6a0; line-height: 1.75; max-width: 560px; margin-bottom: 30px; position: relative; }
.hero-pills { display: flex; gap: 10px; flex-wrap: wrap; position: relative; }
.hero-pill  { background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.22); border-radius: 999px; padding: 6px 15px; font-size: 11px; font-weight: 600; color: #e8f4d0; font-family: 'Space Mono', monospace; transition: background 0.2s; }
.hero-pill:hover { background: rgba(255,255,255,.2); }

/* STATUS STRIP */
.sstrip { display: flex; gap: 10px; margin-bottom: 2rem; flex-wrap: wrap; }
.spill  { background: #fff; border: 1.5px solid #c8c4b8; border-radius: 999px; padding: 6px 14px; font-size: 11.5px; font-weight: 600; color: #3a4e2a; display: flex; align-items: center; gap: 6px; font-family: 'Space Mono', monospace; }
.dot    { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.dg { background: #4caf50; box-shadow: 0 0 6px #4caf5066; } .dy { background: #ff9800; } .dx { background: #9e9e9e; }

/* SECTION HEADER */
.sh       { margin-bottom: 1.4rem; }
.sh-eye   { font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #5a8a28; font-weight: 700; margin-bottom: 4px; font-family: 'Space Mono', monospace; }
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

/* ── HOME FEATURE CARDS (new) ── */
.fc-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.fc {
    background: #fff;
    border: 1.5px solid #d0ccbf;
    border-radius: 16px;
    padding: 20px 18px 22px;
    position: relative;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    min-height: 170px;
}
.fc::before {
    content: '';
    position: absolute; inset: 0;
    background: linear-gradient(145deg, rgba(107,154,58,0.045) 0%, transparent 60%);
    opacity: 0;
    transition: opacity 0.25s ease;
    border-radius: 16px;
}
.fc:hover { border-color: #7ab84a; transform: translateY(-3px); box-shadow: 0 8px 28px rgba(45,74,20,.12), 0 2px 8px rgba(45,74,20,.06); }
.fc:hover::before { opacity: 1; }

.fc-scan-line {
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, #6b9a3a, transparent);
    transform: scaleX(0); transform-origin: left;
    transition: transform 0.4s ease;
}
.fc:hover .fc-scan-line { transform: scaleX(1); }

.fc-corner {
    position: absolute; bottom: 10px; right: 10px;
    width: 18px; height: 18px; opacity: 0; transition: opacity 0.2s ease;
}
.fc:hover .fc-corner { opacity: 0.35; }

.fc-icon-wrap {
    width: 44px; height: 44px; border-radius: 12px;
    border: 1.5px solid rgba(107,154,58,0.35);
    background: rgba(107,154,58,0.08);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 13px;
    transition: border-color 0.2s, background 0.2s;
}
.fc:hover .fc-icon-wrap { border-color: rgba(107,154,58,0.7); background: rgba(107,154,58,0.14); }
.fc-icon-wrap svg { width: 22px; height: 22px; }

.fc-tag {
    position: absolute; top: 14px; right: 14px;
    font-family: 'Space Mono', monospace; font-size: 9.5px;
    color: rgba(107,154,58,0.6); letter-spacing: 0.5px;
    opacity: 0; transition: opacity 0.25s;
}
.fc:hover .fc-tag { opacity: 1; }

.fc-title { font-family: 'DM Serif Display', serif; font-size: 17px; color: #1e2e10; margin-bottom: 0; line-height: 1.2; }
.fc-text  { font-size: 12.5px; color: #3a5020; line-height: 1.65; }

.fc-divider {
    width: 24px; height: 1.5px; background: rgba(107,154,58,0.3);
    margin: 10px 0 9px; border-radius: 2px;
    transition: width 0.3s ease, background 0.2s;
}
.fc:hover .fc-divider { width: 38px; background: rgba(107,154,58,0.7); }

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

/* Remove bordered containers from st.columns blocks (dashboard cards, segmentation strip) */
[data-testid="stHorizontalBlock"]:not(:first-of-type) {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

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

PAGES = ["Home", "Calibration", "Diagnosis", "Sensors", "Dashboard", "Logs", "Train"]

if "_page" not in st.session_state:
    st.session_state["_page"] = "Home"
page = st.session_state["_page"]

# ── Logo + nav buttons in one horizontal row ──
_NAV_ICONS = {
    "Home":        '<path d="M3 10.5L12 3l9 7.5V20a1 1 0 01-1 1H5a1 1 0 01-1-1v-9.5z" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/><path d="M9 21V12h6v9" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "Calibration": '<circle cx="12" cy="12" r="3" stroke="currentColor" stroke-width="1.6" fill="none"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "Diagnosis":   '<path d="M9 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2h-3" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" fill="none"/><rect x="9" y="1" width="6" height="4" rx="1" stroke="currentColor" stroke-width="1.6" fill="none"/><path d="M9 12h6M9 16h4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "Sensors":     '<path d="M12 2C8.5 2 6 5 6 8.5c0 4 6 11 6 11s6-7 6-11C18 5 15.5 2 12 2z" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round"/><circle cx="12" cy="8.5" r="2" stroke="currentColor" stroke-width="1.4" fill="none"/>',
    "Dashboard":   '<rect x="3" y="3" width="7" height="7" rx="1.5" stroke="currentColor" stroke-width="1.6" fill="none"/><rect x="14" y="3" width="7" height="7" rx="1.5" stroke="currentColor" stroke-width="1.6" fill="none"/><rect x="3" y="14" width="7" height="7" rx="1.5" stroke="currentColor" stroke-width="1.6" fill="none"/><path d="M14 17.5h7M17.5 14v7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "Logs":        '<path d="M4 6h16M4 10h16M4 14h10M4 18h7" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>',
    "Train":       '<path d="M12 2l2.4 7.4H22l-6.2 4.5 2.4 7.4L12 17l-6.2 4.3 2.4-7.4L2 9.4h7.6L12 2z" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/>',
}

_logo_col, _sep_col, *_nav_cols = st.columns([1.6, 0.08] + [1]*len(PAGES), gap="small")

with _logo_col:
    st.markdown("""<div style="display:flex;align-items:center;gap:8px;padding:4px 0;font-family:'DM Serif Display',serif;font-size:15px;color:#2d4a14;font-weight:600;white-space:nowrap">
        <svg viewBox="0 0 24 24" width="17" height="17" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2C8 6 4 9 4 13a8 8 0 0016 0c0-4-4-7-8-11z" fill="#4a7a1e" opacity="0.9"/>
          <path d="M12 2C12 8 15 12 15 16" stroke="#6b9a3a" stroke-width="1.2" stroke-linecap="round" opacity="0.8"/>
          <path d="M12 10C10 12 9 14 9 16" stroke="#6b9a3a" stroke-width="1" stroke-linecap="round" opacity="0.6"/>
        </svg>
        LeafGuard
    </div>""", unsafe_allow_html=True)

with _sep_col:
    st.markdown('<div style="width:1px;height:20px;background:#c0bba8;margin:auto"></div>', unsafe_allow_html=True)

for col, p in zip(_nav_cols, PAGES):
    with col:
        is_active = (page == p)
        icon_color = "#e8f4d0" if is_active else "#3a6020"
        icon_svg = f'<svg viewBox="0 0 24 24" width="13" height="13" fill="none" style="display:inline-block;vertical-align:middle;margin-right:5px;margin-bottom:1px" xmlns="http://www.w3.org/2000/svg"><g stroke="{icon_color}" fill="none">{_NAV_ICONS[p]}</g></svg>'
        # Inject the SVG into the button label via a zero-width wrapper markdown
        # Since st.button doesn't support HTML, we overlay an SVG via CSS ::before trick
        # The simplest reliable approach: use a data attribute on a wrapper div
        st.markdown(f'<div class="nav-btn-wrap" data-page="{p}" data-active="{str(is_active).lower()}">', unsafe_allow_html=True)
        if st.button(p, key=f"nav_{p}", use_container_width=True,
                     type="primary" if is_active else "secondary"):
            st.session_state["_page"] = p
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


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


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────
if page == "Home":
    last = st.session_state.last_diagnosis
    st.markdown(f"""
    <div class="hero">
        <div class="hero-grid-bg"></div>
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

    # ── NEW: 3-column science card grid ──
    st.markdown("""
    <div class="fc-grid">

      <div class="fc">
        <div class="fc-scan-line"></div>
        <div class="fc-tag">MOD-01</div>
        <div class="fc-icon-wrap">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 3C9.5 3 7.5 4.8 7 7.2C6 7.7 5 8.8 5 10.5C5 12.5 6.5 14 8 14.3V16H16V14.3C17.5 14 19 12.5 19 10.5C19 8.8 18 7.7 17 7.2C16.5 4.8 14.5 3 12 3Z" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M9 14V19M15 14V19" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M10 19H14" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="12" cy="10" r="1.5" fill="#6b9a3a" opacity="0.6"/>
            <path d="M10 8.5C10.5 7.8 11.2 7.5 12 7.5" stroke="#6b9a3a" stroke-width="1" stroke-linecap="round" opacity="0.5"/>
          </svg>
        </div>
        <div class="fc-title">Diagnose Leaf</div>
        <div class="fc-divider"></div>
        <div class="fc-text">Segmentation and stress classification from a single leaf photo.</div>
        <svg class="fc-corner" viewBox="0 0 18 18" fill="none">
          <path d="M2 16L16 2M10 2H16V8" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <div class="fc">
        <div class="fc-scan-line"></div>
        <div class="fc-tag">MOD-02</div>
        <div class="fc-icon-wrap">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="4" y="14" width="3" height="6" rx="1" fill="#6b9a3a" opacity="0.5"/>
            <rect x="8.5" y="10" width="3" height="10" rx="1" fill="#6b9a3a" opacity="0.65"/>
            <rect x="13" y="6" width="3" height="14" rx="1" fill="#6b9a3a" opacity="0.8"/>
            <rect x="17.5" y="3" width="3" height="17" rx="1" fill="#4a7a1e"/>
            <path d="M3 20H22" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="18" cy="3.5" r="1" fill="#4a7a1e"/>
          </svg>
        </div>
        <div class="fc-title">Severity Analysis</div>
        <div class="fc-divider"></div>
        <div class="fc-text">Infected area percentage with a four-tier severity classification system.</div>
        <svg class="fc-corner" viewBox="0 0 18 18" fill="none">
          <path d="M2 16L16 2M10 2H16V8" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <div class="fc">
        <div class="fc-scan-line"></div>
        <div class="fc-tag">MOD-03</div>
        <div class="fc-icon-wrap">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M9 3H6L5 7H4L3 12H21L20 7H19L18 3H15" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M3 12V19C3 19.6 3.4 20 4 20H20C20.6 20 21 19.6 21 19V12" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M9 3C9 3 10 5 12 5C14 5 15 3 15 3" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <circle cx="12" cy="15.5" r="2" stroke="#6b9a3a" stroke-width="1.2" opacity="0.7"/>
            <path d="M12 13.5V12.5M12 18.5V17.5M10 15.5H9M15 15.5H14" stroke="#6b9a3a" stroke-width="1" stroke-linecap="round" opacity="0.5"/>
          </svg>
        </div>
        <div class="fc-title">Sensor Simulation</div>
        <div class="fc-divider"></div>
        <div class="fc-text">Adjust pH, EC, water level, temperature and humidity to match your system.</div>
        <svg class="fc-corner" viewBox="0 0 18 18" fill="none">
          <path d="M2 16L16 2M10 2H16V8" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <div class="fc">
        <div class="fc-scan-line"></div>
        <div class="fc-tag">MOD-04</div>
        <div class="fc-icon-wrap">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect x="3" y="4" width="18" height="13" rx="2" stroke="#4a7a1e" stroke-width="1.5"/>
            <path d="M6 14L8.5 11L11 13L14 8.5L17 10.5" stroke="#6b9a3a" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <circle cx="17" cy="10.5" r="1.2" fill="#4a7a1e"/>
            <path d="M8 20H16M12 17V20" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
          </svg>
        </div>
        <div class="fc-title">Decision Dashboard</div>
        <div class="fc-divider"></div>
        <div class="fc-text">Image and sensor fusion with actuator state recommendations.</div>
        <svg class="fc-corner" viewBox="0 0 18 18" fill="none">
          <path d="M2 16L16 2M10 2H16V8" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <div class="fc">
        <div class="fc-scan-line"></div>
        <div class="fc-tag">MOD-05</div>
        <div class="fc-icon-wrap">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <circle cx="12" cy="10" r="6" stroke="#4a7a1e" stroke-width="1.5"/>
            <path d="M12 4V10L15.5 7" stroke="#6b9a3a" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round" opacity="0.7"/>
            <path d="M6 16L4 21M18 16L20 21" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M5 18H19" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>
            <path d="M9 21H15" stroke="#4a7a1e" stroke-width="1.3" stroke-linecap="round" opacity="0.5"/>
          </svg>
        </div>
        <div class="fc-title">Calibration</div>
        <div class="fc-divider"></div>
        <div class="fc-text">Find the optimal daily image capture window for your grow environment.</div>
        <svg class="fc-corner" viewBox="0 0 18 18" fill="none">
          <path d="M2 16L16 2M10 2H16V8" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

      <div class="fc">
        <div class="fc-scan-line"></div>
        <div class="fc-tag">MOD-06</div>
        <div class="fc-icon-wrap">
          <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2L13.8 8.2H20L14.9 12L16.7 18.2L12 14.4L7.3 18.2L9.1 12L4 8.2H10.2L12 2Z" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M12 2V14.4" stroke="#6b9a3a" stroke-width="1" stroke-linecap="round" opacity="0.4" stroke-dasharray="1.5 2"/>
            <circle cx="12" cy="10" r="1.5" fill="#4a7a1e" opacity="0.6"/>
          </svg>
        </div>
        <div class="fc-title">Train Model</div>
        <div class="fc-divider"></div>
        <div class="fc-text">Upload your dataset, train the k-NN classifier, and review LOO accuracy.</div>
        <svg class="fc-corner" viewBox="0 0 18 18" fill="none">
          <path d="M2 16L16 2M10 2H16V8" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </div>

    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CALIBRATION
# ─────────────────────────────────────────────
elif page == "Calibration":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Setup</div>
        <div class="sh-title">Image Calibration</div>
        <div class="sh-desc">Upload images taken at different times of day to find the optimal capture window for consistent, reliable diagnosis.</div>
    </div>""", unsafe_allow_html=True)

    bt = st.session_state.best_cal_time

    # ── Top strip: how-it-works pills ──
    st.markdown("""<div style="display:flex;gap:8px;flex-wrap:wrap;margin-bottom:18px">""", unsafe_allow_html=True)
    for step, desc in [
        ("01", "Name files with hour, e.g. plant_08.jpg"),
        ("02", "Scored on brightness, contrast & sharpness"),
        ("03", "Best time window is recommended"),
        ("04", "Schedule daily capture at that time"),
    ]:
        st.markdown(f"""<div style="display:flex;align-items:center;gap:10px;background:#fff;border:1.5px solid #d0ccbf;border-radius:10px;padding:10px 16px;font-size:12.5px;color:#3a5020;flex:1;min-width:200px">
            <span style="font-family:'Space Mono',monospace;font-size:9px;font-weight:700;color:#6b9a3a;background:#eaf4d6;border-radius:5px;padding:3px 7px;flex-shrink:0">{step}</span>
            <span style="line-height:1.4">{desc}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Upload panel (full width) ──
    st.markdown("""<div class="page-panel">
    <div class="page-panel-title">Upload Images</div>
    <div class="info-box">Include the hour in each filename — e.g. <strong>plant_08.jpg</strong> — so the system can map quality scores to time of day.</div>""", unsafe_allow_html=True)
    files = st.file_uploader(
        "Select images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if files:
        # Preview + best window side by side
        prev_col, info_col = st.columns([6, 4], gap="large")

        with prev_col:
            st.markdown(f"""<div class="page-panel">
            <div class="page-panel-title">Preview — {len(files)} image(s)</div>""", unsafe_allow_html=True)
            prev = st.columns(min(3, len(files)))
            for i, f in enumerate(files[:6]):
                img = load_image_rgb(f.read())
                if img is not None:
                    prev[i % 3].markdown('<div class="imgcard">', unsafe_allow_html=True)
                    prev[i % 3].image(img, use_container_width=True)
                    prev[i % 3].markdown(f'<div class="imgcard-lbl">{f.name[:20]}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.button("Run Calibration Analysis", type="primary", use_container_width=True, key="run_cal")

        with info_col:
            if bt:
                st.markdown(f"""<div class="page-panel">
                <div class="page-panel-title">Best Window</div>
                <div style="text-align:center;padding:18px 0 10px">
                    <div style="font-size:9px;font-family:'Space Mono',monospace;letter-spacing:2px;text-transform:uppercase;color:#5a8a28;margin-bottom:6px">Optimal Capture Time</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:52px;color:#2d4a14;line-height:1">{bt}</div>
                    <div style="font-size:12px;color:#6b7b5a;margin-top:6px">Highest combined quality score</div>
                </div>
                </div>
                <div class="rec">
                    <div class="rec-lbl">Recommendation</div>
                    <div class="rec-txt">Schedule your capture at <strong>{bt}</strong> for the most reliable results.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="page-panel" style="text-align:center;padding:32px 24px">
                <div class="page-panel-title">Best Window</div>
                <div style="font-size:13px;color:#9e9e9e;line-height:1.7;margin-top:8px">Click <strong style="color:#3a5020">Run Calibration Analysis</strong><br>to find your optimal capture time.</div>
                </div>""", unsafe_allow_html=True)

        if st.session_state.get("run_cal"):
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
                res_col, chart_col = st.columns([4, 6], gap="large")
                with res_col:
                    st.markdown('<div class="page-panel"><div class="page-panel-title">Results Table</div>', unsafe_allow_html=True)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                with chart_col:
                    st.markdown('<div class="page-panel"><div class="page-panel-title">Quality Score Chart</div>', unsafe_allow_html=True)
                    fig = plot_calibration_scores(df["Image"].tolist(), df["Quality Score"].tolist(), best_i)
                    st.pyplot(fig, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                st.rerun()


# ─────────────────────────────────────────────
# DIAGNOSIS
# ─────────────────────────────────────────────
elif page == "Diagnosis":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Core Pipeline</div>
        <div class="sh-title">Leaf Diagnosis</div>
        <div class="sh-desc">Upload a leaf photo to run the full quality check, segmentation, feature extraction, and stress classification pipeline.</div>
    </div>""", unsafe_allow_html=True)

    # ── Upload full width — always shown ──
    st.markdown("""<div class="page-panel">
    <div class="page-panel-title">Upload Leaf Image</div>
    <div class="info-box">Supported formats: JPG, PNG, JFIF. Results persist in memory — navigate to <strong>Dashboard</strong> after diagnosis without losing data.</div>""", unsafe_allow_html=True)
    uploaded = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png", "jfif"], label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Only re-run pipeline when a NEW image is uploaded ──
    if uploaded:
        _img_id = f"{uploaded.name}_{uploaded.size}"
        if st.session_state.get("_last_img_id") != _img_id:
            original = load_image_rgb(uploaded.read())
            if original is not None:
                with st.spinner("Analysing image..."):
                    decision, quality, feats, pred, masks = _run_pipeline(original)
                st.session_state["_last_img_id"]  = _img_id
                st.session_state["_last_original"] = original
                st.session_state["_last_quality"]  = quality
                st.session_state["_last_masks"]    = masks
                if decision is not None:
                    st.session_state.last_diagnosis = decision
                    st.session_state.last_features  = feats
                    st.session_state.last_pred      = pred

    # ── Read persisted results ──
    _quality  = st.session_state.get("_last_quality")
    _original = st.session_state.get("_last_original")
    _masks    = st.session_state.get("_last_masks")
    d = st.session_state.last_diagnosis
    f = st.session_state.last_features or {}
    p = st.session_state.last_pred or {}

    if _quality is not None and _original is not None:

        # ── Quality chips: full width ──
        st.markdown(f"""<div class="page-panel">
        <div class="page-panel-title">Image Quality Report</div>
        <div class="metric-row">
            <div class="metric-chip">
                <div class="metric-chip-lbl">Sharpness</div>
                <div class="metric-chip-val" style="font-size:16px">{_quality.blur_label}</div>
                <div class="metric-chip-sub">{_quality.blur_score:.0f} score</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-lbl">Brightness</div>
                <div class="metric-chip-val" style="font-size:16px">{_quality.brightness:.0f}</div>
                <div class="metric-chip-sub">mean intensity</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-lbl">Contrast</div>
                <div class="metric-chip-val" style="font-size:16px">{_quality.contrast:.0f}</div>
                <div class="metric-chip-sub">std deviation</div>
            </div>
            <div class="metric-chip">
                <div class="metric-chip-lbl">HF Energy</div>
                <div class="metric-chip-val" style="font-size:16px">{_quality.high_freq_energy:.2f}</div>
                <div class="metric-chip-sub">freq energy</div>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

        if not _quality.passed:
            st.error("Image does not meet minimum quality requirements for reliable diagnosis.")
            for iss in _quality.issues:
                st.warning(iss)
        else:
            # ── 4 segmentation images in one compact row ──
            st.markdown('<div class="page-panel-title" style="font-family:\'Space Mono\',monospace;font-size:10px;letter-spacing:2.5px;text-transform:uppercase;color:#5a8a28;font-weight:700;margin-bottom:12px;display:flex;align-items:center;gap:8px">Segmentation Pipeline<span style="flex:1;height:1px;background:#e0ddd4;display:block"></span></div>', unsafe_allow_html=True)
            sc1, sc2, sc3, sc4 = st.columns(4, gap="small")
            for col, arr, lbl in [
                (sc1, _original,                                           "Original"),
                (sc2, _masks["proc"],                                      "Preprocessed"),
                (sc3, cv2.cvtColor(_masks["leaf"],  cv2.COLOR_GRAY2RGB),  "Leaf Mask"),
                (sc4, cv2.cvtColor(_masks["stress"], cv2.COLOR_GRAY2RGB), "Stress Mask"),
            ]:
                col.markdown('<div class="imgcard">', unsafe_allow_html=True)
                col.image(arr, use_container_width=True)
                col.markdown(f'<div class="imgcard-lbl">{lbl}</div></div>', unsafe_allow_html=True)

            # ── Highlighted image + results side by side ──
            img_col, res_col = st.columns([5, 4], gap="large")

            with img_col:
                st.markdown('<div class="page-panel"><div class="page-panel-title">Highlighted Stress Regions</div>', unsafe_allow_html=True)
                st.image(_masks["hi"], use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with res_col:
                if d:
                    sev_pct   = d["severity_pct"]
                    sev_color = _sev_color(d["severity_class"])
                    alert_cls = "cr" if d["risk_level"] == "high" else ("cy" if d["risk_level"] == "moderate" else "cg")

                    st.markdown(f"""<div class="page-panel">
                    <div class="page-panel-title">Diagnosis Result</div>
                    <div style="margin-bottom:14px">
                        <div class="card-lbl">Condition</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:20px;color:#1e2e10;margin-top:2px">{d['label_display']}</div>
                        <div style="font-size:12px;color:#4a5e3a;margin-top:3px">Confidence: {d['confidence']*100:.1f}%</div>
                    </div>
                    <div>
                        <div class="card-lbl">Severity</div>
                        <div style="font-family:'DM Serif Display',serif;font-size:32px;color:{sev_color};line-height:1">{sev_pct:.1f}%</div>
                        <div class="sevwrap" style="margin-top:6px"><div class="sevbar" style="width:{min(sev_pct,100):.1f}%;background:{sev_color}"></div></div>
                        <div style="font-size:11px;color:#3a5020;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-top:5px">{d['severity_class'].capitalize()}</div>
                    </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown(f"""<div class="card {alert_cls}">
                        <div class="card-lbl">Risk Level</div>
                        <div class="card-val" style="font-size:17px">{_risk_icon(d['risk_level'])} {d['risk_level'].capitalize()}</div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown(f"""<div class="page-panel">
                    <div class="page-panel-title">Leaf Metrics</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                        <div class="metric-chip">
                            <div class="metric-chip-lbl">Leaf Area</div>
                            <div class="metric-chip-val" style="font-size:15px">{f.get('leaf_area',0):,}</div>
                            <div class="metric-chip-sub">pixels</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-lbl">Stress Area</div>
                            <div class="metric-chip-val" style="font-size:15px;color:{sev_color}">{f.get('stress_area',0):,}</div>
                            <div class="metric-chip-sub">pixels</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-lbl">Regions</div>
                            <div class="metric-chip-val" style="font-size:15px">{f.get('num_regions',0)}</div>
                            <div class="metric-chip-sub">detected</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-lbl">Largest</div>
                            <div class="metric-chip-val" style="font-size:15px">{f.get('largest_region_area',0):,}</div>
                            <div class="metric-chip-sub">px region</div>
                        </div>
                    </div>
                    </div>""", unsafe_allow_html=True)

                    st.markdown("""<div class="rec" style="margin-top:4px">
                        <div class="rec-lbl">Next Step</div>
                        <div class="rec-txt" style="font-size:13px;font-family:'DM Sans',sans-serif">Go to <strong>Dashboard</strong> for actuator states, sensor fusion, and session logging.</div>
                    </div>""", unsafe_allow_html=True)

                    with st.expander("Class Probabilities"):
                        for cls, prob in sorted(p.get("all_probs", {}).items(), key=lambda x: -x[1]):
                            st.markdown(f"`{cls}`")
                            st.progress(float(prob), text=f"{prob*100:.1f}%")
                else:
                    st.markdown("""<div class="page-panel">
                    <div class="page-panel-title">No Diagnosis</div>
                    <div style="text-align:center;padding:24px 0">
                        <div style="font-size:13px;color:#9e9e9e;line-height:1.7">Image did not pass quality check.<br>No classification available.</div>
                    </div>
                    </div>""", unsafe_allow_html=True)

    else:
        # No image ever uploaded yet
        st.markdown("""<div class="page-panel"><div class="page-panel-title">Pipeline Overview</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;padding:4px 0 8px">""", unsafe_allow_html=True)
        for icon_path, step_title, step_desc in [
            ('<path d="M4 16l4-8 4 6 3-4 5 6" stroke="#4a7a1e" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" fill="none"/><rect x="3" y="3" width="18" height="18" rx="2" stroke="#4a7a1e" stroke-width="1.5" fill="none"/>',
             "Upload", "Provide a clear leaf image in JPG or PNG format"),
            ('<circle cx="12" cy="12" r="4" stroke="#4a7a1e" stroke-width="1.5" fill="none"/><path d="M3 12h3M18 12h3M12 3v3M12 18v3" stroke="#4a7a1e" stroke-width="1.5" stroke-linecap="round"/>',
             "Quality Check", "Sharpness, brightness and contrast are verified"),
            ('<path d="M12 3C8 3 5 6 5 10c0 5 7 11 7 11s7-6 7-11c0-4-3-7-7-7z" stroke="#4a7a1e" stroke-width="1.5" fill="none"/><path d="M8 10c1-2 2-3 4-3" stroke="#6b9a3a" stroke-width="1.2" stroke-linecap="round" opacity="0.7"/>',
             "Segmentation", "Leaf area and stress regions are isolated"),
            ('<path d="M9 12l2 2 4-4" stroke="#4a7a1e" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/><rect x="3" y="3" width="18" height="18" rx="2" stroke="#4a7a1e" stroke-width="1.5" fill="none"/>',
             "Classification", "Stress type, severity and risk level are reported"),
        ]:
            st.markdown(f"""<div style="background:#f5f0e4;border:1.5px solid #d0ccbf;border-radius:12px;padding:14px 12px;text-align:center">
                <div style="display:flex;align-items:center;justify-content:center;width:34px;height:34px;border-radius:9px;border:1.5px solid rgba(107,154,58,0.35);background:rgba(107,154,58,0.08);margin:0 auto 8px">
                    <svg viewBox="0 0 24 24" width="17" height="17" xmlns="http://www.w3.org/2000/svg">{icon_path}</svg>
                </div>
                <div style="font-family:'DM Serif Display',serif;font-size:12.5px;color:#1e2e10;margin-bottom:4px">{step_title}</div>
                <div style="font-size:10.5px;color:#5a7040;line-height:1.45">{step_desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)



# ─────────────────────────────────────────────
# SENSORS
# ─────────────────────────────────────────────
elif page == "Sensors":
    st.markdown("""<div class="sh">
        <div class="sh-eye">Environment</div>
        <div class="sh-title">Sensor Panel</div>
        <div class="sh-desc">Configure your hydroponic sensor readings. These values are fused with image analysis to produce the final treatment decision.</div>
    </div>""", unsafe_allow_html=True)

    s = st.session_state.sensors
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown('<div class="page-panel" style="padding:28px 28px 24px"><div class="page-panel-title">Nutrient & Water</div>', unsafe_allow_html=True)
        ph  = st.slider("pH Level", 4.0, 8.0, float(s["ph"]), 0.1)
        ec  = st.slider("EC (mS/cm)", 0.5, 3.5, float(s["ec"]), 0.1)
        wl  = st.selectbox("Water Level", ["full", "normal", "low", "empty"],
                           index=["full", "normal", "low", "empty"].index(s["water_level"]))
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="page-panel" style="padding:28px 28px 24px"><div class="page-panel-title">Climate & Light</div>', unsafe_allow_html=True)
        tmp = st.slider("Temperature (°C)", 10.0, 40.0, float(s["temperature"]), 0.5)
        hum = st.slider("Humidity (%)", 20.0, 95.0, float(s["humidity"]), 1.0)
        lgt = st.selectbox("Light Intensity", ["low", "medium", "high"],
                           index=["low", "medium", "high"].index(s["light"]))
        st.markdown('</div>', unsafe_allow_html=True)

    st.session_state.sensors = dict(ph=ph, ec=ec, water_level=wl,
                                    temperature=tmp, humidity=hum, light=lgt)

    def _bc(v, lo, hi): return "b-ok" if lo <= v <= hi else "b-wa"
    def _wc(v): return {"full": "b-ok", "normal": "b-ok", "low": "b-wa", "empty": "b-ba"}.get(v, "b-ok")

    ph_lbl  = "Optimal" if 5.5 <= ph  <= 6.5 else "Out of range"
    ec_lbl  = "Optimal" if 1.2 <= ec  <= 2.4 else ("Low" if ec < 1.2 else "High")
    tmp_lbl = "Optimal" if 18  <= tmp <= 28   else "Out of range"
    hum_lbl = "Optimal" if 50  <= hum <= 80   else "Out of range"

    with right:
        st.markdown(f"""<div class="page-panel" style="padding:28px 28px 24px">
        <div class="page-panel-title">Live Readings</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:4px">
            <div class="metric-chip" style="padding:16px 18px">
                <div class="metric-chip-lbl">pH</div>
                <div class="metric-chip-val">{ph:.1f}</div>
                <div class="metric-chip-sub" style="color:{'#2e7d32' if ph_lbl=='Optimal' else '#b71c1c'}">{ph_lbl}</div>
            </div>
            <div class="metric-chip" style="padding:16px 18px">
                <div class="metric-chip-lbl">EC</div>
                <div class="metric-chip-val">{ec:.1f}</div>
                <div class="metric-chip-sub" style="color:{'#2e7d32' if ec_lbl=='Optimal' else '#b71c1c'}">{ec_lbl}</div>
            </div>
            <div class="metric-chip" style="padding:16px 18px">
                <div class="metric-chip-lbl">Temperature</div>
                <div class="metric-chip-val">{tmp:.1f}°</div>
                <div class="metric-chip-sub" style="color:{'#2e7d32' if tmp_lbl=='Optimal' else '#b71c1c'}">{tmp_lbl}</div>
            </div>
            <div class="metric-chip" style="padding:16px 18px">
                <div class="metric-chip-lbl">Humidity</div>
                <div class="metric-chip-val">{hum:.0f}%</div>
                <div class="metric-chip-sub" style="color:{'#2e7d32' if hum_lbl=='Optimal' else '#b71c1c'}">{hum_lbl}</div>
            </div>
        </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="stbl" style="padding:26px 28px">
            <div style="font-family:'DM Serif Display',serif;font-size:17px;color:#1e2e10;margin-bottom:14px">Full Summary</div>
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
                <span class="srow-v">{tmp:.1f} °C<span class="sb {_bc(tmp,18,28)}">{tmp_lbl}</span></span>
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


# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────
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
        c1, c2, c3, c4 = st.columns(4, gap="medium")
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

        st.markdown("<div style='margin-bottom:1.5rem'></div>", unsafe_allow_html=True)
        ll, rr = st.columns(2, gap="large")

        with ll:
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
            st.markdown(f"""<div class="stbl" style="margin-top:4px">
                <div style="font-size:11px;font-weight:700;color:#3a5020;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px">Active Sensor Readings</div>
                <div class="srow"><span class="srow-k">pH</span><span class="srow-v">{sv['ph']:.1f}</span></div>
                <div class="srow"><span class="srow-k">EC</span><span class="srow-v">{sv['ec']:.1f} mS/cm</span></div>
                <div class="srow"><span class="srow-k">Water Level</span><span class="srow-v">{sv['water_level'].capitalize()}</span></div>
                <div class="srow"><span class="srow-k">Temperature</span><span class="srow-v">{sv['temperature']:.1f} C</span></div>
            </div>""", unsafe_allow_html=True)

        with rr:
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


# ─────────────────────────────────────────────
# LOGS
# ─────────────────────────────────────────────
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


# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
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