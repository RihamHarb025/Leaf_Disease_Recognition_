import streamlit as st

st.set_page_config(
    page_title="LeafGuard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #f7f3eb;
}

[data-testid="stHeader"] {
    background: transparent;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    padding-left: 4rem;
    padding-right: 4rem;
    max-width: 1400px;
}

.hero-box {
    background: linear-gradient(135deg, #799851, #8aa764);
    padding: 65px 55px;
    border-radius: 28px;
    color: white;
    margin-bottom: 50px;
    box-shadow: 0 12px 30px rgba(80, 100, 50, 0.14);
}

.hero-subtitle {
    font-size: 14px;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #f3f6ee;
    margin-bottom: 14px;
    font-weight: 700;
}

.hero-title {
    font-size: 54px;
    font-weight: 800;
    line-height: 1.12;
    margin-bottom: 18px;
    max-width: 760px;
}

.hero-text {
    font-size: 18px;
    color: #f7f8f2;
    line-height: 1.7;
    max-width: 700px;
    margin-bottom: 30px;
}

.hero-buttons {
    display: flex;
    gap: 16px;
    margin-top: 10px;
}

.hero-btn-primary,
.hero-btn-secondary {
    text-decoration: none;
    padding: 13px 24px;
    border-radius: 999px;
    font-size: 15px;
    font-weight: 700;
    display: inline-block;
    transition: 0.2s ease;
}

.hero-btn-primary {
    background: #9fb878;
    color: white !important;
    box-shadow: 0 6px 16px rgba(0,0,0,0.08);
}

.hero-btn-primary:hover {
    background: #91aa6c;
}

.hero-btn-secondary {
    background: rgba(255,255,255,0.16);
    color: white !important;
    border: 1px solid rgba(255,255,255,0.38);
}

.hero-btn-secondary:hover {
    background: rgba(255,255,255,0.24);
}

.section-title {
    font-size: 31px;
    font-weight: 800;
    color: #799851;
    margin-bottom: 10px;
}

.section-text {
    font-size: 16px;
    color: #5f6b63;
    margin-bottom: 28px;
}

.feature-card {
    background: #fffaf0;
    border: 2px solid #799851;
    border-radius: 24px;
    padding: 24px 22px;
    min-height: 210px;
    box-shadow: 0 8px 20px rgba(0,0,0,0.04);
    margin-bottom: 22px;
}

.feature-badge {
    width: 54px;
    height: 54px;
    border-radius: 16px;
    background: linear-gradient(135deg, #9fb878, #8faa67);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
    margin-bottom: 18px;
}

.feature-title {
    font-size: 24px;
    font-weight: 800;
    color: #799851;
    margin-bottom: 10px;
}

.feature-text {
    font-size: 15px;
    color: #54635a;
    line-height: 1.65;
    max-width: 95%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-box">
    <div class="hero-subtitle">Leaf Disease Monitoring</div>
    <div class="hero-title">Smart Leaf Disease Detection for Early Crop Monitoring</div>
    <div class="hero-text">
        Upload leaf images, detect diseased regions, estimate severity, and explore results
        through a clean interactive analysis experience designed for agricultural diagnosis.
    </div>
    <div class="hero-buttons">
        <a href="#" class="hero-btn-primary">Start Diagnosis</a>
        <a href="#" class="hero-btn-secondary">View Dashboard</a>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">Core Features</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-text">A clean overview of the main tools available in your leaf disease detection system.</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-badge">🍃</div>
        <div class="feature-title">Diagnose Leaf</div>
        <div class="feature-text">
            Upload a plant leaf image and detect visible diseased regions using your image processing pipeline.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-badge">📊</div>
        <div class="feature-title">Disease Insights</div>
        <div class="feature-text">
            View segmented regions, extracted information, and clear system interpretation for each analysis result.
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-badge">🩺</div>
        <div class="feature-title">Severity Analysis</div>
        <div class="feature-text">
            Measure how much of the leaf area is affected and classify the infection into severity levels.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-badge">📝</div>
        <div class="feature-title">Reports</div>
        <div class="feature-text">
            Organize results, compare analyses, and prepare simple summaries for evaluation and presentation.
        </div>
    </div>
    """, unsafe_allow_html=True)