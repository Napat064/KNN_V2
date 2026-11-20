import base64
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ---------------------------------------------------------
# 1. PAGE CONFIG & CORE HELPERS
# ---------------------------------------------------------
st.set_page_config(
    page_title="CARDIO.LAB — ระบบวิเคราะห์ความเสี่ยงโรคหัวใจ",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def render_html(html_str: str):
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

@st.cache_data
def get_img_base64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

@st.cache_data
def load_dataset():
    return pd.read_csv("./data/Heart3.csv")

@st.cache_resource
def train_knn_model(df: pd.DataFrame):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")
    model.fit(X_scaled, y)
    cv_acc = cross_val_score(model, X_scaled, y, cv=5).mean() * 100
    return model, scaler, X.columns.tolist(), cv_acc

dt = load_dataset()
knn_model, scaler, feature_cols, model_accuracy = train_knn_model(dt)

img_yes_b64 = get_img_base64("./img/yes.jpg")
img_no_b64 = get_img_base64("./img/no.jpg")

# ---------------------------------------------------------
# 2. WORLD-CLASS BIOMETRIC DARK DESIGN SYSTEM (CSS)
# ---------------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=JetBrains+Mono:wght@500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --bg-void: #080A0F;
        --bg-surface: #111520;
        --bg-elevated: #171C2B;
        --border-hairline: rgba(255, 255, 255, 0.08);
        --border-strong: rgba(255, 255, 255, 0.16);
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --emerald: #10B981;
        --crimson: #F43F5E;
        --amber: #F59E0B;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'IBM Plex Sans Thai', 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        background: radial-gradient(circle at 12% 0%, #161E31 0%, #090C12 48%, #06080C 100%) !important;
        color: var(--text-primary) !important;
    }

    #MainMenu, footer {visibility: hidden;}
    header {background: transparent !important;}

    .block-container {
        max-width: 1260px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }

    /* Typography & Labels inside Streamlit Widgets */
    label, [data-testid="stWidgetLabel"] p {
        color: #E2E8F0 !important;
        font-size: 13.5px !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em;
    }

    /* Inputs & Selectboxes Dark Styling */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #151A27 !important;
        border: 1px solid rgba(255, 255, 255, 0.11) !important;
        border-radius: 12px !important;
        color: #F8FAFC !important;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    div[data-baseweb="select"] > div:hover,
    div[data-baseweb="input"] > div:hover {
        border-color: rgba(255, 255, 255, 0.28) !important;
    }

    div[data-baseweb="select"] span,
    div[data-baseweb="input"] input {
        color: #F8FAFC !important;
    }

    div[data-baseweb="input"] input {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="input"] button {
        background-color: #1E2536 !important;
        color: #E2E8F0 !important;
        border-left: 1px solid rgba(255, 255, 255, 0.08) !important;
    }

    ul[data-baseweb="menu"] {
        background-color: #151A27 !important;
        border: 1px solid rgba(255, 255, 255, 0.14) !important;
        border-radius: 12px !important;
    }

    li[role="option"] {
        color: #E2E8F0 !important;
    }

    li[role="option"]:hover {
        background-color: #222A3D !important;
    }

    /* Animations */
    @keyframes ecgDash {
        0% { stroke-dashoffset: 240; }
        100% { stroke-dashoffset: 0; }
    }

    @keyframes pulseBeacon {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.45; transform: scale(1.35); }
    }

    .ecg-line {
        stroke-dasharray: 240;
        animation: ecgDash 2.8s linear infinite;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #F43F5E;
        box-shadow: 0 0 12px #F43F5E;
        animation: pulseBeacon 1.8s ease-in-out infinite;
        display: inline-block;
    }

    /* Top Navigation / Editorial Hero */
    .bento-hero {
        background: linear-gradient(135deg, rgba(23, 28, 43, 0.92) 0%, rgba(14, 18, 28, 0.96) 100%);
        border: 1px solid var(--border-hairline);
        border-radius: 24px;
        padding: 28px 34px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 24px;
        margin-bottom: 22px;
        position: relative;
        overflow: hidden;
    }

    .bento-hero::before {
        content: "";
        position: absolute;
        top: -80px;
        right: -60px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(244, 63, 94, 0.14) 0%, transparent 70%);
        pointer-events: none;
    }

    .eyebrow-pill {
        display: inline-flex;
        align-items: center;
        gap: 9px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11.5px;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #FDA4AF;
        background: rgba(244, 63, 94, 0.12);
        border: 1px solid rgba(244, 63, 94, 0.28);
        padding: 5px 13px;
        border-radius: 999px;
        margin-bottom: 12px;
    }

    .main-heading {
        font-size: 34px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        letter-spacing: -0.02em;
        line-height: 1.2;
    }

    .main-sub {
        font-size: 14.5px;
        color: #94A3B8;
        margin: 8px 0 0 0;
    }

    .telemetry-strip {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .tele-box {
        background: rgba(9, 12, 18, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 13px 18px;
        min-width: 132px;
        backdrop-filter: blur(10px);
    }

    .tele-label {
        font-size: 11.5px;
        color: #64748B;
        margin-bottom: 3px;
    }

    .tele-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: #F8FAFC;
    }

    /* Cinematic Dual Portrait Showcase */
    .portrait-card {
        position: relative;
        border-radius: 22px;
        overflow: hidden;
        height: 235px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        background: #111520;
        box-shadow: 0 14px 34px rgba(0, 0, 0, 0.4);
        transition: transform 0.3s cubic-bezier(0.22, 1, 0.36, 1), border-color 0.3s ease;
    }

    .portrait-card:hover {
        transform: translateY(-4px);
    }

    .portrait-card.healthy:hover {
        border-color: rgba(16, 185, 129, 0.55);
        box-shadow: 0 18px 40px rgba(16, 185, 129, 0.16);
    }

    .portrait-card.risk:hover {
        border-color: rgba(244, 63, 94, 0.55);
        box-shadow: 0 18px 40px rgba(244, 63, 94, 0.18);
    }

    .portrait-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 22%;
        display: block;
        filter: contrast(1.04) saturate(1.06);
    }

    .portrait-scrim {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(8, 10, 15, 0.08) 25%, rgba(8, 10, 15, 0.88) 100%);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 18px 22px;
    }

    .glass-badge {
        align-self: flex-start;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 5px 12px;
        border-radius: 999px;
        backdrop-filter: blur(12px);
    }

    .badge-healthy {
        background: rgba(16, 185, 129, 0.22);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.45);
    }

    .badge-risk {
        background: rgba(244, 63, 94, 0.24);
        color: #FDA4AF;
        border: 1px solid rgba(244, 63, 94, 0.48);
    }

    .portrait-title {
        font-size: 21px;
        font-weight: 700;
        color: #FFFFFF;
        margin: 0;
        text-shadow: 0 2px 10px rgba(0, 0, 0, 0.6);
    }

    .portrait-caption {
        font-size: 13px;
        color: #CBD5E1;
        margin: 3px 0 0 0;
    }

    /* Section Header inside Bento Form */
    .section-divider {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin: 18px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .section-left {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .index-pill {
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 700;
        color: #38BDF8;
        background: rgba(56, 189, 248, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.25);
        padding: 2px 8px;
        border-radius: 6px;
    }

    .section-name {
        font-size: 15.5px;
        font-weight: 600;
        color: #F1F5F9;
    }

    /* Primary CTA Button */
    div.stButton > button {
        background: linear-gradient(135deg, #E11D48 0%, #BE123C 100%) !important;
        color: #FFFFFF !important;
        font-family: 'IBM Plex Sans Thai', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        border-radius: 14px !important;
        padding: 14px 24px !important;
        box-shadow: 0 10px 28px rgba(225, 29, 72, 0.35) !important;
        transition: all 0.22s ease !important;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%) !important;
        box-shadow: 0 14px 34px rgba(244, 63, 94, 0.5) !important;
        transform: translateY(-2px);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #111520;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 5px;
        border-radius: 14px;
        width: fit-content;
        margin-bottom: 18px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 13.5px;
        color: #94A3B8;
        background: transparent;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1E2538 !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
    }

    /* Result Console Card */
    .result-console {
        background: linear-gradient(180deg, #131825 0%, #0E121C 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        overflow: hidden;
        box-shadow: 0 24px 54px rgba(0, 0, 0, 0.5);
    }

    .result-hero-photo {
        position: relative;
        height: 215px;
        width: 100%;
        overflow: hidden;
    }

    .result-hero-photo img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        object-position: center 20%;
    }

    .result-hero-overlay {
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(14, 18, 28, 0.1) 20%, #131825 100%);
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        padding: 18px 24px;
    }

    .result-body {
        padding: 6px 24px 24px 24px;
    }

    .gauge-panel {
        background: rgba(8, 10, 15, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 18px;
        padding: 18px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 16px;
    }

    .chip-item {
        display: inline-block;
        font-size: 12px;
        font-weight: 500;
        padding: 5px 11px;
        border-radius: 8px;
        margin: 4px 5px 0 0;
    }

    .chip-danger {
        background: rgba(244, 63, 94, 0.14);
        color: #FDA4AF;
        border: 1px solid rgba(244, 63, 94, 0.3);
    }

    .chip-safe {
        background: rgba(16, 185, 129, 0.14);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    /* Benchmark Bar */
    .bench-row {
        margin-top: 10px;
    }

    .bench-head {
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        color: #94A3B8;
        margin-bottom: 4px;
    }

    .bench-track {
        height: 6px;
        background: #1E2536;
        border-radius: 999px;
        overflow: hidden;
    }

    .bench-fill {
        height: 100%;
        border-radius: 999px;
    }

    /* KPI Bento Cards */
    .kpi-bento {
        background: #111520;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 18px 20px;
    }

    .kpi-title {
        font-size: 12.5px;
        color: #94A3B8;
        margin-bottom: 6px;
    }

    .kpi-figure {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: #F8FAFC;
    }

    .kpi-foot {
        font-size: 12px;
        margin-top: 4px;
    }
</style>
""")

# ---------------------------------------------------------
# 3. HERO HEADER & DUAL PORTRAIT SHOWCASE
# ---------------------------------------------------------
render_html(f"""
<div class="bento-hero">
    <div>
        <div class="eyebrow-pill">
            <span class="live-dot"></span>
            CARDIO.LAB • KNN DIAGNOSTIC ENGINE
        </div>
        <h1 class="main-heading">โปรเจคการตรวจโรคหัวใจ</h1>
        <p class="main-sub">ประเมินความเสี่ยงภาวะโรคหัวใจแบบเรียลไทม์จากพารามิเตอร์ทางคลินิก 11 ค่า</p>
    </div>
    <div style="display:flex; align-items:center; gap:18px; flex-wrap:wrap;">
        <svg width="110" height="42" viewBox="0 0 120 44" fill="none">
            <path class="ecg-line" d="M2 24 H24 L31 14 L38 34 L46 6 L54 40 L61 20 L68 24 H118"
                  stroke="#F43F5E" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
        <div class="telemetry-strip">
            <div class="tele-box">
                <div class="tele-label">ฐานข้อมูลคลินิก</div>
                <div class="tele-num">{len(dt):,}</div>
            </div>
            <div class="tele-box">
                <div class="tele-label">ความแม่นยำโมเดล</div>
                <div class="tele-num" style="color:#10B981;">{model_accuracy:.1f}%</div>
            </div>
        </div>
    </div>
</div>
""")

p_col1, p_col2 = st.columns(2, gap="medium")

with p_col1:
    no_src = f"data:image/jpeg;base64,{img_no_b64}" if img_no_b64 else ""
    render_html(f"""
    <div class="portrait-card healthy">
        <img src="{no_src}" class="portrait-img" alt="Healthy State" />
        <div class="portrait-scrim">
            <span class="glass-badge badge-healthy">CLASS 0 • HEALTHY & STRONG</span>
            <div>
                <h3 class="portrait-title">ไม่เป็นโรคหัวใจ</h3>
                <p class="portrait-caption">สมรรถภาพหัวใจแข็งแรง สัญญาณชีพและคลื่นไฟฟ้าอยู่ในเกณฑ์ปกติ</p>
            </div>
        </div>
    </div>
    """)

with p_col2:
    yes_src = f"data:image/jpeg;base64,{img_yes_b64}" if img_yes_b64 else ""
    render_html(f"""
    <div class="portrait-card risk">
        <img src="{yes_src}" class="portrait-img" alt="Cardiac Risk State" />
        <div class="portrait-scrim">
            <span class="glass-badge badge-risk">CLASS 1 • CARDIAC RISK</span>
            <div>
                <h3 class="portrait-title">เป็นโรคหัวใจ</h3>
                <p class="portrait-caption">พบสัญญาณเตือนความเสี่ยงหลอดเลือดหัวใจหรืออาการเจ็บแน่นหน้าอก</p>
            </div>
        </div>
    </div>
    """)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. WORKSPACE TABS
# ---------------------------------------------------------
tab_predict, tab_analytics = st.tabs([
    "วิเคราะห์ความเสี่ยงรายบุคคล (Diagnostic Console)",
    "สถิติและข้อมูลคลินิก (Dataset Analytics)"
])

# =========================================================
# TAB 1: DIAGNOSTIC CONSOLE
# =========================================================
with tab_predict:
    left_form, right_console = st.columns([1.3, 1], gap="large")

    with left_form:
        # 01 Demographics
        render_html("""
        <div class="section-divider">
            <div class="section-left">
                <span class="index-pill">01</span>
                <span class="section-name">ข้อมูลพื้นฐานผู้รับการตรวจ (Demographics)</span>
            </div>
        </div>
        """)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("อายุ (Age - ปี)", min_value=20, max_value=100, value=50)
        with c2:
            sex = st.selectbox(
                "เพศ (Sex)",
                options=[1, 0],
                format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)"
            )

        # 02 Vitals & Blood Labs
        render_html("""
        <div class="section-divider">
            <div class="section-left">
                <span class="index-pill">02</span>
                <span class="section-name">สัญญาณชีพและผลเลือด (Vitals & Lab)</span>
            </div>
        </div>
        """)
        c3, c4 = st.columns(2)
        with c3:
            resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP - mmHg)", min_value=0, max_value=250, value=130)
            fasting_bs = st.selectbox(
                "น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)",
                options=[0, 1],
                format_func=lambda x: "ปกติ <= 120 mg/dl (0)" if x == 0 else "สูง > 120 mg/dl (1)"
            )
        with c4:
            cholesterol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol - mg/dl)", min_value=0, max_value=700, value=210)
            max_hr = st.slider("อัตราการเต้นหัวใจสูงสุด (MaxHR - bpm)", min_value=60, max_value=220, value=145)

        # 03 Symptoms & ECG
        render_html("""
        <div class="section-divider">
            <div class="section-left">
                <span class="index-pill">03</span>
                <span class="section-name">อาการทางคลินิกและคลื่นไฟฟ้าหัวใจ (Symptoms & ECG)</span>
            </div>
        </div>
        """)
        c5, c6 = st.columns(2)
        with c5:
            chest_pain = st.selectbox(
                "ประเภทอาการเจ็บหน้าอก (ChestPainType)",
                options=[1, 2, 3, 4],
                format_func=lambda x: {
                    1: "1 - ไม่มีอาการเจ็บหน้าอก (ASY)",
                    2: "2 - เจ็บหน้าอกแบบไม่คงที่ (ATA)",
                    3: "3 - เจ็บหน้าอกที่ไม่ใช่โรคหัวใจ (NAP)",
                    4: "4 - เจ็บหน้าอกแบบทั่วไป (TA)"
                }[x]
            )
            resting_ecg = st.selectbox(
                "คลื่นไฟฟ้าหัวใจขณะพัก (RestingECG)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - กล้ามเนื้อหัวใจห้องล่างซ้ายหนา (LVH)",
                    2: "2 - ปกติ (Normal)",
                    3: "3 - คลื่น ST-T ผิดปกติ (ST)"
                }[x]
            )
            st_slope = st.selectbox(
                "ความชันของคลื่น ST (ST_Slope)",
                options=[1, 2, 3],
                format_func=lambda x: {
                    1: "1 - ลาดลง (Down)",
                    2: "2 - ราบเรียบ (Flat)",
                    3: "3 - ลาดขึ้น (Up)"
                }[x]
            )
        with c6:
            exercise_angina = st.selectbox(
                "เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)",
                options=[0, 1],
                format_func=lambda x: "ไม่เจ็บ (0)" if x == 0 else "เจ็บ (1)"
            )
            oldpeak = st.number_input("ค่า Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=0.4, step=0.1)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        run_diagnosis = st.button("วิเคราะห์ความเสี่ยงโรคหัวใจ", use_container_width=True)

    # Right Console: Output & Probability Gauge
    with right_console:
        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

        if run_diagnosis:
            x_input = pd.DataFrame(
                [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
                  resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
                columns=feature_cols
            )
            x_input_scaled = scaler.transform(x_input)

            pred = knn_model.predict(x_input_scaled)[0]
            proba = knn_model.predict_proba(x_input_scaled)[0]
            risk_pct = proba[1] * 100
            safe_pct = proba[0] * 100

            if risk_pct >= 65:
                tier_tag = "HIGH CARDIAC RISK"
                accent_hex = "#F43F5E"
                pill_style = "background:rgba(244,63,94,0.25); color:#FDA4AF; border:1px solid rgba(244,63,94,0.5);"
                headline = "มีความเสี่ยงเป็นโรคหัวใจ"
                hero_photo_b64 = img_yes_b64
            elif risk_pct >= 40:
                tier_tag = "MODERATE RISK"
                accent_hex = "#F59E0B"
                pill_style = "background:rgba(245,158,11,0.25); color:#FCD34D; border:1px solid rgba(245,158,11,0.5);"
                headline = "ความเสี่ยงปานกลาง ควรติดตามอาการ" if pred == 1 else "ความเสี่ยงก้ำกึ่ง แนะนำตรวจเช็กเพิ่มเติม"
                hero_photo_b64 = img_yes_b64 if pred == 1 else img_no_b64
            else:
                tier_tag = "OPTIMAL / LOW RISK"
                accent_hex = "#10B981"
                pill_style = "background:rgba(16,185,129,0.25); color:#6EE7B7; border:1px solid rgba(16,185,129,0.5);"
                headline = "สุขภาพหัวใจปกติ ความเสี่ยงต่ำ"
                hero_photo_b64 = img_no_b64

            # SVG Gauge math (r=46 -> circumference = 289.03)
            circ = 289.03
            offset = circ * (1 - (risk_pct / 100.0))

            # Analyze Clinical Markers
            warn_list, ok_list = [], []
            if resting_bp >= 140:
                warn_list.append(f"ความดันสูง ({resting_bp} mmHg)")
            else:
                ok_list.append(f"ความดันปกติ ({resting_bp} mmHg)")

            if cholesterol >= 240:
                warn_list.append(f"คอเลสเตอรอลสูง ({cholesterol})")
            elif cholesterol > 0:
                ok_list.append(f"คอเลสเตอรอล ({cholesterol})")

            if exercise_angina == 1:
                warn_list.append("เจ็บหน้าอกเมื่อออกแรง")
            if fasting_bs == 1:
                warn_list.append("น้ำตาลในเลือดสูง")
            if st_slope in [1, 2]:
                warn_list.append("คลื่น ST ผิดปกติ")
            else:
                ok_list.append("คลื่น ST ปกติ (Up)")
            if max_hr < 120:
                warn_list.append(f"MaxHR ต่ำ ({max_hr} bpm)")

            chips_markup = "".join([f'<span class="chip-item chip-danger">• {w}</span>' for w in warn_list])
            chips_markup += "".join([f'<span class="chip-item chip-safe">✓ {o}</span>' for o in ok_list[:2]])

            bp_pct = min(int((resting_bp / 200) * 100), 100)
            hr_pct = min(int((max_hr / 210) * 100), 100)

            hero_photo_src = f"data:image/jpeg;base64,{hero_photo_b64}" if hero_photo_b64 else ""

            render_html(f"""
            <div class="result-console">
                <div class="result-hero-photo">
                    <img src="{hero_photo_src}" alt="Diagnosis Status" />
                    <div class="result-hero-overlay">
                        <span style="{pill_style} align-self:flex-start; font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; padding:4px 11px; border-radius:999px; backdrop-filter:blur(10px); margin-bottom:8px;">
                            {tier_tag} • CLASS {pred}
                        </span>
                        <h3 style="margin:0; font-size:22px; font-weight:700; color:#FFFFFF; text-shadow:0 2px 10px rgba(0,0,0,0.7);">
                            {headline}
                        </h3>
                    </div>
                </div>

                <div class="result-body">
                    <div class="gauge-panel">
                        <div style="position:relative; width:112px; height:112px; flex-shrink:0;">
                            <svg width="112" height="112" viewBox="0 0 112 112" style="transform: rotate(-90deg);">
                                <circle cx="56" cy="56" r="46" stroke="#1E2536" stroke-width="10" fill="none" />
                                <circle cx="56" cy="56" r="46" stroke="{accent_hex}" stroke-width="10" fill="none"
                                        stroke-dasharray="{circ}" stroke-dashoffset="{offset:.2f}" stroke-linecap="round" />
                            </svg>
                            <div style="position:absolute; inset:0; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                                <span style="font-family:'JetBrains Mono',monospace; font-size:21px; font-weight:700; color:#FFFFFF;">
                                    {risk_pct:.1f}%
                                </span>
                                <span style="font-size:10px; color:#94A3B8;">ความเสี่ยง</span>
                            </div>
                        </div>

                        <div style="flex:1;">
                            <div style="font-size:12px; color:#94A3B8; margin-bottom:4px;">โอกาสเกิดภาวะโรคหัวใจ</div>
                            <div style="font-family:'JetBrains Mono',monospace; font-size:28px; font-weight:700; color:{accent_hex}; line-height:1.1;">
                                {risk_pct:.1f}%
                            </div>
                            <div style="font-size:12.5px; color:#CBD5E1; margin-top:6px;">
                                โอกาสปกติ: <strong style="color:#10B981; font-family:'JetBrains Mono',monospace;">{safe_pct:.1f}%</strong>
                            </div>
                        </div>
                    </div>

                    <div style="margin-bottom:14px;">
                        <div style="font-size:12.5px; font-weight:600; color:#CBD5E1; margin-bottom:4px;">
                            ปัจจัยบ่งชี้จากค่าที่ตรวจวัด
                        </div>
                        <div>{chips_markup}</div>
                    </div>

                    <div style="border-top:1px solid rgba(255,255,255,0.08); padding-top:12px;">
                        <div class="bench-row">
                            <div class="bench-head">
                                <span>ความดันโลหิตขณะพัก (เกณฑ์ปกติ ~130)</span>
                                <span style="font-family:'JetBrains Mono',monospace; color:#F8FAFC;">{resting_bp} mmHg</span>
                            </div>
                            <div class="bench-track">
                                <div class="bench-fill" style="width:{bp_pct}%; background:{'#F43F5E' if resting_bp >= 140 else '#10B981'};"></div>
                            </div>
                        </div>
                        <div class="bench-row">
                            <div class="bench-head">
                                <span>อัตราการเต้นหัวใจสูงสุด (ค่าเฉลี่ยกลุ่มปกติ ~148)</span>
                                <span style="font-family:'JetBrains Mono',monospace; color:#F8FAFC;">{max_hr} bpm</span>
                            </div>
                            <div class="bench-track">
                                <div class="bench-fill" style="width:{hr_pct}%; background:#38BDF8;"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """)
        else:
            render_html(f"""
            <div class="result-console" style="padding: 34px 26px; text-align:center;">
                <div style="width:58px; height:58px; border-radius:16px; background:rgba(244,63,94,0.12); border:1px solid rgba(244,63,94,0.3); display:inline-flex; align-items:center; justify-content:center; margin-bottom:16px;">
                    <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#F43F5E" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                </div>
                <h3 style="margin:0 0 8px 0; font-size:20px; font-weight:700; color:#FFFFFF;">
                    คอนโซลพร้อมประมวลผล
                </h3>
                <p style="margin:0 0 22px 0; font-size:13.5px; color:#94A3B8; line-height:1.65;">
                    กำหนดค่าพารามิเตอร์ทั้ง 11 รายการทางฝั่งซ้าย แล้วกดปุ่ม <strong style="color:#F8FAFC;">"วิเคราะห์ความเสี่ยงโรคหัวใจ"</strong> ระบบจะคำนวณเปอร์เซ็นต์ความเสี่ยงจากฐานข้อมูล {len(dt):,} รายทันที
                </p>
                <div style="background:rgba(8,10,15,0.6); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:16px; text-align:left; font-size:13px; color:#CBD5E1;">
                    <div style="font-family:'JetBrains Mono',monospace; font-size:11px; color:#64748B; margin-bottom:8px;">CURRENT INPUT TELEMETRY</div>
                    <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span>อายุ / เพศ</span>
                        <strong style="font-family:'JetBrains Mono',monospace;">{age} ปี • {"ชาย" if sex == 1 else "หญิง"}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid rgba(255,255,255,0.05);">
                        <span>ความดัน / คอเลสเตอรอล</span>
                        <strong style="font-family:'JetBrains Mono',monospace;">{resting_bp} / {cholesterol}</strong>
                    </div>
                    <div style="display:flex; justify-content:space-between; padding:4px 0;">
                        <span>MaxHR / Oldpeak</span>
                        <strong style="font-family:'JetBrains Mono',monospace;">{max_hr} bpm • {oldpeak}</strong>
                    </div>
                </div>
            </div>
            """)

# =========================================================
# TAB 2: CLINICAL DATASET ANALYTICS
# =========================================================
with tab_analytics:
    total_cases = len(dt)
    heart_yes = int((dt["HeartDisease"] == 1).sum())
    heart_no = int((dt["HeartDisease"] == 0).sum())
    avg_age = dt["Age"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-bento">
            <div class="kpi-title">กลุ่มตัวอย่างทั้งหมด</div>
            <div class="kpi-figure">{total_cases:,}</div>
            <div class="kpi-foot" style="color:#64748B;">Heart3.csv • 11 Features</div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="kpi-bento">
            <div class="kpi-title">เป็นโรคหัวใจ (Class 1)</div>
            <div class="kpi-figure" style="color:#F43F5E;">{heart_yes:,}</div>
            <div class="kpi-foot" style="color:#FDA4AF;">{heart_yes/total_cases*100:.1f}% ของทั้งหมด</div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="kpi-bento">
            <div class="kpi-title">ไม่เป็นโรคหัวใจ (Class 0)</div>
            <div class="kpi-figure" style="color:#10B981;">{heart_no:,}</div>
            <div class="kpi-foot" style="color:#6EE7B7;">{heart_no/total_cases*100:.1f}% ของทั้งหมด</div>
        </div>
        """)
    with k4:
        render_html(f"""
        <div class="kpi-bento">
            <div class="kpi-title">อายุเฉลี่ยผู้รับการตรวจ</div>
            <div class="kpi-figure">{avg_age:.1f}</div>
            <div class="kpi-foot" style="color:#94A3B8;">ช่วงอายุ {dt['Age'].min()} – {dt['Age'].max()} ปี</div>
        </div>
        """)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    chart_col, data_col = st.columns([1, 1.35], gap="large")

    with chart_col:
        render_html("""
        <div style="font-size:16px; font-weight:700; color:#F8FAFC; margin-bottom:4px;">
            สัดส่วนข้อมูลกลุ่มปกติเทียบกับกลุ่มเสี่ยง
        </div>
        <p style="font-size:13px; color:#94A3B8; margin-bottom:14px;">
            แสดงจำนวนผู้ป่วยในแต่ละคลาสจากชุดข้อมูลฝึกสอนโมเดล KNN
        </p>
        """)

        if st.button("แสดงการจินตทัศน์ข้อมูล", use_container_width=True):
            plt.rcParams["font.sans-serif"] = ["Tahoma", "Leelawadee UI", "DejaVu Sans"]
            fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
            fig.patch.set_facecolor("#111520")
            ax.set_facecolor("#111520")

            labels = ["Normal (0)\nไม่เป็นโรคหัวใจ", "Heart Disease (1)\nเป็นโรคหัวใจ"]
            vals = [heart_no, heart_yes]
            bar_colors = ["#10B981", "#F43F5E"]

            bars = ax.bar(labels, vals, color=bar_colors, width=0.46, zorder=3)
            ax.grid(axis="y", linestyle="--", alpha=0.18, color="#94A3B8", zorder=0)
            ax.set_ylim(0, max(vals) * 1.22)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#334155")
            ax.spines["bottom"].set_color("#334155")
            ax.tick_params(colors="#CBD5E1", labelsize=10)
            ax.set_ylabel("จำนวนผู้ป่วย (ราย)", fontsize=10, color="#94A3B8")

            for bar, count in zip(bars, vals):
                pct = count / total_cases * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 14,
                    f"{count:,} ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#F8FAFC"
                )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with data_col:
        render_html("""
        <div style="font-size:16px; font-weight:700; color:#F8FAFC; margin-bottom:4px;">
            ตัวอย่างชุดข้อมูลทางคลินิก (Heart3.csv)
        </div>
        <p style="font-size:13px; color:#94A3B8; margin-bottom:14px;">
            แสดงข้อมูล 15 รายการแรกจากทั้งหมด 918 รายการ
        </p>
        """)
        st.dataframe(dt.head(15), use_container_width=True, height=340)