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
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="CardioScan AI | ระบบประเมินความเสี่ยงโรคหัวใจ",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ฟังก์ชันช่วยล้างช่องว่างหน้าบรรทัด ป้องกัน Streamlit แปลง HTML เป็น Code Block
def render_html(html_str: str):
    cleaned = "\n".join(line.strip() for line in html_str.strip().splitlines())
    st.markdown(cleaned, unsafe_allow_html=True)

# ฟังก์ชันแปลงรูปภาพในโฟลเดอร์ ./img เป็น Base64 เพื่อจัดกรอบสวยๆ แบบไร้ขอบ
@st.cache_data
def get_img_base64(path: str) -> str:
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

# โหลดข้อมูลและเทรนโมเดล (Cache ไว้เพื่อให้เปลี่ยนค่า Slider ได้ลื่นไหลไม่สะดุด)
@st.cache_data
def load_dataset():
    return pd.read_csv("./data/Heart3.csv")

@st.cache_resource
def train_knn_model(df: pd.DataFrame):
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ใช้ k=7, weights='distance' และ metric='manhattan' เพื่อให้ได้ค่าความน่าจะเป็นที่ละเอียดและแม่นยำ (~83.5%)
    model = KNeighborsClassifier(n_neighbors=7, weights="distance", metric="manhattan")
    model.fit(X_scaled, y)
    cv_acc = cross_val_score(model, X_scaled, y, cv=5).mean() * 100
    return model, scaler, X.columns.tolist(), cv_acc

dt = load_dataset()
knn_model, scaler, feature_cols, model_accuracy = train_knn_model(dt)

img_yes_b64 = get_img_base64("./img/yes.jpg")
img_no_b64 = get_img_base64("./img/no.jpg")

# ---------------------------------------------------------
# 2. CUSTOM DESIGN SYSTEM (CSS)
# ---------------------------------------------------------
render_html("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Thai:wght@300;400;500;600;700&family=Plus+Jakarta+Sans:wght@500;600;700;800&display=swap');

    :root {
        --bg-canvas: #F6F8FA;
        --bg-surface: #FFFFFF;
        --border-subtle: #E2E8F0;
        --text-primary: #0F172A;
        --text-secondary: #475569;
        --text-muted: #64748B;
        --accent-crimson: #BE123C;
        --accent-rose: #E11D48;
        --accent-emerald: #059669;
        --accent-amber: #D97706;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'IBM Plex Sans Thai', 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        background-color: var(--bg-canvas);
        color: var(--text-primary);
    }

    /* ซ่อนแถบเมนูส่วนเกินของ Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {background: transparent !important;}

    .block-container {
        max-width: 1220px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* Editorial Top Header */
    .hero-shell {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 28px 34px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03), 0 8px 24px rgba(15, 23, 42, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 24px;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--accent-crimson);
        background: #FFF1F2;
        border: 1px solid #FFE4E6;
        padding: 5px 12px;
        border-radius: 999px;
        margin-bottom: 10px;
    }

    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: var(--accent-rose);
        border-radius: 50%;
        display: inline-block;
    }

    .hero-title {
        font-size: 32px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        line-height: 1.25;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        font-size: 15px;
        color: var(--text-secondary);
        margin: 6px 0 0 0;
    }

    .hero-badges {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
    }

    .meta-pill {
        background: #F8FAFC;
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 12px 18px;
        min-width: 135px;
    }

    .meta-pill-label {
        font-size: 12px;
        color: var(--text-muted);
        margin-bottom: 2px;
    }

    .meta-pill-value {
        font-family: 'Plus Jakarta Sans', 'IBM Plex Sans Thai', sans-serif;
        font-size: 18px;
        font-weight: 700;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    /* Reference Status Cards */
    .ref-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 14px;
        display: flex;
        align-items: center;
        gap: 16px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .ref-card:hover {
        transform: translateY(-2px);
    }

    .ref-card.danger:hover { border-color: #FDA4AF; }
    .ref-card.safe:hover { border-color: #6EE7B7; }

    .ref-thumb {
        width: 76px;
        height: 76px;
        border-radius: 12px;
        object-fit: cover;
        flex-shrink: 0;
        border: 1px solid #F1F5F9;
    }

    .status-tag {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.05em;
        padding: 3px 9px;
        border-radius: 6px;
        display: inline-block;
        margin-bottom: 4px;
    }

    .tag-danger { background: #FFF1F2; color: #BE123C; }
    .tag-safe { background: #ECFDF5; color: #047857; }

    .ref-title {
        font-size: 17px;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
    }

    .ref-desc {
        font-size: 13px;
        color: var(--text-muted);
        margin: 2px 0 0 0;
    }

    /* Section Group Headers inside Form */
    .form-group-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 18px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #E2E8F0;
    }

    .step-badge {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 11px;
        font-weight: 700;
        color: #FFFFFF;
        background: #0F172A;
        padding: 3px 8px;
        border-radius: 6px;
    }

    .group-title {
        font-size: 16px;
        font-weight: 600;
        color: var(--text-primary);
    }

    /* Styling Streamlit Widgets */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px !important;
        border-color: #CBD5E1 !important;
        background-color: #FFFFFF !important;
    }

    /* Primary Action Button */
    div.stButton > button {
        background: #0F172A !important;
        color: #FFFFFF !important;
        font-family: 'IBM Plex Sans Thai', sans-serif !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        border: 1px solid #0F172A !important;
        border-radius: 12px !important;
        padding: 13px 24px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15) !important;
        transition: all 0.2s ease !important;
    }

    div.stButton > button:hover {
        background: #BE123C !important;
        border-color: #BE123C !important;
        box-shadow: 0 6px 18px rgba(190, 18, 60, 0.28) !important;
        transform: translateY(-1px);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 5px;
        border-radius: 12px;
        width: fit-content;
        margin-bottom: 18px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px;
        padding: 8px 20px;
        font-weight: 600;
        font-size: 14px;
        color: #475569;
        background: transparent;
        border: none;
    }

    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.08);
    }

    /* Diagnostic Result Card */
    .diag-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 20px;
        padding: 26px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
    }

    .factor-chip {
        display: inline-block;
        font-size: 12.5px;
        font-weight: 500;
        padding: 5px 11px;
        border-radius: 8px;
        margin: 4px 4px 0 0;
    }

    .chip-warn {
        background: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
    }

    .chip-ok {
        background: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
    }

    /* KPI Stat Cards */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
    }

    .kpi-label {
        font-size: 13px;
        color: var(--text-muted);
        margin-bottom: 6px;
    }

    .kpi-val {
        font-family: 'Plus Jakarta Sans', 'IBM Plex Sans Thai', sans-serif;
        font-size: 26px;
        font-weight: 700;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
    }

    .kpi-sub {
        font-size: 12px;
        margin-top: 4px;
        font-weight: 500;
    }
</style>
""")

# ---------------------------------------------------------
# 3. EDITORIAL HEADER & REFERENCE CLASS CARDS
# ---------------------------------------------------------
render_html(f"""
<div class="hero-shell">
    <div>
        <div class="hero-kicker">
            <span class="pulse-dot"></span>
            CARDIOVASCULAR CLINICAL ANALYTICS
        </div>
        <h1 class="hero-title">โปรเจคการตรวจโรคหัวใจ</h1>
        <p class="hero-subtitle">ระบบวิเคราะห์ความเสี่ยงภาวะโรคหัวใจจากพารามิเตอร์ทางคลินิก 11 รายการ ด้วยโมเดล K-Nearest Neighbors</p>
    </div>
    <div class="hero-badges">
        <div class="meta-pill">
            <div class="meta-pill-label">ฐานข้อมูลผู้ป่วย</div>
            <div class="meta-pill-value">{len(dt):,} ราย</div>
        </div>
        <div class="meta-pill">
            <div class="meta-pill-label">ความแม่นยำโมเดล</div>
            <div class="meta-pill-value" style="color: #059669;">{model_accuracy:.1f}%</div>
        </div>
    </div>
</div>
""")

ref_col1, ref_col2 = st.columns(2)

with ref_col1:
    yes_img_tag = f'<img src="data:image/jpeg;base64,{img_yes_b64}" class="ref-thumb" alt="Heart Disease">' if img_yes_b64 else ""
    render_html(f"""
    <div class="ref-card danger">
        {yes_img_tag}
        <div>
            <span class="status-tag tag-danger">CLASS 1 • POSITIVE RISK</span>
            <h3 class="ref-title">เป็นโรคหัวใจ</h3>
            <p class="ref-desc">พบความผิดปกติของหลอดเลือดหรือคลื่นไฟฟ้าหัวใจ มีแนวโน้มเสี่ยงสูง</p>
        </div>
    </div>
    """)

with ref_col2:
    no_img_tag = f'<img src="data:image/jpeg;base64,{img_no_b64}" class="ref-thumb" alt="Healthy Heart">' if img_no_b64 else ""
    render_html(f"""
    <div class="ref-card safe">
        {no_img_tag}
        <div>
            <span class="status-tag tag-safe">CLASS 0 • NORMAL STATE</span>
            <h3 class="ref-title">ไม่เป็นโรคหัวใจ</h3>
            <p class="ref-desc">การทำงานของหัวใจและสัญญาณชีพอยู่ในเกณฑ์ปกติ ความเสี่ยงต่ำ</p>
        </div>
    </div>
    """)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. MAIN WORKSPACE TABS
# ---------------------------------------------------------
tab_predict, tab_analytics = st.tabs([
    "วินิจฉัยความเสี่ยง (Risk Assessment)",
    "สถิติและฐานข้อมูลคลินิก (Clinical Dataset)"
])

# =========================================================
# TAB 1: RISK ASSESSMENT WORKSPACE
# =========================================================
with tab_predict:
    form_col, result_col = st.columns([1.35, 1], gap="large")

    with form_col:
        # หมวดที่ 1
        render_html("""
        <div class="form-group-header">
            <span class="step-badge">01</span>
            <span class="group-title">ข้อมูลทั่วไปของผู้รับการตรวจ (Demographics)</span>
        </div>
        """)
        c1, c2 = st.columns(2)
        with c1:
            age = st.slider("อายุ (Age - ปี)", min_value=20, max_value=100, value=52)
        with c2:
            sex = st.selectbox(
                "เพศ (Sex)",
                options=[1, 0],
                format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)"
            )

        # หมวดที่ 2
        render_html("""
        <div class="form-group-header">
            <span class="step-badge">02</span>
            <span class="group-title">สัญญาณชีพและผลตรวจทางห้องปฏิบัติการ (Vitals & Lab)</span>
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
            max_hr = st.slider("อัตราการเต้นหัวใจสูงสุด (MaxHR - bpm)", min_value=60, max_value=220, value=142)

        # หมวดที่ 3
        render_html("""
        <div class="form-group-header">
            <span class="step-badge">03</span>
            <span class="group-title">อาการทางคลินิกและคลื่นไฟฟ้าหัวใจ (Symptoms & ECG)</span>
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
            oldpeak = st.number_input("ค่า Oldpeak (ST Depression)", min_value=-3.0, max_value=7.0, value=0.5, step=0.1)

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        predict_clicked = st.button("วิเคราะห์ความเสี่ยงโรคหัวใจ", use_container_width=True)

    # ฝั่งขวา: แผงแสดงผลการวินิจฉัย
    with result_col:
        st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

        if predict_clicked:
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

            # กำหนดสีและระดับความเสี่ยง
            if risk_pct >= 65:
                tier_label = "ความเสี่ยงสูง (HIGH RISK)"
                ring_color = "#E11D48"
                badge_bg = "#FFF1F2"
                badge_text = "#BE123C"
                summary_title = "ตรวจพบแนวโน้มเป็นโรคหัวใจ"
                result_img = img_yes_b64
            elif risk_pct >= 40:
                tier_label = "ความเสี่ยงปานกลาง (MODERATE RISK)"
                ring_color = "#D97706"
                badge_bg = "#FFFBEB"
                badge_text = "#B45309"
                summary_title = "ควรเฝ้าระวังและตรวจเพิ่มเติม" if pred == 1 else "ความเสี่ยงก้ำกึ่ง ควรติดตามอาการ"
                result_img = img_yes_b64 if pred == 1 else img_no_b64
            else:
                tier_label = "ความเสี่ยงต่ำ (LOW RISK)"
                ring_color = "#059669"
                badge_bg = "#ECFDF5"
                badge_text = "#047857"
                summary_title = "ไม่พบความเสี่ยงเป็นโรคหัวใจ"
                result_img = img_no_b64

            # คำนวณเส้นรอบวง SVG Radial Gauge (r = 58 -> circumference = 2 * pi * 58 = 364.42)
            circumference = 364.42
            dash_offset = circumference * (1 - (risk_pct / 100.0))

            # วิเคราะห์ปัจจัยเสี่ยงเด่นจากค่าที่ผู้ใช้กรอก
            risk_factors = []
            normal_factors = []

            if resting_bp >= 140:
                risk_factors.append(f"ความดันสูง ({resting_bp} mmHg)")
            else:
                normal_factors.append(f"ความดันปกติ ({resting_bp} mmHg)")

            if cholesterol >= 240:
                risk_factors.append(f"คอเลสเตอรอลสูง ({cholesterol} mg/dl)")
            elif cholesterol > 0:
                normal_factors.append(f"คอเลสเตอรอล ({cholesterol} mg/dl)")

            if exercise_angina == 1:
                risk_factors.append("เจ็บหน้าอกเมื่อออกแรง")
            if fasting_bs == 1:
                risk_factors.append("น้ำตาลในเลือดสูง (>120)")
            if st_slope in [1, 2]:
                risk_factors.append("กราฟคลื่น ST ผิดปกติ")
            else:
                normal_factors.append("กราฟคลื่น ST ปกติ (Up)")
            if max_hr < 120:
                risk_factors.append(f"อัตราหัวใจสูงสุดต่ำ ({max_hr} bpm)")

            chips_html = "".join([f'<span class="factor-chip chip-warn">• {f}</span>' for f in risk_factors])
            chips_html += "".join([f'<span class="factor-chip chip-ok">✓ {f}</span>' for f in normal_factors[:2]])

            res_img_html = (
                f'<img src="data:image/jpeg;base64,{result_img}" style="width:68px;height:68px;border-radius:14px;object-fit:cover;border:1px solid #E2E8F0;" />'
                if result_img else ""
            )

            render_html(f"""
            <div class="diag-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:12px; margin-bottom:18px;">
                    <div>
                        <span style="background:{badge_bg}; color:{badge_text}; font-size:11px; font-weight:700; padding:4px 10px; border-radius:6px; letter-spacing:0.04em;">
                            {tier_label}
                        </span>
                        <h3 style="margin:8px 0 2px 0; font-size:21px; font-weight:700; color:#0F172A;">
                            {summary_title}
                        </h3>
                        <p style="margin:0; font-size:13px; color:#64748B;">
                            ผลลัพธ์จากโมเดล KNN (Class {pred})
                        </p>
                    </div>
                    {res_img_html}
                </div>

                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:16px; padding:22px; text-align:center; margin-bottom:18px;">
                    <svg width="150" height="150" viewBox="0 0 140 140" style="transform: rotate(-90deg);">
                        <circle cx="70" cy="70" r="58" stroke="#E2E8F0" stroke-width="11" fill="none" />
                        <circle cx="70" cy="70" r="58" stroke="{ring_color}" stroke-width="11" fill="none"
                                stroke-dasharray="{circumference}" stroke-dashoffset="{dash_offset:.2f}"
                                stroke-linecap="round" />
                    </svg>
                    <div style="margin-top:-104px; margin-bottom:44px;">
                        <div style="font-family:'Plus Jakarta Sans',sans-serif; font-size:32px; font-weight:800; color:#0F172A; line-height:1;">
                            {risk_pct:.1f}%
                        </div>
                        <div style="font-size:12px; color:#64748B; margin-top:4px;">
                            โอกาสเป็นโรคหัวใจ
                        </div>
                    </div>
                    <div style="display:flex; justify-content:space-around; border-top:1px solid #E2E8F0; padding-top:12px; font-size:13px;">
                        <div>
                            <span style="color:#64748B;">โอกาสเป็นโรคหัวใจ:</span>
                            <strong style="color:#BE123C;">{risk_pct:.1f}%</strong>
                        </div>
                        <div>
                            <span style="color:#64748B;">โอกาสปกติ:</span>
                            <strong style="color:#059669;">{safe_pct:.1f}%</strong>
                        </div>
                    </div>
                </div>

                <div>
                    <div style="font-size:13px; font-weight:600; color:#334155; margin-bottom:6px;">
                        ปัจจัยทางคลินิกที่ตรวจพบจากข้อมูลนำเข้า
                    </div>
                    <div>{chips_html}</div>
                </div>
            </div>
            """)
        else:
            # สถานะเริ่มต้นก่อนกดปุ่มทำนาย
            render_html(f"""
            <div class="diag-card" style="text-align:center; padding:36px 26px;">
                <div style="width:54px; height:54px; border-radius:14px; background:#FFF1F2; color:#E11D48; display:inline-flex; align-items:center; justify-content:center; margin-bottom:14px;">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M22 12h-4l-3 9L9 3l-3 9H2"/>
                    </svg>
                </div>
                <h3 style="margin:0 0 8px 0; font-size:19px; font-weight:700; color:#0F172A;">
                    พร้อมวิเคราะห์ข้อมูลสุขภาพ
                </h3>
                <p style="margin:0 0 20px 0; font-size:14px; color:#64748B; line-height:1.6;">
                    ปรับค่าพารามิเตอร์ทางคลินิกทั้ง 11 รายการทางด้านซ้าย แล้วกดปุ่ม <strong>"วิเคราะห์ความเสี่ยงโรคหัวใจ"</strong> เพื่อคำนวณเปอร์เซ็นต์ความเสี่ยงจากฐานข้อมูลผู้ป่วย {len(dt):,} ราย
                </p>
                <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px; text-align:left; font-size:13px; color:#475569;">
                    <div style="font-weight:600; color:#0F172A; margin-bottom:6px;">ค่าที่เลือกไว้ปัจจุบัน:</div>
                    <div>• อายุ <strong>{age} ปี</strong> | เพศ <strong>{"ชาย" if sex == 1 else "หญิง"}</strong></div>
                    <div>• ความดัน <strong>{resting_bp} mmHg</strong> | คอเลสเตอรอล <strong>{cholesterol} mg/dl</strong></div>
                    <div>• อัตราหัวใจสูงสุด <strong>{max_hr} bpm</strong> | Oldpeak <strong>{oldpeak}</strong></div>
                </div>
            </div>
            """)

# =========================================================
# TAB 2: CLINICAL DATASET & VISUALIZATION
# =========================================================
with tab_analytics:
    total_cases = len(dt)
    heart_yes = int((dt["HeartDisease"] == 1).sum())
    heart_no = int((dt["HeartDisease"] == 0).sum())
    avg_age = dt["Age"].mean()

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-label">จำนวนผู้ป่วยในฐานข้อมูล</div>
            <div class="kpi-val">{total_cases:,}</div>
            <div class="kpi-sub" style="color:#64748B;">ไฟล์ Heart3.csv (11 Features)</div>
        </div>
        """)
    with k2:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-label">กลุ่มเป็นโรคหัวใจ (Class 1)</div>
            <div class="kpi-val" style="color:#BE123C;">{heart_yes:,}</div>
            <div class="kpi-sub" style="color:#E11D48;">คิดเป็น {heart_yes/total_cases*100:.1f}% ของทั้งหมด</div>
        </div>
        """)
    with k3:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-label">กลุ่มปกติ (Class 0)</div>
            <div class="kpi-val" style="color:#059669;">{heart_no:,}</div>
            <div class="kpi-sub" style="color:#10B981;">คิดเป็น {heart_no/total_cases*100:.1f}% ของทั้งหมด</div>
        </div>
        """)
    with k4:
        render_html(f"""
        <div class="kpi-card">
            <div class="kpi-label">อายุเฉลี่ยผู้รับการตรวจ</div>
            <div class="kpi-val">{avg_age:.1f} ปี</div>
            <div class="kpi-sub" style="color:#64748B;">ช่วงอายุ {dt['Age'].min()} - {dt['Age'].max()} ปี</div>
        </div>
        """)

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    vis_col, table_col = st.columns([1, 1.35], gap="large")

    with vis_col:
        render_html("""
        <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:6px;">
            การกระจายตัวของกลุ่มตัวอย่าง (Class Distribution)
        </div>
        <p style="font-size:13px; color:#64748B; margin-bottom:14px;">
            เปรียบเทียบจำนวนผู้ป่วยโรคหัวใจและผู้ที่มีสุขภาพหัวใจปกติในชุดข้อมูล
        </p>
        """)

        if st.button("แสดงการจินตทัศน์ข้อมูล", use_container_width=True):
            plt.rcParams["font.sans-serif"] = ["Tahoma", "Leelawadee UI", "DejaVu Sans"]
            fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=150)
            fig.patch.set_facecolor("#FFFFFF")
            ax.set_facecolor("#FFFFFF")

            categories = ["Normal (0)\nไม่เป็นโรคหัวใจ", "Heart Disease (1)\nเป็นโรคหัวใจ"]
            counts = [heart_no, heart_yes]
            colors = ["#10B981", "#E11D48"]

            bars = ax.bar(categories, counts, color=colors, width=0.48, zorder=3)
            ax.grid(axis="y", linestyle="--", alpha=0.4, color="#CBD5E1", zorder=0)
            ax.set_ylim(0, max(counts) * 1.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#CBD5E1")
            ax.spines["bottom"].set_color("#CBD5E1")
            ax.tick_params(colors="#334155", labelsize=10)
            ax.set_ylabel("จำนวนผู้ป่วย (ราย)", fontsize=10, color="#475569")

            for bar, count in zip(bars, counts):
                pct = count / total_cases * 100
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 14,
                    f"{count:,} ราย ({pct:.1f}%)",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="#0F172A"
                )

            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with table_col:
        render_html("""
        <div style="font-size:16px; font-weight:700; color:#0F172A; margin-bottom:6px;">
            ตัวอย่างข้อมูลทางคลินิก (Heart3.csv)
        </div>
        <p style="font-size:13px; color:#64748B; margin-bottom:14px;">
            แสดงรายการข้อมูล 15 แถวแรกจากทั้งหมด 918 รายการที่ใช้สอนโมเดล
        </p>
        """)
        st.dataframe(dt.head(15), use_container_width=True, height=340)