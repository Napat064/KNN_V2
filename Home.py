from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import streamlit as st
import pandas as pd

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="โปรเจคการตรวจโรคหัวใจ",
    page_icon="❤️",
    layout="centered"
)

# ตกแต่งสไตล์รวมทั้งหน้าเว็บ (CSS)
st.markdown("""
<style>
    /* พื้นหลังหน้าเว็บโทนสว่างนุ่มตา */
    .stApp {
        background: linear-gradient(180deg, #FFF5F5 0%, #F4F6F9 100%);
    }

    /* กล่องการ์ดสีขาวมีเงา */
    .card-box {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 20px;
    }

    /* แถบหัวข้อส่วนต่างๆ */
    .section-banner-red {
        background: linear-gradient(90deg, #E74C3C 0%, #EC7063 100%);
        color: white;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 8px rgba(231, 76, 60, 0.25);
        margin: 20px 0 15px 0;
    }

    .section-banner-blue {
        background: linear-gradient(90deg, #0083B0 0%, #00B4DB 100%);
        color: white;
        padding: 14px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 3px 8px rgba(0, 180, 219, 0.25);
        margin: 25px 0 15px 0;
    }

    /* ตกแต่งปุ่มกด */
    div.stButton > button {
        background: linear-gradient(90deg, #D90429 0%, #EF233C 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        box-shadow: 0 4px 10px rgba(217, 4, 41, 0.3);
        transition: 0.3s;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #9B001C 0%, #D90429 100%);
        color: #FFF3B0;
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# 1. แบนเนอร์หัวข้อหลักด้านบนสุด
st.markdown("""
<div style="
    background: linear-gradient(135deg, #C0392B 0%, #E74C3C 50%, #FF7675 100%);
    padding: 25px 20px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0 6px 15px rgba(192, 57, 43, 0.25);
    margin-bottom: 25px;
">
    <h1 style="color: white; margin: 0; font-size: 34px; text-shadow: 1px 2px 4px rgba(0,0,0,0.25);">
        ❤️ โปรเจคการตรวจโรคหัวใจ ❤️
    </h1>
    <p style="color: #FDEDEC; margin: 8px 0 0 0; font-size: 16px;">
        ระบบประเมินและทำนายความเสี่ยงโรคหัวใจด้วยอัลกอริทึม K-Nearest Neighbors (KNN)
    </p>
</div>
""", unsafe_allow_html=True)

# 2. การ์ดแสดงภาพประกอบ 2 ฝั่ง
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div style="background-color: white; padding: 12px; border-radius: 15px; border-top: 5px solid #E74C3C; box-shadow: 0 4px 10px rgba(0,0,0,0.06); text-align: center;">
        <h3 style="color: #C0392B; margin: 0;">⚠️ เป็นโรคหัวใจ</h3>
    </div>
    """, unsafe_allow_html=True)
    st.image("./img/yes.jpg", use_container_width=True)

with col2:
    st.markdown("""
    <div style="background-color: white; padding: 12px; border-radius: 15px; border-top: 5px solid #27AE60; box-shadow: 0 4px 10px rgba(0,0,0,0.06); text-align: center;">
        <h3 style="color: #1E8449; margin: 0;">✅ ไม่เป็นโรคหัวใจ</h3>
    </div>
    """, unsafe_allow_html=True)
    st.image("./img/no.jpg", use_container_width=True)

# 3. ส่วนสถิติข้อมูล
st.markdown("""
<div class="section-banner-red">
    <h4 style="margin: 0;">📊 สถิติข้อมูลการเป็นโรคหัวใจ</h4>
</div>
""", unsafe_allow_html=True)

dt = pd.read_csv("./data/Heart3.csv")

# แสดงตัวเลขสรุปภาพรวม 3 ช่อง
total_cases = len(dt)
heart_yes = int((dt['HeartDisease'] == 1).sum())
heart_no = int((dt['HeartDisease'] == 0).sum())

m1, m2, m3 = st.columns(3)
m1.metric("จำนวนข้อมูลทั้งหมด", f"{total_cases:,} คน")
m2.metric("เป็นโรคหัวใจ (1)", f"{heart_yes:,} คน", f"{heart_yes/total_cases*100:.1f}%", delta_color="inverse")
m3.metric("ไม่เป็นโรคหัวใจ (0)", f"{heart_no:,} คน", f"{heart_no/total_cases*100:.1f}%")

with st.expander("📋 คลิกเพื่อดูตัวอย่างตารางข้อมูล (10 แถวแรก)", expanded=True):
    st.dataframe(dt.head(10), use_container_width=True)

heart_counts = dt['HeartDisease'].value_counts().rename(
    index={0: "ไม่เป็นโรคหัวใจ (0)", 1: "เป็นโรคหัวใจ (1)"}
)

if st.button("📈 แสดงการจินตทัศน์ข้อมูล", use_container_width=True):
    st.bar_chart(heart_counts, color="#E74C3C")

# 4. ส่วนกรอกข้อมูลเพื่อทำนายผล
st.markdown("""
<div class="section-banner-blue">
    <h4 style="margin: 0;">🩺 กรอกข้อมูลสุขภาพเพื่อทำนายผล</h4>
</div>
""", unsafe_allow_html=True)

# แบ่งช่องกรอกข้อมูลเป็น 2 คอลัมน์ซ้าย-ขวาให้ดูสะอาดและกรอกง่าย
left_col, right_col = st.columns(2)

with left_col:
    age = st.slider("อายุ (Age)", min_value=20, max_value=100, value=50)
    sex = st.selectbox(
        "เพศ (Sex)",
        options=[1, 0],
        format_func=lambda x: "ชาย (1)" if x == 1 else "หญิง (0)"
    )
    resting_bp = st.number_input("ความดันโลหิตขณะพัก (RestingBP)", min_value=0, max_value=250, value=130)
    cholesterol = st.number_input("ระดับคอเลสเตอรอล (Cholesterol)", min_value=0, max_value=700, value=200)
    fasting_bs = st.selectbox(
        "น้ำตาลในเลือดหลังอดอาหาร > 120 mg/dl (FastingBS)",
        options=[0, 1],
        format_func=lambda x: "ปกติ <= 120 mg/dl (0)" if x == 0 else "สูง > 120 mg/dl (1)"
    )
    max_hr = st.slider("อัตราการเต้นของหัวใจสูงสุด (MaxHR)", min_value=60, max_value=220, value=140)

with right_col:
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
    exercise_angina = st.selectbox(
        "เจ็บหน้าอกขณะออกกำลังกาย (ExerciseAngina)",
        options=[0, 1],
        format_func=lambda x: "ไม่เจ็บ (0)" if x == 0 else "เจ็บ (1)"
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
    oldpeak = st.number_input("ค่า Oldpeak (ST depression)", min_value=-3.0, max_value=7.0, value=0.0, step=0.1)
    st_slope = st.selectbox(
        "ความชันของคลื่น ST (ST_Slope)",
        options=[1, 2, 3],
        format_func=lambda x: {
            1: "1 - ลาดลง (Down)",
            2: "2 - ราบเรียบ (Flat)",
            3: "3 - ลาดขึ้น (Up)"
        }[x]
    )

st.markdown("")

# 5. ปุ่มทำนายผลและแสดงเปอร์เซ็นต์ความเสี่ยง
if st.button("🔍 ทำนายผลความเสี่ยงโรคหัวใจ", use_container_width=True):
    X = dt.drop('HeartDisease', axis=1)
    y = dt['HeartDisease']

    # ปรับสเกลข้อมูลให้ทุกตัวแปรมีน้ำหนักเท่าเทียมกัน
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ใช้ n_neighbors=5 เพื่อให้ได้ค่าความแม่นยำสูงและแบ่งเปอร์เซ็นต์ได้ชัดเจน
    Knn_model = KNeighborsClassifier(n_neighbors=5)
    Knn_model.fit(X_scaled, y)

    x_input = pd.DataFrame(
        [[age, sex, chest_pain, resting_bp, cholesterol, fasting_bs,
          resting_ecg, max_hr, exercise_angina, oldpeak, st_slope]],
        columns=X.columns
    )
    x_input_scaled = scaler.transform(x_input)

    out = Knn_model.predict(x_input_scaled)
    proba = Knn_model.predict_proba(x_input_scaled)[0]

    risk_percent = proba[1] * 100  # เปอร์เซ็นต์โอกาสเป็นโรคหัวใจ
    safe_percent = proba[0] * 100  # เปอร์เซ็นต์โอกาสไม่เป็นโรคหัวใจ

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])

    if out[0] == 1:
        with res_col1:
            st.image("./img/yes.jpg", use_container_width=True)
        with res_col2:
            st.markdown(f"""
            <div style="background-color: #FDEDEC; border-left: 6px solid #E74C3C; padding: 20px; border-radius: 12px;">
                <h3 style="color: #C0392B; margin: 0 0 10px 0;">⚠️ ผลการทำนาย: มีความเสี่ยงเป็นโรคหัวใจ</h3>
                <p style="font-size: 24px; font-weight: bold; color: #E74C3C; margin: 0;">
                    โอกาสเป็นโรคหัวใจ: {risk_percent:.1f}%
                </p>
                <p style="font-size: 15px; color: #555; margin: 5px 0 0 0;">
                    (โอกาสไม่เป็นโรคหัวใจ: {safe_percent:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(risk_percent))
    else:
        with res_col1:
            st.image("./img/no.jpg", use_container_width=True)
        with res_col2:
            st.markdown(f"""
            <div style="background-color: #EAFAF1; border-left: 6px solid #27AE60; padding: 20px; border-radius: 12px;">
                <h3 style="color: #1E8449; margin: 0 0 10px 0;">✅ ผลการทำนาย: ไม่เป็นโรคหัวใจ</h3>
                <p style="font-size: 24px; font-weight: bold; color: #27AE60; margin: 0;">
                    โอกาสเป็นโรคหัวใจ: {risk_percent:.1f}%
                </p>
                <p style="font-size: 15px; color: #555; margin: 5px 0 0 0;">
                    (โอกาสไม่เป็นโรคหัวใจ: {safe_percent:.1f}%)
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.progress(int(risk_percent))