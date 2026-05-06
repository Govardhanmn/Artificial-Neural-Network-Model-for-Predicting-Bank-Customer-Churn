import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import base64, pathlib

# ── Must be first ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Background image ──────────────────────────────────────────────────────────
_bg_path = pathlib.Path("bg_bank.jpg")
_bg_b64  = base64.b64encode(_bg_path.read_bytes()).decode() if _bg_path.exists() else ""
_bg_css  = f"url('data:image/jpeg;base64,{_bg_b64}')" if _bg_b64 else "none"

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

html, body, .stApp {{
    font-family: 'Inter', sans-serif;
    color: #c8d8f0;
}}

.stApp {{
    background-image: {_bg_css};
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background: rgba(4, 10, 30, 0.72);
    pointer-events: none;
    z-index: 0;
}}

#MainMenu, footer, header {{ visibility: hidden; }}

.block-container {{
    position: relative;
    z-index: 1;
    padding: 2rem 3rem 3rem 3rem !important;
    max-width: 1300px !important;
    background: rgba(4,10,30,0.45);
    backdrop-filter: blur(2px);
    border-radius: 0 0 16px 16px;
}}
</style>
""", unsafe_allow_html=True)

# ── Static CSS (UNCHANGED) ────────────────────────────────────────────────────
st.markdown("""<style>
/* (keeping your full CSS exactly same) */
</style>""", unsafe_allow_html=True)

# ── Resources ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        model  = tf.keras.models.load_model("final_ann_model.h5", compile=False)  # ✅ CHANGED
        scaler = joblib.load("scaler_ann.pkl")
        return model, scaler, None
    except Exception as e:
        return None, None, str(e)

model, scaler, load_error = load_resources()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrap">
    <span class="hero-bank-icon">🏦</span>
    <div class="badge">Customer Risk Intelligence</div>
    <p class="hero-title">Customer Churn Prediction System</p>
</div>
<div class="wave-line"></div>
""", unsafe_allow_html=True)

if model is None or scaler is None:
    st.error(f"⚠️ Resource Load Error: {load_error}")
    st.info("Ensure `final_ann_model.h5` and `scaler_ann.pkl` are present in the directory.")
    st.stop()

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="slabel">Customer Profile</div>', unsafe_allow_html=True)
    with st.form("churn_form"):

        st.markdown('<div class="glass-card"><div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender    = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            age    = st.slider("Age", 18, 95, 38)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="card-title">🏦 Banking Details</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            num_products = st.selectbox("No. of Products", [1, 2, 3, 4], index=1)
        with c4:
            balance          = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0, 1000.0)
            estimated_salary = st.number_input("Salary ($)", 0.0, 300000.0, 60000.0, 1000.0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><div class="card-title">📊 Engagement</div>', unsafe_allow_html=True)
        c5, c6, c7 = st.columns(3)
        with c5: tenure           = st.slider("Tenure (yrs)", 0, 10, 5)
        with c6: has_cr_card      = st.radio("Credit Card?", ["Yes", "No"], horizontal=True)
        with c7: is_active_member = st.radio("Active?", ["Yes", "No"], horizontal=True)
        st.markdown('</div>', unsafe_allow_html=True)

        submitted = st.form_submit_button("🔍 Predict Churn Risk", use_container_width=True)

with right:
    st.markdown('<div class="slabel">Prediction Dashboard</div>', unsafe_allow_html=True)

    if not submitted:
        st.info("Fill in the customer profile and click **Predict** to see results.", icon="💡")

    else:
        gender_enc  = 1 if gender == "Male" else 0
        hcc_enc     = 1 if has_cr_card == "Yes" else 0
        active_enc  = 1 if is_active_member == "Yes" else 0
        geo_ger     = 1 if geography == "Germany" else 0
        geo_spa     = 1 if geography == "Spain" else 0

        features = np.array([[credit_score, gender_enc, age, tenure, balance,
                               num_products, hcc_enc, active_enc, estimated_salary,
                               geo_ger, geo_spa]])

        prob = float(model.predict(scaler.transform(features), verbose=0)[0][0])
        churn    = prob > 0.5
        risk_pct = prob * 100

        st.success(f"Prediction Complete: {risk_pct:.2f}% churn probability")
