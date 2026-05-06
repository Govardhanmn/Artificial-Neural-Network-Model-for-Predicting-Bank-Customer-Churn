import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import base64, pathlib

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

# ── BACKGROUND + GLASS EFFECT ──────────────────────────────
_bg_path = pathlib.Path("bg_bank.jpg")
_bg_b64 = base64.b64encode(_bg_path.read_bytes()).decode() if _bg_path.exists() else ""
_bg_css = f"url('data:image/jpeg;base64,{_bg_b64}')" if _bg_b64 else "linear-gradient(135deg,#0f172a,#1e293b)"

st.markdown(f"""
<style>
.stApp {{
    background: {_bg_css};
    background-size: cover;
    background-attachment: fixed;
}}

.block-container {{
    padding-top: 2rem;
}}

.glass {{
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}}

h1, h2, h3, h4 {{
    color: white;
}}

label {{
    color: #e5e7eb !important;
}}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("final_ann_model.h5")
    scaler = joblib.load("scaler_ann.pkl")
    return model, scaler

model, scaler = load_resources()

# ── HEADER ────────────────────────────────────────────────
st.markdown("<h1 style='text-align:center;'>🏦 Churn Intelligence Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#cbd5f5;'>AI-powered customer retention insights</p>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Insights"])

# ── TAB 1: PREDICTION ─────────────────────────────────────
with tab1:

    col1, col2 = st.columns([1.1, 1])

    with col1:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        with st.form("form"):
            st.subheader("Customer Profile")

            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            age = st.slider("Age", 18, 95, 35)

            credit_score = st.slider("Credit Score", 300, 850, 650)
            balance = st.number_input("Balance", 0.0, 300000.0, 75000.0)
            salary = st.number_input("Salary", 0.0, 300000.0, 60000.0)

            tenure = st.slider("Tenure", 0, 10, 5)
            num_products = st.selectbox("Products", [1,2,3,4])
            has_card = st.radio("Credit Card", ["Yes", "No"], horizontal=True)
            active = st.radio("Active Member", ["Yes", "No"], horizontal=True)

            submit = st.form_submit_button("🚀 Predict Churn")

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass">', unsafe_allow_html=True)

        if submit:

            # Encoding
            gender = 1 if gender == "Male" else 0
            geo_ger = 1 if geography == "Germany" else 0
            geo_spa = 1 if geography == "Spain" else 0
            card = 1 if has_card == "Yes" else 0
            active = 1 if active == "Yes" else 0

            features = np.array([[credit_score, gender, age, tenure,
                                  balance, num_products, card,
                                  active, salary, geo_ger, geo_spa]])

            prob = model.predict(scaler.transform(features))[0][0]
            risk = prob * 100
            churn = prob > 0.5

            # Gauge
            color = "#ef4444" if churn else "#22c55e"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk,
                number={"suffix": "%", "font": {"size": 38, "color": color}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": color},
                }
            ))

            st.plotly_chart(fig, use_container_width=True)

            # Result
            if churn:
                st.error(f"⚠️ High Risk ({risk:.2f}%)")
            else:
                st.success(f"✅ Safe ({risk:.2f}%)")

        st.markdown('</div>', unsafe_allow_html=True)

# ── TAB 2: INSIGHTS ───────────────────────────────────────
with tab2:

    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("Smart Insights")

    if 'submit' in locals() and submit:

        if age > 50:
            st.warning("Older customers tend to churn more")

        if num_products == 1:
            st.warning("Low product usage")

        if active == 0:
            st.error("Inactive user → High churn probability")

        if num_products >= 2:
            st.success("Good engagement")

        if active == 1:
            st.success("Active user → Strong retention")

    else:
        st.info("Run a prediction to see insights")

    st.markdown('</div>', unsafe_allow_html=True)
