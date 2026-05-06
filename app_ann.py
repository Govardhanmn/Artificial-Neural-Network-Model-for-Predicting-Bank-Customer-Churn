import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import base64, pathlib

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Background ───────────────────────────────────────────────
_bg_path = pathlib.Path("bg_bank.jpg")
_bg_b64  = base64.b64encode(_bg_path.read_bytes()).decode() if _bg_path.exists() else ""
_bg_css  = f"url('data:image/jpeg;base64,{_bg_b64}')" if _bg_b64 else "none"

st.markdown(f"""
<style>
.stApp {{
    background-image: {_bg_css};
    background-size: cover;
    background-attachment: fixed;
}}
.stApp::before {{
    content:'';
    position:fixed;
    inset:0;
    background:rgba(4,10,30,0.75);
}}
</style>
""", unsafe_allow_html=True)

# ── Load Resources (FIXED) ───────────────────────────────────
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("final_ann_model.h5")  # ✅ FIX
    scaler = joblib.load("scaler_ann.pkl")
    return model, scaler

model, scaler = load_resources()

# ── Hero ─────────────────────────────────────────────────────
st.markdown("## 🏦 Customer Churn Prediction System")

left, right = st.columns([1.1, 0.9])

# ── INPUT PANEL ──────────────────────────────────────────────
with left:
    with st.form("churn_form"):

        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age", 18, 95, 38)

        credit_score = st.slider("Credit Score", 300, 850, 650)
        balance = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0)
        salary = st.number_input("Salary ($)", 0.0, 300000.0, 60000.0)

        tenure = st.slider("Tenure", 0, 10, 5)
        num_products = st.selectbox("Products", [1,2,3,4])
        has_card = st.radio("Credit Card", ["Yes", "No"], horizontal=True)
        active = st.radio("Active Member", ["Yes", "No"], horizontal=True)

        submit = st.form_submit_button("🔍 Predict")

# ── OUTPUT PANEL ─────────────────────────────────────────────
with right:

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

        # ── Gauge ────────────────────────────────────────────
        color = "#f87171" if churn else "#34d399"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%", "font": {"size": 40, "color": color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(16,185,129,0.1)"},
                    {"range": [30, 60], "color": "rgba(251,191,36,0.1)"},
                    {"range": [60, 100], "color": "rgba(220,38,38,0.1)"},
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # ── Result ───────────────────────────────────────────
        if churn:
            st.error(f"⚠️ High Churn Risk ({risk:.2f}%)")
        else:
            st.success(f"✅ Likely to Stay ({risk:.2f}%)")

        # ── Insights ─────────────────────────────────────────
        st.markdown("### 📊 Risk Insights")

        if age > 50:
            st.warning("Older customers have higher churn tendency")
        if num_products == 1:
            st.warning("Low product engagement")
        if active == 0:
            st.error("Inactive customer - HIGH RISK")
        if num_products >= 2:
            st.success("Good product engagement")
        if active == 1:
            st.success("Active customer - strong retention")