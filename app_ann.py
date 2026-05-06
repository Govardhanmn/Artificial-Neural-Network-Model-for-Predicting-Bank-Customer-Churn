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

# ── BACKGROUND + OVERLAY ───────────────────────────────────
_bg_path = pathlib.Path("bg_bank.jpg")
_bg_b64 = base64.b64encode(_bg_path.read_bytes()).decode() if _bg_path.exists() else ""
_bg_css = f"url('data:image/jpeg;base64,{_bg_b64}')" if _bg_b64 else "linear-gradient(135deg,#020617,#0f172a)"

st.markdown(f"""
<style>
.stApp {{
    background: {_bg_css};
    background-size: cover;
    background-attachment: fixed;
}}

.stApp::before {{
    content:'';
    position:fixed;
    inset:0;
    background:rgba(2,6,23,0.75);
    backdrop-filter: blur(4px);
}}

.block-container {{
    padding: 2rem 3rem;
}}

.glass {{
    background: rgba(255,255,255,0.07);
    backdrop-filter: blur(18px);
    border-radius: 24px;
    padding: 30px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 10px 40px rgba(0,0,0,0.5);
}}

h1, h2, h3 {{
    color: white;
}}

label {{
    color: #e2e8f0 !important;
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
st.markdown("<h1 style='text-align:center;'>🏦 Churn Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#94a3b8;'>AI-powered retention analytics</p>", unsafe_allow_html=True)

col1, col2 = st.columns([1.1, 1])

# ── INPUT PANEL ────────────────────────────────────────────
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

        submit = st.form_submit_button("🚀 Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# ── OUTPUT PANEL ───────────────────────────────────────────
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

        # ── PREMIUM GAUGE ───────────────────────────────
        color = "#ef4444" if churn else "#22c55e"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={
                "suffix": "%",
                "font": {"size": 42, "color": color}
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color, "thickness": 0.3},
                "bgcolor": "rgba(255,255,255,0.05)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.25)"},
                    {"range": [30, 60], "color": "rgba(234,179,8,0.25)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.25)"}
                ],
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            margin=dict(t=30, b=0, l=0, r=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # ── RESULT ─────────────────────────────────────
        if churn:
            st.error(f"⚠️ High Churn Risk ({risk:.2f}%)")
        else:
            st.success(f"✅ Likely to Stay ({risk:.2f}%)")

        # ── INSIGHTS BELOW GAUGE ───────────────────────
        st.markdown("### 📊 Insights")

        if age > 50:
            st.warning("Older customers tend to churn more")
        if num_products == 1:
            st.warning("Low product engagement")
        if active == 0:
            st.error("Inactive customer → HIGH RISK")
        if num_products >= 2:
            st.success("Good product engagement")
        if active == 1:
            st.success("Active customer → Strong retention")

    st.markdown('</div>', unsafe_allow_html=True)
