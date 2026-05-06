import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import base64, pathlib

# ── CONFIG ─────────────────────────────────────────────
st.set_page_config(page_title="Churn Intelligence", layout="wide")

# ── BACKGROUND ─────────────────────────────────────────
_bg_path = pathlib.Path("bg_bank.jpg")
_bg_b64 = base64.b64encode(_bg_path.read_bytes()).decode() if _bg_path.exists() else ""
_bg_css = f"url('data:image/jpeg;base64,{_bg_b64}')" if _bg_b64 else "linear-gradient(135deg,#020617,#0f172a)"

st.markdown(f"""
<style>
.stApp {{
    background: {_bg_css};
    background-size: cover;
}}

.stApp::before {{
    content:'';
    position:fixed;
    inset:0;
    background: rgba(2,6,23,0.85);
}}

.block-container {{
    padding: 1.5rem 3rem;
}}

/* REMOVE EMPTY BLOCKS */
div[data-testid="stHorizontalBlock"]:empty {{
    display: none !important;
}}
div[data-testid="stVerticalBlock"]:empty {{
    display: none !important;
}}

/* GLASS CARD */
.glass {{
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 25px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 10px 35px rgba(0,0,0,0.5);
}}

/* 🌈 KPI STRIPE STYLE */
.kpi-card {{
    position: relative;
    border-radius: 18px;
    padding: 20px;
    text-align: center;
    overflow: hidden;
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    transition: all 0.3s ease;
}}

.kpi-card::before {{
    content: '';
    position: absolute;
    inset: -1px;
    border-radius: 18px;
    background: linear-gradient(120deg, var(--glow), #3b82f6, #a855f7);
    opacity: 0.25;
    filter: blur(20px);
    z-index: 0;
}}

.kpi-card * {{
    position: relative;
    z-index: 2;
}}

.kpi-card:hover {{
    transform: translateY(-6px) scale(1.02);
    box-shadow: 0 15px 40px rgba(0,0,0,0.6);
}}

.kpi-title {{
    color: #94a3b8;
    font-size: 13px;
}}

.kpi-value {{
    color: white;
    font-size: 26px;
    font-weight: 600;
}}

h1, h2, h3 {{
    color: white;
}}

label {{
    color: #e2e8f0 !important;
}}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model("final_ann_model.h5")
    scaler = joblib.load("scaler_ann.pkl")
    return model, scaler

model, scaler = load_resources()

# ── HEADER ─────────────────────────────────────────────
st.markdown("<h1>🏦 Churn Intelligence</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#94a3b8;'>AI-powered customer retention dashboard</p>", unsafe_allow_html=True)

# ── KPI CARDS ──────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

risk = None
active_flag = None

risk_val = "--"
score_val = "--"
activity_val = "--"
product_val = "--"

with k1: kpi1 = st.empty()
with k2: kpi2 = st.empty()
with k3: kpi3 = st.empty()
with k4: kpi4 = st.empty()

# ── MAIN LAYOUT ─────────────────────────────────────────
left, right = st.columns([1, 1])

# ── INPUT PANEL ─────────────────────────────────────────
with left:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    with st.form("form"):
        st.subheader("Customer Details")

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

# ── OUTPUT PANEL ───────────────────────────────────────
with right:
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    if submit:
        gender_val = 1 if gender == "Male" else 0
        geo_ger = 1 if geography == "Germany" else 0
        geo_spa = 1 if geography == "Spain" else 0
        card = 1 if has_card == "Yes" else 0
        active_flag = 1 if active == "Yes" else 0

        features = np.array([[credit_score, gender_val, age, tenure,
                              balance, num_products, card,
                              active_flag, salary, geo_ger, geo_spa]])

        prob = model.predict(scaler.transform(features))[0][0]
        risk = prob * 100
        churn = prob > 0.5

        # GAUGE
        color = "#ef4444" if churn else "#22c55e"

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%", "font": {"size": 40, "color": color}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 30], "color": "rgba(34,197,94,0.2)"},
                    {"range": [30, 60], "color": "rgba(234,179,8,0.2)"},
                    {"range": [60, 100], "color": "rgba(239,68,68,0.2)"}
                ]
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            margin=dict(t=20, b=0)
        )

        st.plotly_chart(fig, use_container_width=True)

        # RESULT
        if churn:
            st.error(f"High Churn Risk ({risk:.2f}%)")
        else:
            st.success(f"Customer Stable ({risk:.2f}%)")

        # INSIGHTS
        st.markdown("### 📊 Insights")

        if age > 50:
            st.warning("Older customers show higher churn")
        if num_products == 1:
            st.warning("Low engagement detected")
        if active_flag == 0:
            st.error("Inactive customer")
        if num_products >= 2:
            st.success("Good product usage")
        if active_flag == 1:
            st.success("Active engagement")

        # KPI VALUES UPDATE
        risk_val = f"{risk:.1f}%"
        score_val = f"{100-risk:.0f}"
        activity_val = "Active" if active_flag == 1 else "Inactive"
        product_val = f"{num_products}"

    st.markdown('</div>', unsafe_allow_html=True)

# ── KPI RENDER ─────────────────────────────────────────
if risk:
    if risk > 60:
        glow = "#ef4444"
    elif risk > 30:
        glow = "#f59e0b"
    else:
        glow = "#22c55e"
else:
    glow = "#3b82f6"

kpi1.markdown(f"<div class='kpi-card' style='--glow:{glow}'><div class='kpi-title'>Churn Risk</div><div class='kpi-value'>{risk_val}</div></div>", unsafe_allow_html=True)

kpi2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Customer Score</div><div class='kpi-value'>{score_val}</div></div>", unsafe_allow_html=True)

kpi3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Activity</div><div class='kpi-value'>{activity_val}</div></div>", unsafe_allow_html=True)

kpi4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Products</div><div class='kpi-value'>{product_val}</div></div>", unsafe_allow_html=True)
