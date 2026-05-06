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

/* REMOVE EMPTY BOXES */
div[data-testid="stHorizontalBlock"]:empty,
div[data-testid="stVerticalBlock"]:empty,
div[data-testid="stColumn"]:empty {{
    display: none !important;
}}

/* BACKGROUND */
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

/* LAYOUT */
.block-container {{
    padding: 1.2rem 2rem;
}}

/* KPI CARDS */
.kpi-card {{
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    transition: 0.3s;
}}

.kpi-card:hover {{
    transform: translateY(-4px);
}}

.kpi-title {{
    color: #94a3b8;
    font-size: 13px;
}}

.kpi-value {{
    color: white;
    font-size: 24px;
    font-weight: 600;
}}

/* INSIGHTS */
.insight-text {{
    font-size: 13px;
    color: #cbd5e1;
    margin-bottom: 4px;
}}

/* TEXT */
h1 {{
    color: white;
}}

label {{
    color: #cbd5e1 !important;
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
st.markdown("<p style='color:#94a3b8;'>AI-powered retention dashboard</p>", unsafe_allow_html=True)

# ── KPI ROW ────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

risk_val = "--"
score_val = "--"
activity_val = "--"
product_val = "--"

kpi1 = k1.empty()
kpi2 = k2.empty()
kpi3 = k3.empty()
kpi4 = k4.empty()

# ── MAIN GRID ──────────────────────────────────────────
left, right = st.columns([1,1])

# ── INPUT SECTION (SIDE-BY-SIDE) ───────────────────────
with left:
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        age = st.slider("Age", 18, 95, 35)
        balance = st.number_input("Balance", 0.0, 300000.0, 75000.0)
        tenure = st.slider("Tenure", 0, 10, 5)
        has_card = st.radio("Credit Card", ["Yes", "No"])

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        credit_score = st.slider("Credit Score", 300, 850, 650)
        salary = st.number_input("Salary", 0.0, 300000.0, 60000.0)
        num_products = st.selectbox("Products", [1,2,3,4])
        active = st.radio("Active Member", ["Yes", "No"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict = st.button("🚀 Predict", use_container_width=True)

# ── OUTPUT SECTION ─────────────────────────────────────
with right:
    if predict:
        gender_val = 1 if gender == "Male" else 0
        geo_ger = 1 if geography == "Germany" else 0
        geo_spa = 1 if geography == "Spain" else 0
        card = 1 if has_card == "Yes" else 0
        active_flag = 1 if active == "Yes" else 0

        X = np.array([[credit_score, gender_val, age, tenure,
                       balance, num_products, card,
                       active_flag, salary, geo_ger, geo_spa]])

        prob = model.predict(scaler.transform(X))[0][0]
        risk = prob * 100

        # ── GAUGE (CENTERED & SMALLER) ─────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}}
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "white"},
            height=280
        )

        g1, g2, g3 = st.columns([1,2,1])
        with g2:
            st.plotly_chart(fig, use_container_width=True)

        # ── INSIGHTS (SMALL FONT) ──────────────────────
        st.markdown("### 📊 Insights")

        def insight(msg, type="normal"):
            color = {
                "success": "#22c55e",
                "warning": "#facc15",
                "error": "#ef4444",
                "normal": "#cbd5e1"
            }[type]

            st.markdown(
                f"<div class='insight-text' style='color:{color}'>{msg}</div>",
                unsafe_allow_html=True
            )

        if risk > 60:
            insight("⚠️ High churn risk detected. Immediate action needed.", "error")
        elif risk > 30:
            insight("⚡ Moderate churn risk. Monitor customer.", "warning")
        else:
            insight("✅ Customer is stable.", "success")

        if age > 50:
            insight("Older customers show higher churn tendency.", "warning")

        if num_products == 1:
            insight("Low product engagement (1 product).", "warning")

        if num_products >= 3:
            insight("Strong engagement (multiple products).", "success")

        if active_flag == 0:
            insight("Inactive customer is a major churn driver.", "error")

        if active_flag == 1:
            insight("Active customer improves retention.", "success")

        if balance < 10000:
            insight("Low balance indicates weak engagement.", "warning")

        if balance > 100000:
            insight("High balance → high-value customer.", "success")

        if credit_score < 500:
            insight("Low credit score increases churn risk.", "error")

        if credit_score > 700:
            insight("Good credit score → stable profile.", "success")

        # ── KPI UPDATE ────────────────────────────────
        risk_val = f"{risk:.1f}%"
        score_val = f"{100-risk:.0f}"
        activity_val = "Active" if active_flag else "Inactive"
        product_val = str(num_products)

# ── KPI RENDER ─────────────────────────────────────────
kpi1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Churn Risk</div><div class='kpi-value'>{risk_val}</div></div>", unsafe_allow_html=True)

kpi2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Customer Score</div><div class='kpi-value'>{score_val}</div></div>", unsafe_allow_html=True)

kpi3.markdown(f"<div class='kpi-card'><div class='kpi-title'>Activity</div><div class='kpi-value'>{activity_val}</div></div>", unsafe_allow_html=True)

kpi4.markdown(f"<div class='kpi-card'><div class='kpi-title'>Products</div><div class='kpi-value'>{product_val}</div></div>", unsafe_allow_html=True)
