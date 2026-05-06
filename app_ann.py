import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go
import base64, pathlib

# ── CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide"
)

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

/* KPI */
.kpi-card {{
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
}}
.kpi-title {{color:#94a3b8;font-size:13px}}
.kpi-value {{color:white;font-size:24px;font-weight:600}}

/* GLASS */
.glass-card {{
    backdrop-filter: blur(16px);
    background: rgba(255,255,255,0.05);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
}}

/* INSIGHTS */
.insight-text {{
    font-size: 13px;
    margin-bottom: 4px;
}}

h1, h2, h3 {{color:white}}
label {{color:#cbd5e1 !important}}
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────────────────────
@st.cache_resource
def load():
    return (
        tf.keras.models.load_model("final_ann_model.h5"),
        joblib.load("scaler_ann.pkl")
    )

model, scaler = load()

# ── HEADER ─────────────────────────────────────────────
st.markdown("## 🏦 Customer Churn Intelligence")

# ── KPI PLACEHOLDERS ───────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
kpi1, kpi2, kpi3, kpi4 = k1.empty(), k2.empty(), k3.empty(), k4.empty()

risk_val, score_val, activity_val, product_val = "--","--","--","--"

# ── MAIN GRID ──────────────────────────────────────────
left, right = st.columns([1.1, 1])

# ── INPUT FORM (FROM YOUR CODE) ────────────────────────
with left:
    with st.form("churn_form"):

        col1, col2 = st.columns(2)

        with col1:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            age = st.slider("Age", 18, 95, 38)
            balance = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0)
            tenure = st.slider("Tenure", 0, 10, 5)
            has_card = st.radio("Credit Card", ["Yes", "No"])

        with col2:
            gender = st.selectbox("Gender", ["Male", "Female"])
            credit_score = st.slider("Credit Score", 300, 850, 650)
            salary = st.number_input("Salary ($)", 0.0, 300000.0, 60000.0)
            num_products = st.selectbox("Products", [1,2,3,4])
            active = st.radio("Active Member", ["Yes", "No"])

        submit = st.form_submit_button("🚀 Predict", use_container_width=True)

# ── OUTPUT ─────────────────────────────────────────────
with right:
    if submit:

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
        churn = prob > 0.5

        # ── PREMIUM GAUGE ─────────────────────────────
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%", "font": {"size": 38}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"thickness": 0.3},
                "steps": [
                    {"range": [0, 30], "color": "#22c55e"},
                    {"range": [30, 60], "color": "#facc15"},
                    {"range": [60, 100], "color": "#ef4444"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "value": risk
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            height=380,
            font={"color":"white"}
        )

        g1, g2, g3 = st.columns([1,2.5,1])
        with g2:
            st.plotly_chart(fig, use_container_width=True)

        # ── RESULT ────────────────────────────────────
        if churn:
            st.error(f"⚠️ High Churn Risk ({risk:.1f}%)")
        else:
            st.success(f"✅ Likely to Stay ({risk:.1f}%)")

        # ── INSIGHTS (SMALL FONT) ─────────────────────
        st.markdown("### 📊 Insights")

        def insight(msg, color):
            st.markdown(f"<div class='insight-text' style='color:{color}'>{msg}</div>", unsafe_allow_html=True)

        if age > 50:
            insight("Older customers have higher churn tendency", "#facc15")
        if num_products == 1:
            insight("Low product engagement", "#facc15")
        if active_flag == 0:
            insight("Inactive customer - HIGH RISK", "#ef4444")
        if num_products >= 2:
            insight("Good product engagement", "#22c55e")
        if active_flag == 1:
            insight("Active customer - strong retention", "#22c55e")

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
