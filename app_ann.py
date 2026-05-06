import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import plotly.graph_objects as go

# ── CONFIG ─────────────────────────────
st.set_page_config(page_title="Churn Intelligence", layout="wide")

# ── CLEAN CSS (KEY FIX) ─────────────────
st.markdown("""
<style>

/* REMOVE ALL DEFAULT BOXES */
div[data-testid="stForm"],
div[data-testid="stVerticalBlock"],
div[data-testid="stHorizontalBlock"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    padding: 0 !important;
}

/* REMOVE INPUT BOX FEEL */
div[data-baseweb="select"],
div[data-baseweb="input"],
div[data-baseweb="base-input"],
div[data-testid="stNumberInput"],
div[data-testid="stTextInput"] {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 8px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* REMOVE EXTRA CONTAINER SPACING */
.block-container {
    padding: 1rem 2rem;
}

/* KPI CARDS CLEAN */
.kpi {
    padding: 18px;
    border-radius: 14px;
    background: rgba(255,255,255,0.04);
    text-align: center;
}

.kpi h4 {
    color: #94a3b8;
    margin-bottom: 5px;
}

.kpi h2 {
    color: white;
}

/* LABELS */
label {
    color: #cbd5f5 !important;
}

/* TITLE */
h1 {
    color: white;
}

/* REMOVE EMPTY BLOCKS COMPLETELY */
div:empty {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ─────────────────────────
@st.cache_resource
def load():
    model = tf.keras.models.load_model("final_ann_model.h5")
    scaler = joblib.load("scaler_ann.pkl")
    return model, scaler

model, scaler = load()

# ── HEADER ─────────────────────────────
st.title("🏦 Churn Intelligence")

# ── KPI ROW ───────────────────────────
k1, k2, k3, k4 = st.columns(4)

risk_val = "--"
score_val = "--"
activity_val = "--"
product_val = "--"

with k1:
    st.markdown(f"<div class='kpi'><h4>Churn Risk</h4><h2>{risk_val}</h2></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><h4>Customer Score</h4><h2>{score_val}</h2></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><h4>Activity</h4><h2>{activity_val}</h2></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><h4>Products</h4><h2>{product_val}</h2></div>", unsafe_allow_html=True)

# ── MAIN LAYOUT ───────────────────────
left, right = st.columns([1,1])

# ── INPUT SIDE (FLAT DESIGN) ──────────
with left:
    st.subheader("Customer Details")

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        age = st.slider("Age", 18, 95, 35)
        balance = st.number_input("Balance", 0.0, 300000.0, 75000.0)
        tenure = st.slider("Tenure", 0, 10, 5)

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        credit_score = st.slider("Credit Score", 300, 850, 650)
        salary = st.number_input("Salary", 0.0, 300000.0, 60000.0)
        num_products = st.selectbox("Products", [1,2,3,4])

    has_card = st.radio("Credit Card", ["Yes", "No"], horizontal=True)
    active = st.radio("Active Member", ["Yes", "No"], horizontal=True)

    predict = st.button("🚀 Predict")

# ── OUTPUT SIDE ───────────────────────
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

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}}
        ))

        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)
