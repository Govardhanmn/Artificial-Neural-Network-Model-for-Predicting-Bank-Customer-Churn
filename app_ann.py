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

/* ── Full-page background image ── */
.stApp {{
    background-image: {_bg_css};
    background-size: cover;
    background-position: center center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}

/* Dark overlay so content stays readable */
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
    -webkit-backdrop-filter: blur(2px);
    border-radius: 0 0 16px 16px;
}}
</style>
""", unsafe_allow_html=True)

# ── Static CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ─── HERO BANNER ─────────────────────────────────────────── */
.hero-wrap {
    position: relative;
    background: linear-gradient(135deg, rgba(6,14,36,0.85) 0%, rgba(11,26,66,0.85) 40%, rgba(7,22,56,0.85) 100%);
    border: 1px solid rgba(56,100,200,0.25);
    border-radius: 20px;
    padding: 2.2rem 2.8rem 1.8rem;
    margin: 1.2rem 0 1.6rem;
    overflow: hidden;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
}
.hero-wrap::before {
    content:'';
    position:absolute; top:-80px; right:-80px;
    width:320px; height:320px;
    background: radial-gradient(circle, rgba(56,100,220,0.18) 0%, transparent 70%);
    border-radius:50%;
    pointer-events:none;
}
.hero-wrap::after {
    content:'AI';
    position:absolute; top:18px; right:40px;
    font-size:4rem; font-weight:900;
    color:rgba(99,163,255,0.12);
    letter-spacing:.1em;
    pointer-events:none;
}
.hero-bank-icon {
    position:absolute; bottom:10px; left:28px;
    font-size:3.5rem; opacity:0.07;
    pointer-events:none;
}
.badge {
    display:inline-block;
    background:rgba(56,100,220,0.18);
    border:1px solid rgba(99,163,255,0.3);
    color:#63a3ff;
    font-size:.85rem; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase;
    padding:.35rem 1rem; border-radius:20px;
    margin-bottom:.8rem;
}
.hero-title {
    font-size:2.4rem; font-weight:800;
    background:linear-gradient(90deg,#63a3ff,#a78bfa,#f6ad55);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    background-clip:text; line-height:1.15; margin-bottom:.35rem;
}
.hero-sub { font-size:.92rem; color:#7a9cc8; }

/* ─── WAVE DIVIDER ─────────────────────────────────────────── */
.wave-line {
    width:100%; height:2px;
    background:linear-gradient(90deg, transparent, rgba(63,130,255,0.4), rgba(167,139,250,0.4), transparent);
    margin: .5rem 0 1.4rem;
    border-radius:2px;
}
.slabel {
    font-size:.65rem; font-weight:700; letter-spacing:.12em;
    text-transform:uppercase; color:#63a3ff; margin-bottom:.5rem;
}

/* ─── GLASS CARD ──────────────────────────────────────────── */
.glass-card {
    background:rgba(6,14,40,0.65);
    border:1px solid rgba(56,100,200,0.22);
    border-radius:16px;
    padding:1.4rem 1.6rem 1.6rem;
    backdrop-filter:blur(10px);
    -webkit-backdrop-filter:blur(10px);
    margin-bottom:1rem;
    transition: border-color .3s;
}
.glass-card:hover { border-color:rgba(99,163,255,0.4); }
.card-title {
    font-size:.75rem; font-weight:600; color:#7a9cc8;
    text-transform:uppercase; letter-spacing:.1em;
    padding-bottom:.65rem; margin-bottom:.9rem;
    border-bottom:1px solid rgba(255,255,255,0.06);
}

/* ─── WIDGETS ────────────────────────────────────────────── */
label { color:#7a9cc8 !important; font-size:.8rem !important; font-weight:500 !important; }
div[data-baseweb="input"] input,
div[data-baseweb="select"] div {
    background:rgba(4,10,30,0.8) !important;
    border:1px solid rgba(56,100,200,0.35) !important;
    border-radius:8px !important; color:#c8d8f0 !important;
}
div[data-baseweb="input"] input:focus {
    border-color:rgba(99,163,255,0.6) !important;
    box-shadow:0 0 0 3px rgba(99,163,255,0.12) !important;
}
div[data-testid="stForm"] { background:transparent !important; border:none !important; padding:0 !important; }

/* ─── BUTTON ─────────────────────────────────────────────── */
.stButton > button {
    width:100% !important;
    background:linear-gradient(135deg,#1e40af,#6d28d9) !important;
    color:#fff !important; font-weight:700 !important;
    font-size:.95rem !important; border:none !important;
    border-radius:12px !important; height:50px !important;
    letter-spacing:.04em !important;
    box-shadow:0 4px 20px rgba(30,64,175,0.45) !important;
    transition:all .3s !important;
}
.stButton > button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 8px 30px rgba(30,64,175,0.65) !important;
}

/* ─── RESULT PANELS ───────────────────────────────────────── */
.res-churn {
    background:linear-gradient(135deg,rgba(220,38,38,0.14),rgba(252,129,74,0.08));
    border:1px solid rgba(220,38,38,0.4); border-radius:18px;
    padding:1.6rem; margin-top:.8rem; animation:fadeUp .5s ease-out;
    text-align:center; backdrop-filter:blur(6px);
}
.res-stay {
    background:linear-gradient(135deg,rgba(16,185,129,0.14),rgba(59,130,246,0.08));
    border:1px solid rgba(16,185,129,0.4); border-radius:18px;
    padding:1.6rem; margin-top:.8rem; animation:fadeUp .5s ease-out;
    text-align:center; backdrop-filter:blur(6px);
}
.res-verdict { font-size:1.5rem; font-weight:800; margin:.3rem 0; }
.verdict-churn { color:#f87171; }
.verdict-stay  { color:#34d399; }
.res-tip {
    background:rgba(255,255,255,0.04); border-radius:10px;
    padding:.65rem .9rem; font-size:.82rem; color:#94a3b8; margin-top:.8rem;
}
.risk-item {
    display:flex; align-items:flex-start; gap:.55rem;
    padding:.5rem 0; border-bottom:1px solid rgba(255,255,255,0.04);
    font-size:.82rem;
}
.risk-item:last-child { border-bottom:none; }
.risk-dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; margin-top:.3rem; }
.dot-red  { background:#f87171; box-shadow:0 0 6px #f87171; }
.dot-yel  { background:#fbbf24; box-shadow:0 0 6px #fbbf24; }
.dot-grn  { background:#34d399; box-shadow:0 0 6px #34d399; }
.risk-label { font-weight:600; color:#c8d8f0; }
.risk-desc  { color:#64748b; font-size:.77rem; }

@keyframes fadeUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Resources ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_resources():
    try:
        model  = tf.keras.models.load_model("final_ann_model.h5", compile=False)
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
    st.info("Ensure `final_ann_model.keras` and `scaler_ann.pkl` are present in the directory.")
    st.stop()

# ── Layout ─────────────────────────────────────────────────────────────────────
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.markdown('<div class="slabel">Customer Profile</div>', unsafe_allow_html=True)
    with st.form("churn_form"):

        # Card 1 – Personal
        st.markdown('<div class="glass-card"><div class="card-title">👤 Personal Details</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
            gender    = st.selectbox("Gender", ["Male", "Female"])
        with c2:
            age    = st.slider("Age", 18, 95, 38)
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 2 – Banking
        st.markdown('<div class="glass-card"><div class="card-title">🏦 Banking Details</div>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        with c3:
            credit_score = st.slider("Credit Score", 300, 850, 650)
            num_products = st.selectbox("No. of Products", [1, 2, 3, 4], index=1)
        with c4:
            balance          = st.number_input("Balance ($)", 0.0, 300000.0, 75000.0, 1000.0)
            estimated_salary = st.number_input("Salary ($)", 0.0, 300000.0, 60000.0, 1000.0)
        st.markdown('</div>', unsafe_allow_html=True)

        # Card 3 – Engagement
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
        # Placeholder gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            number={"suffix": "%", "font": {"size": 40, "color": "#334155"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#1e3a5f"},
                "bar": {"color": "#1e3a5f", "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],   "color": "rgba(16,185,129,0.08)"},
                    {"range": [30, 60],  "color": "rgba(251,191,36,0.08)"},
                    {"range": [60, 100], "color": "rgba(220,38,38,0.08)"},
                ],
                "threshold": {"line": {"color": "#334155", "width": 2}, "thickness": 0.75, "value": 50},
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#4a6080", "family": "Inter"},
            height=260, margin=dict(l=30, r=30, t=30, b=10),
        )
        st.markdown('<div class="glass-card"><div class="card-title">⭕ CHURN RISK GAUGE</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="gauge_empty")
        st.markdown('</div>', unsafe_allow_html=True)
        st.info("Fill in the customer profile and click **Predict** to see results.", icon="💡")

    else:
        # ── Encode ────────────────────────────────────────────────────────────
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

        # ── Donut Gauge ───────────────────────────────────────────────────────
        gauge_clr = "#f87171" if churn else "#34d399"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_pct,
            number={"suffix": "%", "font": {"size": 44, "color": gauge_clr}, "valueformat": ".1f"},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#1e3a5f", "tickwidth": 1},
                "bar": {"color": gauge_clr, "thickness": 0.22},
                "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],   "color": "rgba(16,185,129,0.1)"},
                    {"range": [30, 60],  "color": "rgba(251,191,36,0.1)"},
                    {"range": [60, 100], "color": "rgba(220,38,38,0.1)"},
                ],
                "threshold": {"line": {"color": "#fff", "width": 3}, "thickness": 0.8, "value": 50},
            },
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#c8d8f0", "family": "Inter"},
            height=260, margin=dict(l=30, r=30, t=30, b=10),
        )
        verdict_cls = "verdict-churn" if churn else "verdict-stay"
        verdict_txt = "High Churn Risk" if churn else "Likely to Retain"

        st.markdown('<div class="glass-card"><div class="card-title">⭕ CHURN RISK GAUGE</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, key="gauge_result")
        st.markdown(f"""
        <div style="text-align:center;margin-top:-.5rem;margin-bottom:.5rem">
            <div class="res-verdict {verdict_cls}">{verdict_txt}</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Result card ───────────────────────────────────────────────────────
        if churn:
            tip = "Target this customer with personalised retention offers — loyalty rewards or a dedicated RM outreach."
            box_cls = "res-churn"
        else:
            tip = "Customer appears satisfied. Maintain regular engagement and monitor for behavioural shifts."
            box_cls = "res-stay"

        st.markdown(f"""
        <div class="{box_cls}">
            <div style="font-size:2rem;margin-bottom:.3rem">{'⚠️' if churn else '✅'}</div>
            <p style="color:#94a3b8;font-size:.9rem">Churn probability: <strong style="color:{'#f87171' if churn else '#34d399'}">{risk_pct:.1f}%</strong></p>
            <div class="res-tip">💡 {tip}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Risk signals ──────────────────────────────────────────────────────
        signals = []
        if age > 50:           signals.append(("red", "Age > 50", "Older customers churn at higher rates."))
        if num_products == 1:  signals.append(("red", "Single Product", "Low product engagement increases churn risk."))
        if is_active_member == "No": signals.append(("red", "Inactive Member", "Inactivity is the strongest churn predictor."))
        if geography == "Germany": signals.append(("yel", "Germany Region", "Higher average churn rate in this region."))
        if balance == 0:       signals.append(("yel", "Zero Balance", "Zero-balance accounts are volatile."))
        if credit_score < 500: signals.append(("yel", "Low Credit Score", "May indicate financial stress."))
        if is_active_member == "Yes": signals.append(("grn", "Active Member", "Active engagement strongly reduces churn."))
        if num_products >= 2:  signals.append(("grn", "Multiple Products", "Cross-product use boosts retention."))
        if tenure >= 5:        signals.append(("grn", "Long Tenure", "Loyalty history is a strong retention signal."))

        if signals:
            st.markdown('<div class="glass-card" style="margin-top:.8rem"><div class="card-title">📡 RISK SIGNALS</div>', unsafe_allow_html=True)
            for dot, label, desc in signals:
                st.markdown(f"""
                <div class="risk-item">
                    <div class="risk-dot dot-{dot}"></div>
                    <div>
                        <div class="risk-label">{label}</div>
                        <div class="risk-desc">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
