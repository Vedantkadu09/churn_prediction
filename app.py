# ==========================================
# Binary Encoding Reference:
# Gender: Female = 1, Male = 0
# Churn: Yes = 1, No = 0
# Feature Pipeline Order: ['Age', 'Gender', 'Tenure', 'MonthlyCharges']
# Artifacts: scaler.pkl, model.pkl
# ==========================================

import streamlit as st
import numpy as np
import joblib

# ----------------- App Configuration -----------------
st.set_page_config(
    page_title="AI Churn Intelligence",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ----------------- Modern CSS & Keyframe Animations -----------------
st.markdown("""
<style>
    @keyframes pulseGlow {
        0% { box-shadow: 0 0 15px rgba(14, 165, 233, 0.3); }
        50% { box-shadow: 0 0 30px rgba(14, 165, 233, 0.7); }
        100% { box-shadow: 0 0 15px rgba(14, 165, 233, 0.3); }
    }

    @keyframes slideUpFade {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Ambient Hero Banner */
    .hero-banner {
        background: radial-gradient(circle at 10% 20%, #0f172a 0%, #1e1b4b 50%, #311042 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 30px 24px;
        text-align: center;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        animation: pulseGlow 4s infinite ease-in-out;
    }
    
    .hero-banner h1 {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin-bottom: 8px;
    }
    
    .hero-banner p {
        color: #cbd5e1;
        font-size: 15px;
        letter-spacing: 0.2px;
    }

    /* Input Container Glassmorphism */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(12px);
        border-radius: 18px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 10px;
    }

    /* Neon Action Button */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #6366f1 50%, #a855f7 100%) !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 17px !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 0.75rem 1.5rem !important;
        letter-spacing: 0.5px !important;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.01) !important;
        box-shadow: 0 12px 30px rgba(99, 102, 241, 0.5) !important;
    }

    /* Output Alert Cards with Entrance Animation */
    .badge-card {
        padding: 24px;
        border-radius: 16px;
        text-align: center;
        animation: slideUpFade 0.5s ease-out forwards;
        margin-top: 20px;
    }
    
    .churn-risk {
        background: linear-gradient(145deg, rgba(239, 68, 68, 0.12), rgba(185, 28, 28, 0.2));
        border: 2px solid #ef4444;
        color: #fca5a5;
    }

    .churn-risk h2 {
        color: #f87171 !important;
        margin-bottom: 4px;
        font-size: 26px;
    }

    .loyalty-safe {
        background: linear-gradient(145deg, rgba(34, 197, 94, 0.12), rgba(21, 128, 61, 0.2));
        border: 2px solid #22c55e;
        color: #86efac;
    }

    .loyalty-safe h2 {
        color: #4ade80 !important;
        margin-bottom: 4px;
        font-size: 26px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- Load Pipelines -----------------
@st.cache_resource
def load_artifacts():
    scaler_obj = joblib.load("scaler.pkl")
    model_obj = joblib.load("model.pkl")
    return scaler_obj, model_obj

scaler, model = load_artifacts()

# ----------------- Visual Header -----------------
st.markdown("""
<div class="hero-banner">
    <h1>⚡ Customer Retention Analytics</h1>
    <p>Predict client departure probability using real-time ML inference</p>
</div>
""", unsafe_allow_html=True)

# ----------------- Parameter Inputs -----------------
with st.container(border=True):
    col_left, col_right = st.columns(2, gap="medium")

    with col_left:
        cust_age = st.number_input(
            "Customer Age (Years)",
            min_value=10,
            max_value=100,
            value=30,
            step=1,
            help="Demographic age of the account holder"
        )
        cust_tenure = st.number_input(
            "Account Tenure (Months)",
            min_value=0,
            max_value=130,
            value=10,
            step=1,
            help="Total duration of client association in months"
        )

    with col_right:
        cust_gender = st.selectbox(
            "Identified Gender",
            options=["Male", "Female"],
            index=0,
            help="Account holder profile gender"
        )
        cust_bill = st.number_input(
            "Monthly Subscription Charge ($)",
            min_value=3.0,
            value=150.0,
            step=5.0,
            format="%.2f",
            help="Average monthly bill amount"
        )

st.write("")
trigger_inference = st.button("Run Churn Diagnostic ✨", use_container_width=True)

# ----------------- Prediction Workflow -----------------
if trigger_inference:
    # Feature transformations
    gender_encoded = 1 if cust_gender == "Female" else 0
    raw_payload = np.array([[cust_age, gender_encoded, cust_tenure, cust_bill]])
    
    # Preprocessing & Scoring
    normalized_features = scaler.transform(raw_payload)
    outcome_label = model.predict(normalized_features)[0]
    
    st.write("")
    if outcome_label == 1:
        st.markdown("""
        <div class="badge-card churn-risk">
            <h2>⚠️ High Churn Risk Detected</h2>
            <p style="font-size: 16px; margin: 0;">Predicted Outcome: <b>Yes (Customer likely to discontinue)</b></p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="badge-card loyalty-safe">
            <h2>🎉 Stable Account Status</h2>
            <p style="font-size: 16px; margin: 0;">Predicted Outcome: <b>No (Customer likely to retain)</b></p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.caption("ℹ️ Configure client parameters above and launch the diagnostic.")