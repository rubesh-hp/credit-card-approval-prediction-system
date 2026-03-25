import streamlit as st
import joblib
import pandas as pd

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Credit Card Approval Predictor",
    page_icon="💳",
    layout="wide"
)

# -----------------------------
# Load model and scaler
# -----------------------------
model = joblib.load("model/credit_card_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>
/* App background */
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

/* Header */
.title {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    color: white;
    margin-top: 3rem;      
    margin-bottom: 0.5rem; 
}

.subtitle {
    text-align: center;
    color: #cbd5e1;
    font-size: 1rem;
    margin-bottom: 2rem;
}

/* Card */
.card {
    background: rgba(255, 255, 255, 0.07);
    padding: 1.6rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    height: 100%;
}

/* Section */
.section {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 1rem;
    color: white;
}

/* Button */
div.stButton {
    display: flex;
    justify-content: center;
    margin-top: 1rem;
}

div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #06b6d4);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-size: 1rem;
    font-weight: 700;
    min-width: 220px;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
    transform: scale(1.02);
}

/* Metric box */
.metric-box {
    background: rgba(255,255,255,0.05);
    padding: 1rem;
    border-radius: 14px;
    text-align: center;
    margin-bottom: 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}

.metric-title {
    color: #cbd5e1;
    font-size: 0.95rem;
    margin-bottom: 0.4rem;
}

.metric-value {
    color: white;
    font-size: 1.8rem;
    font-weight: 800;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="title">💳 Credit Card Approval Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">A Machine Learning based system to predict credit card approval status</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Layout
# -----------------------------
left_col, right_col = st.columns([1, 1], gap="large")

# -----------------------------
# LEFT SIDE - INPUT FORM
# -----------------------------
with left_col:
    st.markdown('<div class="section">Applicant Details</div>', unsafe_allow_html=True)

    car_owner = st.selectbox("Car Owner", ["Yes", "No"])
    property_owner = st.selectbox("Property Owner", ["Yes", "No"])
    annual_income = st.number_input("Annual Income", min_value=0, step=10000)
    education = st.selectbox(
        "Education Level",
        ["Secondary", "Higher", "Graduate", "Postgraduate"]
    )

    predict_btn = st.button("Check Approval")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# RIGHT SIDE - RESULT
# -----------------------------
with right_col:
    st.markdown('<div class="section">Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        # Encode inputs
        car_owner_encoded = 1 if car_owner == "Yes" else 0
        property_owner_encoded = 1 if property_owner == "Yes" else 0

        education_map = {
            "Secondary": 0,
            "Higher": 1,
            "Graduate": 2,
            "Postgraduate": 3
        }
        education_encoded = education_map[education]

        # Prepare input
        input_df = pd.DataFrame(
            [[car_owner_encoded, property_owner_encoded, annual_income, education_encoded]],
            columns=["Car_Owner", "Propert_Owner", "Annual_income", "EDUCATION"]
        )

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1] * 100

        # Hybrid logic
        final_approval = prediction == 1 or (
            car_owner_encoded == 1 and
            property_owner_encoded == 1 and
            annual_income >= 500000 and
            education_encoded >= 1
        )

        # Strength label
        if probability >= 70:
            strength = "Strong"
        elif probability >= 40:
            strength = "Moderate"
        else:
            strength = "Low"

        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Approval Probability</div>
                <div class="metric-value">{probability:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">Profile Strength</div>
                <div class="metric-value">{strength}</div>
            </div>
        """, unsafe_allow_html=True)

        st.progress(min(int(probability), 100))

        if final_approval:
            st.success("✅ Credit Card Approved")
            st.info("This applicant has a strong financial profile based on ownership, income, and education.")
        else:
            st.error("❌ Credit Card Rejected")
            st.warning("This applicant may not meet the approval criteria based on the entered profile.")

    else:
        st.info("Enter applicant details on the left and click **Check Approval** to view the prediction result.")

    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown(
    '<div class="footer">Built with Python • Scikit-learn • Streamlit</div>',
    unsafe_allow_html=True
)