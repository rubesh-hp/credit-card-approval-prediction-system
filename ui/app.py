import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model/credit_card_model.pkl")
scaler = joblib.load("model/scaler.pkl")

st.set_page_config(page_title="Credit Card Approval", page_icon="💳")
st.title("💳 Credit Card Approval Prediction System")
st.write("Enter applicant details to check credit card approval status.")

# -----------------------------
# User Inputs
# -----------------------------
car_owner = st.selectbox("Car Owner", ["Yes", "No"])
property_owner = st.selectbox("Property Owner", ["Yes", "No"])
annual_income = st.number_input("Annual Income", min_value=0)
education = st.selectbox(
    "Education Level",
    ["Secondary", "Higher", "Graduate", "Postgraduate"]
)

# -----------------------------
# Encode Inputs
# -----------------------------
car_owner = 1 if car_owner == "Yes" else 0
property_owner = 1 if property_owner == "Yes" else 0

education_map = {
    "Secondary": 0,
    "Higher": 1,
    "Graduate": 2,
    "Postgraduate": 3
}
education = education_map[education]

# -----------------------------
# Prediction
# -----------------------------
if st.button("Check Credit Approval"):
    input_data = np.array([[car_owner, property_owner, annual_income, education]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.success("✅ Credit Card Approved")
    else:
        st.error("❌ Credit Card Rejected")
