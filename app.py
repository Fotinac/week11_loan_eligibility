"""
app.py
Streamlit interface for collecting user inputs and displaying loan eligibility predictions.
Modularized UI — logic is abstracted to main.py and other modules.
"""

import streamlit as st
from main import predict_loan_eligibility, show_feature_importance_chart

st.title("🏦 Credit Loan Eligibility Predictor")

# Collect inputs
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income = st.number_input("Applicant Income", min_value=0)
co_income = st.number_input("Coapplicant Income", min_value=0)
loan_amt = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Term (months)", [360, 180, 240, 120, 60])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

if st.button("Predict"):
    user_input = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": income,
        "CoapplicantIncome": co_income,
        "LoanAmount": loan_amt,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    result = predict_loan_eligibility(user_input)

    st.subheader("Prediction Result:")
    st.success("✅ You are likely eligible!" if result == 1 else "❌ Sorry, not eligible.")

    st.subheader("Feature Importance")
    show_feature_importance_chart()
