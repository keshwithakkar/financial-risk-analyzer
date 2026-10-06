import streamlit as st
import joblib
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Financial Risk Analyzer",
    page_icon="💰",
    layout="wide"
)

# Load trained files
BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "models" / "random_forest_model.pkl")
encoder = joblib.load(BASE_DIR / "models" / "encoder.pkl")
scaler = joblib.load(BASE_DIR / "models" / "scaler.pkl")

st.title("💰 Financial Risk Analyzer")
st.subheader("AI-Powered Loan Default Risk Prediction")

st.write(
    "Enter applicant and loan details to estimate the probability "
    "of loan default using Machine Learning."
)

st.success("Machine Learning model loaded successfully!")

# 2 columns
col1, col2 = st.columns(2)

with col1:
    # Add the input form
    st.header("👤 Applicant Details")

    person_age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    person_income = st.number_input(
        "Annual Income",
        min_value=0,
        value=50000
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )

    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0,
        max_value=50.0,
        value=3.0
    )

with col2:
    st.header("🏦 Loan Details")

    # Add loan details
    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "PERSONAL",
            "DEBTCONSOLIDATION",
            "HOMEIMPROVEMENT"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_amnt = st.number_input(
        "Loan Amount",
        min_value=0,
        value=10000
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=10.5
    )

# loan/credit/risk related details
st.header("📊 Credit & Risk Details")

loan_percent_income = st.number_input(
    "Loan Percent of Income",
    min_value=0.0,
    max_value=1.0,
    value=0.20
)

cb_person_default_on_file = st.selectbox(
    "Previous Default on File",
    ["N", "Y"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0.0,
    max_value=50.0,
    value=4.0
)

# add predict button
if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        "person_age": person_age,
        "person_income": person_income,
        "person_home_ownership": person_home_ownership,
        "person_emp_length": person_emp_length,
        "loan_intent": loan_intent,
        "loan_grade": loan_grade,
        "loan_amnt": loan_amnt,
        "loan_int_rate": loan_int_rate,
        "loan_percent_income": loan_percent_income,
        "cb_person_default_on_file": cb_person_default_on_file,
        "cb_person_cred_hist_length": cb_person_cred_hist_length
    }])

    numerical_data = scaler.transform(
        input_data[
            [
                "person_age",
                "person_income",
                "person_emp_length",
                "loan_amnt",
                "loan_int_rate",
                "loan_percent_income",
                "cb_person_cred_hist_length"
            ]
        ]
    )

    categorical_data = encoder.transform(
        input_data[
            [
                "person_home_ownership",
                "loan_intent",
                "loan_grade",
                "cb_person_default_on_file"
            ]
        ]
    )

    from scipy.sparse import hstack

    final_data = hstack([
        numerical_data,
        categorical_data
    ])

    prediction = model.predict(final_data)[0]
    probability = model.predict_proba(final_data)[0][1]

    probability_percent = probability * 100

    st.subheader("📊 Risk Assessment")

    if prediction == 0:
        st.success("🟢 Low Risk")
    else:
        st.error("🔴 High Risk")

    st.metric(
        "Estimated Default Probability",
        f"{probability_percent:.2f}%"
    )

    # progress bar
    st.progress(float(probability))

    # explaination
    if prediction == 0:
        st.info(
            "The applicant is predicted to have a low risk of loan default "
            "based on the information provided."
        )
    else:
        st.warning(
            "The applicant is predicted to have a high risk of loan default "
            "based on the information provided."
        )

# add disclaimer
st.divider()

st.caption(
    "Disclaimer: This tool provides an ML-based risk estimate for demonstration "
    "purposes only. It should not be used as the sole basis for financial decisions."
)