
import streamlit as st
import pandas as pd
import requests

st.set_page_config(
    page_title="Financial Risk Analyzer",
    page_icon="💰",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/api"

st.title("💰 Financial Risk Analyzer")
st.subheader("AI-Powered Loan Default Risk Prediction")

st.write(
    "Enter applicant and loan details to estimate the probability "
    "of loan default." # using Machine Learning."
)

# st.info("ML models are served through the Django REST API.")

col1, col2 = st.columns(2)

with col1:
    st.header("👤 Applicant Details")

    person_age = st.number_input(
        "Age", min_value=18, max_value=100, value=25
    )

    person_income = st.number_input(
        "Annual Income", min_value=0, value=50000
    )

    person_home_ownership = st.selectbox(
        "Home Ownership",
        ["RENT", "MORTGAGE", "OWN", "OTHER"]
    )

    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0, max_value=50.0, value=3.0
    )

with col2:
    st.header("🏦 Loan Details")

    loan_intent = st.selectbox(
        "Loan Intent",
        [
            "EDUCATION", "MEDICAL", "VENTURE",
            "PERSONAL", "DEBTCONSOLIDATION",
            "HOMEIMPROVEMENT"
        ]
    )

    loan_grade = st.selectbox(
        "Loan Grade",
        ["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_amnt = st.number_input(
        "Loan Amount", min_value=0, value=10000
    )

    loan_int_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0, max_value=100.0, value=10.5
    )

st.header("📊 Credit & Risk Details")

loan_percent_income = st.number_input(
    "Loan Percent of Income",
    min_value=0.0, max_value=1.0, value=0.20
)

cb_person_default_on_file = st.selectbox(
    "Previous Default on File", ["N", "Y"]
)

cb_person_cred_hist_length = st.number_input(
    "Credit History Length (years)",
    min_value=0.0, max_value=50.0, value=4.0
)

if st.button("Predict Risk"):

    api_data = {
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
    }

    try:
        response = requests.post(
            f"{API_URL}/predict/",
            json=api_data,
            timeout=120
        )

        if response.status_code != 200:
            st.error(f"Prediction failed: {response.text}")
            st.stop()

        api_result = response.json()
        # st.success("Django API prediction successful!")

    except requests.exceptions.RequestException as e:
        st.error(f"Unable to connect: {e}") # to Django API: {e}")
        st.stop()

    # Random Forest Results
    st.subheader("📊 Random Forest Model Prediction")

    rf_result = api_result["random_forest"]

    if rf_result["risk"] == "Low Risk":
        st.success("🟢 Low Risk")
    else:
        st.error("🔴 High Risk")

    st.metric(
        "Estimated Default Probability",
        f'{rf_result["probability"] * 100:.2f}%'
    )

    st.progress(float(rf_result["probability"]))

    st.divider()

    # PyTorch Results
    st.subheader("📊 PyTorch Model Prediction")

    pytorch_result = api_result["pytorch"]

    # st.write("Risk:", pytorch_result["risk"])

    if pytorch_result['risk'] == "Low Risk":
        st.success("🟢 Low Risk")
    else:
        st.error("🔴 High Risk")

    st.metric(
        "Estimated Default Probability:",
        f'{pytorch_result["probability"] * 100:.2f}%'
    )

    st.progress(float(pytorch_result["probability"]))

    st.divider()

    # Model Comparison
    st.subheader("🤖 Model Comparison")

    comparison_col1, comparison_col2 = st.columns(2)

    with comparison_col1:
        st.metric(
            "Random Forest",
            f'{rf_result["probability"] * 100:.2f}%'
        )

    with comparison_col2:
        st.metric(
            "PyTorch",# Neural Network",
            f'{pytorch_result["probability"] * 100:.2f}%'
        )

    st.divider()

    # Gemini Explanation
    st.subheader("🤖 AI-Generated Risk Explanation")

    explanation = api_result.get("explanation")

    if explanation:
        st.markdown(explanation)
    else:
        st.warning(
            "AI explanation is currently unavailable."
        )

        if api_result.get("explanation_error"):
            st.caption(api_result["explanation_error"])

    st.divider()

    # RAG Policy Sections
    st.subheader("📚 Retrieved Financial Policy Information")

    policy_sections = api_result.get("policy_sections", [])

    if policy_sections:
        for index, section in enumerate(policy_sections, start=1):
            with st.expander(f"📄 Retrieved Policy {index}", expanded=True):
                st.markdown(section)
    else:
        st.info("No policy sections were retrieved.")

# Prediction History
st.divider()

st.subheader("📜 Prediction History")

try:
    history_response = requests.get(
        f"{API_URL}/history/",
        timeout=30
    )

    if history_response.status_code == 200:
        history_data = history_response.json()

        if history_data:
            history_df = pd.DataFrame(history_data)

            history_df = history_df[
                [
                    "created_at",
                    "random_forest_risk",
                    "random_forest_probability",
                    "pytorch_risk",
                    "pytorch_probability"
                ]
            ]

            history_df = history_df.rename(
                columns={
                    "created_at": "Date",
                    "random_forest_risk": "Random Forest Risk",
                    "random_forest_probability": "RF Probability",
                    "pytorch_risk": "PyTorch Risk",
                    "pytorch_probability": "PyTorch Probability"
                }
            )

            history_df["RF Probability"] = (
                history_df["RF Probability"] * 100
            ).round(2)

            history_df["PyTorch Probability"] = (
                history_df["PyTorch Probability"] * 100
            ).round(2)

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True
            )

        else:
            st.info("No prediction history available.")

    else:
        st.error("Unable to load prediction history.")

except requests.exceptions.RequestException:
    # st.error("Unable to connect to Django API for prediction history.")
    st.error("Unable to fetch prediction history.")

st.divider()

st.caption(
    "Disclaimer: This tool provides risk estimate for "
    "demonstration purposes only. It should not be used as the sole "
    "basis for financial decisions."
)