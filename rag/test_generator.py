from generator import generate_risk_explanation

# Sample applicant information
input_data = {
    "person_age": 28,
    "person_income": 45000,
    "person_home_ownership": "RENT",
    "person_emp_length": 3,
    "loan_intent": "PERSONAL",
    "loan_grade": "B",
    "loan_amnt": 10000,
    "loan_int_rate": 11.5,
    "loan_percent_income": 0.22,
    "cb_person_default_on_file": "N",
    "cb_person_cred_hist_length": 5
}

# Sample model predictions
rf_result = {
    "risk": "Low Risk",
    "probability": 0.15
}

pytorch_result = {
    "risk": "Low Risk",
    "probability": 0.20
}

result = generate_risk_explanation(
    input_data,
    rf_result,
    pytorch_result
)

print("\nGENERATED RISK EXPLANATION:\n")
print(result["explanation"])

print("\nRETRIEVED POLICY SECTIONS:\n")
for section in result["policy_sections"]:
    print(section)
    print("----------------------")