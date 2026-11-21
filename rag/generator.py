import os
from dotenv import load_dotenv
from google import genai

try:
    from .retriever import retrieve_policy
except ImportError:
    from retriever import retrieve_policy


# Load Gemini API key
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(BASE_DIR, ".env"))

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("Key not found")

client = genai.Client(api_key=api_key)


def generate_risk_explanation(input_data, rf_result, pytorch_result):

    # Retrieve relevant policy sections
    query = (
        f"Credit risk assessment, income, loan amount, "
        f"loan grade, previous default, and repayment risk. "
        f"Applicant income: {input_data.get('person_income')}. "
        f"Loan amount: {input_data.get('loan_amnt')}. "
        f"Previous default: {input_data.get('cb_person_default_on_file')}."
    )

    policy_sections = retrieve_policy(query)

    policy_context = "\n\n".join(policy_sections)

    # Calculate probabilities as percentages
    rf_percentage = rf_result["probability"] * 100
    pytorch_percentage = pytorch_result["probability"] * 100

    probability_difference = abs(
        rf_percentage - pytorch_percentage
    )

    prompt = f"""
You are a financial risk analysis assistant.

Analyze the applicant using only the information provided below.

APPLICANT INFORMATION:
{input_data}

RANDOM FOREST RESULT:
Risk: {rf_result["risk"]}
Default probability: {rf_percentage:.2f}%

PYTORCH RESULT:
Risk: {pytorch_result["risk"]}
Default probability: {pytorch_percentage:.2f}%

Difference between probabilities: {probability_difference:.2f} percentage points.

RETRIEVED POLICY DOCUMENTS:
{policy_context}

IMPORTANT RULES:
1. Explain the results in simple, beginner-friendly language.
2. Use the exact model risk classifications and probabilities.
3. Clearly explain that the two models have different probability estimates.
4. Do not describe a large probability difference as minor.
5. Do not calculate or invent financial ratios.
6. Use the provided loan_percent_income value directly if discussing loan-to-income.
7. Do not assume that a loan grade or loan intent guarantees creditworthiness.
8. Do not invent policy rules or facts.
9. Base policy explanations only on the retrieved policy documents.
10. Do not claim guaranteed loan approval or rejection.

FORMAT YOUR RESPONSE USING THESE HEADINGS:

### 1. Overall Risk Assessment
Explain both model classifications.

### 2. Important Applicant Factors
Discuss relevant applicant details without inventing conclusions.

### 3. Supporting Policy Information
Explain how the retrieved policy sections relate to this applicant.

### 4. Model Comparison
Compare both probabilities and explain their difference in percentage points.

### 5. Disclaimer
State that this is an educational estimate, not financial advice or a lending decision.

Keep the explanation clear, factual, and concise.
"""

    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=prompt
    )

    return {
        "explanation": response.text,
        "policy_sections": policy_sections
    }