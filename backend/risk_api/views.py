import sys
import os

import joblib
import pandas as pd
from scipy.sparse import hstack

from rest_framework.decorators import api_view
from rest_framework.response import Response

from .serializers import (
    LoanRiskSerializer,
    PredictionHistorySerializer
)

from .models import Prediction


# Project root
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

sys.path.append(PROJECT_ROOT)

from src.pytorch_predict import predict_risk
from rag.generator import generate_risk_explanation


# Load Random Forest files
BASE_DIR = PROJECT_ROOT

rf_model = joblib.load(
    os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
)

rf_encoder = joblib.load(
    os.path.join(BASE_DIR, "models", "encoder.pkl")
)

rf_scaler = joblib.load(
    os.path.join(BASE_DIR, "models", "scaler.pkl")
)


@api_view(["POST"])
def predict(request):

    try:
        serializer = LoanRiskSerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                {"errors": serializer.errors},
                status=400
            )

        input_data = serializer.validated_data

        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # -------------------------
        # Random Forest Prediction
        # -------------------------

        numerical_data = rf_scaler.transform(
            df[
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

        categorical_data = rf_encoder.transform(
            df[
                [
                    "person_home_ownership",
                    "loan_intent",
                    "loan_grade",
                    "cb_person_default_on_file"
                ]
            ]
        )

        final_data = hstack([
            numerical_data,
            categorical_data
        ])

        rf_prediction = rf_model.predict(final_data)[0]

        rf_probability = float(
            rf_model.predict_proba(final_data)[0][1]
        )

        if rf_prediction == 0:
            rf_risk = "Low Risk"
        else:
            rf_risk = "High Risk"

        # -------------------------
        # PyTorch Prediction
        # -------------------------

        pytorch_result = predict_risk(input_data)

        # -------------------------
        # Save prediction to PostgreSQL
        # -------------------------

        Prediction.objects.create(
            person_age=input_data["person_age"],
            person_income=input_data["person_income"],
            person_home_ownership=input_data["person_home_ownership"],
            person_emp_length=input_data["person_emp_length"],
            loan_intent=input_data["loan_intent"],
            loan_grade=input_data["loan_grade"],
            loan_amnt=input_data["loan_amnt"],
            loan_int_rate=input_data["loan_int_rate"],
            loan_percent_income=input_data["loan_percent_income"],
            cb_person_default_on_file=input_data["cb_person_default_on_file"],
            cb_person_cred_hist_length=input_data["cb_person_cred_hist_length"],

            random_forest_risk=rf_risk,
            random_forest_probability=rf_probability,

            pytorch_risk=pytorch_result["risk"],
            pytorch_probability=pytorch_result["probability"]
        )

        # -------------------------
        # Gemini + RAG Explanation
        # -------------------------

        explanation = None
        policy_sections = []
        explanation_error = None

        try:
            rf_result = {
                "risk": rf_risk,
                "probability": rf_probability
            }

            explanation_result = generate_risk_explanation(
                input_data,
                rf_result,
                pytorch_result
            )

            explanation = explanation_result["explanation"]
            policy_sections = explanation_result["policy_sections"]

        except Exception as e:
            explanation_error = str(e)

        # -------------------------
        # Return all results
        # -------------------------

        return Response({
            "random_forest": {
                "risk": rf_risk,
                "probability": rf_probability
            },

            "pytorch": {
                "risk": pytorch_result["risk"],
                "probability": pytorch_result["probability"]
            },

            "explanation": explanation,
            "policy_sections": policy_sections,
            "explanation_error": explanation_error
        })

    except Exception as e:

        return Response(
            {"error": str(e)},
            status=400
        )


@api_view(["GET"])
def prediction_history(request):

    predictions = Prediction.objects.all().order_by("-created_at")

    serializer = PredictionHistorySerializer(
        predictions,
        many=True
    )

    return Response(serializer.data)