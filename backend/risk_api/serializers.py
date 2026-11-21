from rest_framework import serializers
from .models import Prediction


class LoanRiskSerializer(serializers.Serializer):

    person_age = serializers.IntegerField(
        min_value=18,
        max_value=100
    )

    person_income = serializers.FloatField(
        min_value=0
    )

    person_home_ownership = serializers.ChoiceField(
        choices=["RENT", "MORTGAGE", "OWN", "OTHER"]
    )

    person_emp_length = serializers.FloatField(
        min_value=0,
        max_value=50
    )

    loan_intent = serializers.ChoiceField(
        choices=[
            "EDUCATION",
            "MEDICAL",
            "VENTURE",
            "PERSONAL",
            "DEBTCONSOLIDATION",
            "HOMEIMPROVEMENT"
        ]
    )

    loan_grade = serializers.ChoiceField(
        choices=["A", "B", "C", "D", "E", "F", "G"]
    )

    loan_amnt = serializers.FloatField(
        min_value=0
    )

    loan_int_rate = serializers.FloatField(
        min_value=0,
        max_value=100
    )

    loan_percent_income = serializers.FloatField(
        min_value=0,
        max_value=1
    )

    cb_person_default_on_file = serializers.ChoiceField(
        choices=["N", "Y"]
    )

    cb_person_cred_hist_length = serializers.FloatField(
        min_value=0,
        max_value=50
    )

class PredictionHistorySerializer(serializers.ModelSerializer):

    class Meta:
        model = Prediction
        fields = "__all__"