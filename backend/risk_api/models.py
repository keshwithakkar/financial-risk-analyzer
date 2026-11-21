from django.db import models


class Prediction(models.Model):

    person_age = models.IntegerField()
    person_income = models.FloatField()
    person_home_ownership = models.CharField(max_length=20)
    person_emp_length = models.FloatField()

    loan_intent = models.CharField(max_length=30)
    loan_grade = models.CharField(max_length=5)
    loan_amnt = models.FloatField()
    loan_int_rate = models.FloatField()

    loan_percent_income = models.FloatField()
    cb_person_default_on_file = models.CharField(max_length=1)
    cb_person_cred_hist_length = models.FloatField()

    random_forest_risk = models.CharField(max_length=20)
    random_forest_probability = models.FloatField()

    pytorch_risk = models.CharField(max_length=20)
    pytorch_probability = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.random_forest_risk} - {self.created_at}"