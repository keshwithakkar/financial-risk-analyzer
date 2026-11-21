from django.urls import path
from .views import predict, prediction_history

urlpatterns = [
    path("predict/", predict, name="predict"),
    path("history/", prediction_history, name="prediction_history"),
]