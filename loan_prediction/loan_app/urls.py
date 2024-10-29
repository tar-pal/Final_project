from django.urls import path
from . import views

urlpatterns = [
    path('', views.loan_prediction, name='loan_prediction'),
]
