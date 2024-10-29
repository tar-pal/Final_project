from django.shortcuts import render
from .forms import LoanForm
import joblib
import numpy as np

# Завантаження моделі
model = joblib.load('loan_app/loan_model.pkl')


def loan_prediction(request):
    prediction = None
    if request.method == 'POST':
        form = LoanForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            features = np.array([
                data['int_rate'],
                data['installment'],
                data['log_annual_inc'],
                data['dti'],
                data['fico']
            ]).reshape(1, -1)

            prediction = model.predict(features)
            prediction = 'Схвалено' if prediction == 0 else 'Не схвалено'
    else:
        form = LoanForm()

    return render(request, 'loan_app/loan_form.html', {'form': form, 'prediction': prediction})
