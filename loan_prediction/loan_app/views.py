from django.shortcuts import render
from .forms import LoanForm
import joblib
import numpy as np


def loan_prediction(request):
    # Завантаження моделі
    model = joblib.load('loan_app/loan_model.pkl')

    purposes = {
    'credit_card': 0,
    'debt_consolidation': 1,
    'home_improvement': 2,
    'major_purchase': 3,
    'small_business': 4
    }

    prediction = None
    if request.method == 'POST':
        form = LoanForm(request.POST)
        if form.is_valid():
            data = form.cleaned_data
            purpose = purposes[data['purpose']]
            data['purpose'] = purpose

            log_annual_inc = np.log(data['annual_income'])
            data['log_annual_inc'] = log_annual_inc

            data['dti'] = float(data['installment'])/(float(data['annual_income'])/12)

            features = np.array([
                # data['purpose'],
                data['int_rate'],
                data['installment'],
                data['log_annual_inc'],
                data['dti'],
                data['fico'],

            ]).reshape(1, -1)

            prediction = model.predict(features)
            prediction = 'Схвалено' if prediction == 0 else 'Не схвалено'
    else:
        form = LoanForm()

    return render(request, 'loan_app/loan_form.html', {'form': form, 'prediction': prediction})
