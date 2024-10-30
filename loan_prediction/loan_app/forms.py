from django import forms

class LoanForm(forms.Form):
    int_rate = forms.FloatField(label='Відсоткова ставка')
    installment = forms.FloatField(label='Щомісячний платіж')
    annual_income = forms.FloatField(label='Річний дохід', show_hidden_initial=False)
    # dti = forms.FloatField(label='Співвідношення боргу до доходу (DTI)')
    fico = forms.IntegerField(label='FICO бал')
    purpose = forms.ChoiceField(
        choices=[('credit_card', 'Кредитна карта'),
                 ('debt_consolidation', 'Консолідація боргу'),
                 ('home_improvement', 'Ремонт будинку'),
                 ('major_purchase', 'Велика покупка'),
                 ('small_business', 'Малий бізнес')],
        label='Ціль кредиту'
    )
