import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Завантаження даних
file_path = 'loan_data.csv'  # Заміни на шлях до твого файлу
loan_data = pd.read_csv(file_path)

# Формуємо датафрейм
data_frame = pd.DataFrame(loan_data)

# Трансформуємо стовбчик purpose
data_frame['purpose'] = data_frame['purpose'].astype('category').cat.codes

loan_data = data_frame

# Аналіз даних
print("Опис даних:")
print(loan_data.info())  # Основна інформація про датасет
print("\nПерші 5 рядків:")
print(loan_data.head())  # Перші 5 рядків датасету

# Перевірка кореляції

# Побудова кореляційної матриці
sns.set(style="whitegrid")
correlation_matrix = loan_data.corr()

# Побудова теплової карти кореляцій
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
heatmap.set_title('Correlation Matrix of Loan Data', fontsize=16)
plt.show()

# Підготовка даних
encoded_data =  loan_data

# Нормалізація числових змінних
scaler = StandardScaler()
numerical_columns = ['int.rate', 'installment', 'log.annual.inc', 'dti',
                     'fico', 'days.with.cr.line', 'revol.bal', 'revol.util',
                     'inq.last.6mths', 'delinq.2yrs', 'pub.rec']

encoded_data[numerical_columns] = scaler.fit_transform(encoded_data[numerical_columns])

# Результат
print("\nПерші 5 рядків підготовлених даних:")
print(encoded_data.head())

# Збереження підготовлених даних у новий CSV файл
encoded_data.to_csv('prepared_loan_data.csv', index=False)
print("\nПідготовлені дані збережено у файл 'prepared_loan_data.csv'.")
