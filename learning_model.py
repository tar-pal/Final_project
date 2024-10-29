import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Завантаження підготовлених даних
data = pd.read_csv('prepared_loan_data.csv')

# Поділ на ознаки (X) та мітки (y)
X = data.drop('not.fully.paid', axis=1)
y = data['not.fully.paid']

# Поділ на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Визначення числових та категоріальних стовпців
numerical_columns = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_columns = X.select_dtypes(include=['object']).columns.tolist()

# Попередня обробка даних
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Побудова Pipeline з RandomForestClassifier
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])
# Налаштування параметрів для GridSearchCV
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Налаштування GridSearchCV з перехресною валідацією
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Виведення найкращих параметрів
print("Найкращі параметри моделі:")
print(grid_search.best_params_)

# Прогнозування на тестовому наборі
y_pred = grid_search.predict(X_test)

# Оцінка моделі
print("\nЗвіт класифікації:")
print(classification_report(y_test, y_pred))

# Побудова матриці плутанини
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Paid', 'Not Paid'], yticklabels=['Paid', 'Not Paid'])
plt.xlabel('Прогнозоване значення')
plt.ylabel('Фактичне значення')
plt.title('Матриця плутанини')
plt.show()
model = RandomForestClassifier()


# Важливість ознак з найкращої моделі RandomForest
best_model = grid_search.best_estimator_.named_steps['classifier']
feature_importances = best_model.feature_importances_

# Візуалізація важливості ознак
feature_names = X.columns
importances = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title('Важливість ознак у моделі RandomForest')
plt.xlabel('Важливість')
plt.ylabel('Ознаки')
plt.tight_layout()
plt.show()

# Розподіл основних числових змінних
numerical_columns = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico']

plt.figure(figsize=(16, 12))
for i, col in enumerate(numerical_columns, 1):
    plt.subplot(3, 2, i)
    sns.histplot(data[col], bins=30, kde=True, color='blue')
    plt.title(f'Розподіл змінної: {col}')
    plt.xlabel(col)
    plt.ylabel('Частота')

plt.tight_layout()
plt.show()

# Збереження моделі
joblib.dump(model, 'loan_model.pkl')

