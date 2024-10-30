import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Вибір лише необхідних ознак
features = ['purpose', 'int.rate', 'installment', 'log.annual.inc', 'fico']
X = data[features]
y = data['not.fully.paid']

# Поділ на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Визначення числових та категоріальних стовпців
numerical_columns = ['int.rate', 'installment', 'log.annual.inc', 'fico']
categorical_columns = ['purpose']

# Попередня обробка даних
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ]
)

# Навчання preprocessor на даних X_train
preprocessor.fit(X_train)

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
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
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

# Важливість ознак з найкращої моделі RandomForest
best_model = grid_search.best_estimator_.named_steps['classifier']
feature_importances = best_model.feature_importances_

# Отримання назв ознак з попередньо навченого preprocessor
numerical_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_columns)
feature_names = np.concatenate([numerical_features, categorical_features])

# Візуалізація важливості ознак
importances = pd.Series(feature_importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title('Важливість ознак у моделі RandomForest')
plt.xlabel('Важливість')
plt.ylabel('Ознаки')
plt.tight_layout()
plt.show()

# Збереження моделі
joblib.dump(best_model, 'loan_model.pkl')