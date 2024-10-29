from joblib import load
import numpy as np

# Завантаження моделі з файлу
try:
    model = load("loan_model.pkl")
    print("Модель успішно завантажена!")
except Exception as e:
    print("Помилка при завантаженні моделі:", e)
    model = None

# Перевірка моделі, якщо вона завантажена успішно
if model:
    # Перевірка, чи модель містить назви ознак
    if hasattr(model, 'feature_names_in_'):
        feature_names = model.feature_names_in_
        print("Назви ознак, збережені в моделі:", feature_names)
    else:
        print("Назви ознак не збережені в моделі.")
        feature_names = None

    # Перевірка важливості ознак, якщо це підтримується моделлю
    if feature_names is not None and hasattr(model, 'feature_importances_'):
        print("\nВажливість ознак:")
        for feature_name, importance in zip(feature_names, model.feature_importances_):
            print(f"Ознака '{feature_name}': Важливість {importance}")
    elif hasattr(model, 'feature_importances_'):
        print("\nВажливість ознак (імена відсутні):")
        for i, importance in enumerate(model.feature_importances_):
            print(f"Ознака {i + 1}: Важливість {importance}")
    else:
        print("Ця модель не підтримує атрибут 'feature_importances_'.")

    # Перегляд параметрів моделі
    print("\nПараметри моделі:")
    print(model.get_params())

    # Створення тестового запису для перевірки форми
    try:
        X_example = np.random.rand(1, model.n_features_in_)  # тестові дані з правильною кількістю ознак
        prediction = model.predict(X_example)
        print("\nПередбачення для тестового запису:", prediction)
    except Exception as e:
        print("\nПомилка при перевірці тестового запису:", e)
