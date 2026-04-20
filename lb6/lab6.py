import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('abalone.csv')
print(f"Размер набора данных: {df.shape}")

X = df.drop('Rings', axis=1)
y = df['Rings']

feature = 'Length'
X_simple = X[[feature]]
print(f"Признак для одномерной регрессии: {feature}")
print(f"Первые 5 значений: {X_simple[:5].values.ravel()}")
print(f"Соответствующие целевые значения: {y[:5].values}")

X_train, X_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)
print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(f"Коэффициент (наклон): {regressor.coef_[0]:.4f}")
print(f"Свободный член (интерцепт): {regressor.intercept_:.4f}")

y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"\nMSE на обучающей выборке: {train_mse:.4f}")
print(f"MSE на тестовой выборке: {test_mse:.4f}")
print(f"R² на обучающей выборке: {train_r2:.4f}")
print(f"R² на тестовой выборке: {test_r2:.4f}")

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, alpha=0.5, color='blue', label='Обучающие данные')
plt.plot(X_train, regressor.predict(X_train), color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Длина (Length)')
plt.ylabel('Количество колец (Rings)')
plt.title('Одномерная линейная регрессия (обучающая выборка)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, alpha=0.5, color='green', label='Тестовые данные')
plt.plot(X_test, regressor.predict(X_test), color='red', linewidth=2, label='Линия регрессии')
plt.xlabel('Длина (Length)')
plt.ylabel('Количество колец (Rings)')
plt.title('Одномерная линейная регрессия (тестовая выборка)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab6_univariate_regression.png', dpi=150)
print("График сохранен как 'lab6_univariate_regression.png'")

new_abalone = np.array([[0.55]])
prediction = regressor.predict(new_abalone)
print(f"\nДля моллюска с длиной 0.55 мм предсказано колец: {prediction[0]:.2f}")
print(f"Округленное значение: {int(round(prediction[0]))}")