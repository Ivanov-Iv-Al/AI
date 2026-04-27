import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('abalone.csv')

print("Первые 5 строк данных:")
print(dataset.head())
print(f"\nРазмер набора данных: {dataset.shape}")
print("\nИнформация о данных:")
print(dataset.info())

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

print(f"\nМатрица признаков (Length): {X.shape}")
print(f"Целевая переменная (Rings): {y.shape}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='red', s=10, alpha=0.5, label='Исходные данные')
plt.xlabel('Длина моллюска (Length)', fontsize=12)
plt.ylabel('Количество колец (Rings)', fontsize=12)
plt.title('Зависимость возраста моллюска от длины', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig('lab8_initial_data.png', dpi=150)
plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_linear = lin_reg.predict(X)

mse_linear = mean_squared_error(y, y_pred_linear)
r2_linear = r2_score(y, y_pred_linear)

print("\nЛинейная регрессия на Abalone (степень 1)")
print(f"MSE: {mse_linear:.4f}")
print(f"R²: {r2_linear:.4f}")
print(f"Коэффициент: {lin_reg.coef_[0]:.4f}")
print(f"Свободный член: {lin_reg.intercept_:.4f}")


def polynomial_regression(degree, X, y):
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    lin_reg_2 = LinearRegression()
    lin_reg_2.fit(X_poly, y)
    y_pred = lin_reg_2.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return lin_reg_2, poly_reg, y_pred, mse, r2


degrees = [1, 2, 3, 4, 5, 6, 7, 8]
results = []

print("\n=== Полиномиальная регрессия для Abalone ===")
for d in degrees:
    model, transformer, pred, mse, r2 = polynomial_regression(d, X, y)
    results.append({
        'degree': d,
        'mse': mse,
        'r2': r2,
        'model': model,
        'transformer': transformer
    })
    print(f"Степень {d}: MSE = {mse:.4f}, R² = {r2:.6f}")

X_grid = np.arange(min(X), max(X), 0.005).reshape(-1, 1)

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for idx, d in enumerate(degrees):
    poly_reg = PolynomialFeatures(degree=d)
    X_poly_grid = poly_reg.fit_transform(X_grid)

    model, _, _, _, _ = polynomial_regression(d, X, y)
    y_pred_grid = model.predict(poly_reg.fit_transform(X_grid))

    axes[idx].scatter(X, y, color='red', s=10, alpha=0.5, label='Данные')
    axes[idx].plot(X_grid, y_pred_grid, color='blue', linewidth=2,
                   label=f'Полином степени {d}')
    axes[idx].set_xlabel('Length', fontsize=10)
    axes[idx].set_ylabel('Rings', fontsize=10)
    axes[idx].set_title(f'Степень {d}, R²={results[idx]["r2"]:.4f}', fontsize=10)
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle('Полиномиальная регрессия для Abalone (зависимость Rings от Length)', fontsize=16)
plt.tight_layout()
plt.savefig('lab8_polynomial_comparison.png', dpi=150)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nОценка на тестовой выборке")
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")

test_results = []
for d in degrees:
    poly_reg = PolynomialFeatures(degree=d)
    X_train_poly = poly_reg.fit_transform(X_train)
    X_test_poly = poly_reg.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)

    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)

    test_results.append({
        'degree': d,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_r2': train_r2,
        'test_r2': test_r2
    })
    print(
        f"Степень {d}: Train R²={train_r2:.4f}, Test R²={test_r2:.4f} | Train MSE={train_mse:.4f}, Test MSE={test_mse:.4f}")

X_test_new = np.array([0.6, 0.65, 0.7, 0.75, 0.8]).reshape(-1, 1)

print("\nПредсказания для новых значений длины")
print("\nСтепень | Предсказанные возраста (Rings) для Length=0.6,0.65,0.7,0.75,0.8")

for d in [1, 2, 3, 4, 5, 6, 7, 8]:
    poly_reg = PolynomialFeatures(degree=d)
    X_test_poly = poly_reg.fit_transform(X_test_new)
    model, _, _, _, _ = polynomial_regression(d, X, y)
    predictions = model.predict(X_test_poly)
    print(f"  {d:2d}     | {', '.join([f'{p:.1f}' for p in predictions])}")

optimal_degree = 3
poly_reg_opt = PolynomialFeatures(degree=optimal_degree)
X_poly_opt = poly_reg_opt.fit_transform(X)
lin_reg_opt = LinearRegression()
lin_reg_opt.fit(X_poly_opt, y)

y_pred_opt = lin_reg_opt.predict(X_poly_opt)
X_smooth = np.arange(min(X) - 0.05, max(X) + 0.05, 0.005).reshape(-1, 1)
X_smooth_poly = poly_reg_opt.transform(X_smooth)
y_smooth_pred = lin_reg_opt.predict(X_smooth_poly)

plt.figure(figsize=(12, 8))
plt.scatter(X, y, color='red', s=15, alpha=0.4, label='Исходные данные', zorder=5)
plt.plot(X_smooth, y_smooth_pred, color='blue', linewidth=3,
         label=f'Полиномиальная регрессия (степень {optimal_degree})')
plt.xlabel('Длина моллюска (Length)', fontsize=14)
plt.ylabel('Количество колец (Rings)', fontsize=14)
plt.title('Оптимальная модель полиномиальной регрессии для Abalone', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('lab8_optimal_model.png', dpi=150)
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

degrees_plot = [r['degree'] for r in results]
mse_values = [r['mse'] for r in results]
r2_values = [r['r2'] for r in results]

axes[0].plot(degrees_plot, mse_values, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Степень полинома', fontsize=12)
axes[0].set_ylabel('MSE', fontsize=12)
axes[0].set_title('Зависимость MSE от степени полинома (Abalone)', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=optimal_degree, color='r', linestyle='--', label=f'Оптимум (d={optimal_degree})')
axes[0].legend()

axes[1].plot(degrees_plot, r2_values, 'ro-', linewidth=2, markersize=8)
axes[1].set_xlabel('Степень полинома', fontsize=12)
axes[1].set_ylabel('R²', fontsize=12)
axes[1].set_title('Зависимость R² от степени полинома (Abalone)', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=optimal_degree, color='r', linestyle='--', label=f'Оптимум (d={optimal_degree})')
axes[1].legend()

plt.tight_layout()
plt.savefig('lab8_quality_by_degree.png', dpi=150)
plt.show()

new_abalone = np.array([[0.55]])
new_abalone_poly = poly_reg_opt.transform(new_abalone)
predicted_rings = lin_reg_opt.predict(new_abalone_poly)

print(f"\n=== Предсказание для нового моллюска ===")
print(f"Длина моллюска: 0.55")
print(f"Предсказанное количество колец (степень {optimal_degree}): {predicted_rings[0]:.2f}")
print(f"Округленный возраст: {int(round(predicted_rings[0]))} лет")

print("\n=== Сравнение предсказаний разных моделей для Length=0.55 ===")
print("-" * 50)
for d in [1, 2, 3, 4, 5, 6, 7, 8]:
    poly_reg = PolynomialFeatures(degree=d)
    new_poly = poly_reg.fit_transform(new_abalone)
    model, _, _, _, _ = polynomial_regression(d, X, y)
    pred = model.predict(new_poly)
    print(f"Степень {d}: {pred[0]:.2f} колец")


def create_polynomial_pipeline(degree, include_bias=False):
    return Pipeline([
        ('poly_features', PolynomialFeatures(degree=degree, include_bias=include_bias)),
        ('linear_regression', LinearRegression())
    ])


print("\nСравнение пайплайнов различных степеней ")

for d in [1, 2, 3, 4, 5]:
    pipeline = create_polynomial_pipeline(d)
    pipeline.fit(X, y)
    y_pred = pipeline.predict(X)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print(f"Пайплайн (степень {d}): MSE = {mse:.4f}, R² = {r2:.6f}")

best_pipeline = create_polynomial_pipeline(optimal_degree)
best_pipeline.fit(X, y)

print(f"\nОптимальный пайплайн (степень {optimal_degree})")
print(f"Коэффициенты: {best_pipeline.named_steps['linear_regression'].coef_}")
print(f"Свободный член: {best_pipeline.named_steps['linear_regression'].intercept_:.4f}")

print("Итоговая таблица результатов")

print(f"{'Степень':^10} | {'MSE (обучение)':^20} | {'R² (обучение)':^15} | {'Вердикт':^20}")

verdicts = {
    1: "Недообучение",
    2: "Слабая аппроксимация",
    3: "Хорошо (компромисс)",
    4: "Улучшение, риск переобучения",
    5: "Улучшение, риск переобучения",
    6: "Начало переобучения",
    7: "Переобучение",
    8: "ПЕРЕОБУЧЕНИЕ!"
}

for r in results:
    d = r['degree']
    print(f"{d:^10} | {r['mse']:20.4f} | {r['r2']:15.6f} | {verdicts.get(d, 'Проверить'):^20}")