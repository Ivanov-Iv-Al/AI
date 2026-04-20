import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

# Загрузка данных
df = pd.read_csv('abalone.csv')
print(f"Размер набора данных: {df.shape}")
print("\nПервые 5 строк:")
print(df.head())

# Подготовка данных
X = df.drop('Rings', axis=1)
y = df['Rings']

# One-Hot Encoding для категориального признака Sex
X_encoded = pd.get_dummies(X, columns=['Sex'], drop_first=True)
print(f"\nРазмер данных после One-Hot Encoding: {X_encoded.shape}")
print(f"Типы данных после кодирования:\n{X_encoded.dtypes}")

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
print(f"\nРазмер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")

# ============================================
# Модель со всеми признаками (All-in)
# ============================================
print("\n" + "=" * 50)
print("Модель со всеми признаками (All-in)")
print("=" * 50)

regressor_all = LinearRegression()
regressor_all.fit(X_train, y_train)

y_pred_train_all = regressor_all.predict(X_train)
y_pred_test_all = regressor_all.predict(X_test)

train_mse_all = mean_squared_error(y_train, y_pred_train_all)
test_mse_all = mean_squared_error(y_test, y_pred_test_all)
train_r2_all = r2_score(y_train, y_pred_train_all)
test_r2_all = r2_score(y_test, y_pred_test_all)

print(f"MSE на обучающей выборке: {train_mse_all:.4f}")
print(f"MSE на тестовой выборке: {test_mse_all:.4f}")
print(f"R² на обучающей выборке: {train_r2_all:.4f}")
print(f"R² на тестовой выборке: {test_r2_all:.4f}")

# ============================================
# Backward Elimination с использованием p-значений
# ============================================
print("\n" + "=" * 50)
print("Backward Elimination (p-значения)")
print("=" * 50)

# Преобразуем данные в numpy array для statsmodels
X_train_array = X_train.values.astype(np.float64)
X_train_with_const = sm.add_constant(X_train_array)

model = sm.OLS(y_train.values.astype(np.float64), X_train_with_const).fit()

print("\nСтатистика модели (p-значения):")
p_values = model.pvalues
feature_names = ['const'] + list(X_train.columns)
for i, (name, p_val) in enumerate(zip(feature_names, p_values)):
    print(f"  {name}: p = {p_val:.6f}")

print(f"\nR² скорректированный: {model.rsquared_adj:.4f}")

# Определяем значимые признаки (p < 0.05)
significant_indices = [i for i, p_val in enumerate(p_values) if p_val < 0.05 and i > 0]
significant_vars = [X_train.columns[i - 1] for i in significant_indices]

print(f"\nЗначимые признаки (p < 0.05): {significant_vars}")

# ============================================
# Модель только со значимыми признаками
# ============================================
if significant_vars:
    X_train_sig = X_train[significant_vars]
    X_test_sig = X_test[significant_vars]

    regressor_sig = LinearRegression()
    regressor_sig.fit(X_train_sig, y_train)

    y_pred_train_sig = regressor_sig.predict(X_train_sig)
    y_pred_test_sig = regressor_sig.predict(X_test_sig)

    train_mse_sig = mean_squared_error(y_train, y_pred_train_sig)
    test_mse_sig = mean_squared_error(y_test, y_pred_test_sig)
    train_r2_sig = r2_score(y_train, y_pred_train_sig)
    test_r2_sig = r2_score(y_test, y_pred_test_sig)

    print("\n" + "=" * 50)
    print("Модель только со значимыми признаками")
    print("=" * 50)
    print(f"MSE на обучающей выборке: {train_mse_sig:.4f}")
    print(f"MSE на тестовой выборке: {test_mse_sig:.4f}")
    print(f"R² на обучающей выборке: {train_r2_sig:.4f}")
    print(f"R² на тестовой выборке: {test_r2_sig:.4f}")
else:
    print("\nВсе признаки значимы, модель не изменилась")
    regressor_sig = regressor_all
    X_test_sig = X_test
    y_pred_test_sig = y_pred_test_all
    test_r2_sig = test_r2_all

# ============================================
# Анализ важности признаков
# ============================================
print("\n" + "=" * 50)
print("Важность признаков (коэффициенты модели)")
print("=" * 50)

feature_importance = pd.DataFrame({
    'Признак': X_train.columns,
    'Коэффициент': regressor_all.coef_
})
feature_importance = feature_importance.reindex(
    feature_importance['Коэффициент'].abs().sort_values(ascending=False).index
)
for _, row in feature_importance.iterrows():
    print(f"  {row['Признак']}: {row['Коэффициент']:.4f}")

# ============================================
# Корреляционный анализ
# ============================================
print("\n" + "=" * 50)
print("Корреляционный анализ")
print("=" * 50)

numeric_cols = df.select_dtypes(include=[np.number]).columns
correlations = df[numeric_cols].corr()['Rings'].sort_values(ascending=False)
for feat, corr in correlations.items():
    print(f"  {feat}: {corr:.4f}")

# Построение корреляционной матрицы
plt.figure(figsize=(10, 8))
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, fmt='.2f')
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.savefig('lab7_correlation_matrix.png', dpi=150)
print("\nГрафик корреляционной матрицы сохранен как 'lab7_correlation_matrix.png'")

# ============================================
# Визуализация результатов
# ============================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(y_train, y_pred_train_all, alpha=0.5, c='blue', edgecolors='k')
axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Реальные значения (Rings)')
axes[0, 0].set_ylabel('Предсказанные значения (Rings)')
axes[0, 0].set_title(f'Все признаки (обучение)\nR² = {train_r2_all:.4f}')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].scatter(y_test, y_pred_test_all, alpha=0.5, c='green', edgecolors='k')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Реальные значения (Rings)')
axes[0, 1].set_ylabel('Предсказанные значения (Rings)')
axes[0, 1].set_title(f'Все признаки (тест)\nR² = {test_r2_all:.4f}')
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(y_train, y_pred_train_sig, alpha=0.5, c='blue', edgecolors='k')
axes[1, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
axes[1, 0].set_xlabel('Реальные значения (Rings)')
axes[1, 0].set_ylabel('Предсказанные значения (Rings)')
axes[1, 0].set_title(f'Значимые признаки (обучение)\nR² = {train_r2_sig:.4f}')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].scatter(y_test, y_pred_test_sig, alpha=0.5, c='green', edgecolors='k')
axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('Реальные значения (Rings)')
axes[1, 1].set_ylabel('Предсказанные значения (Rings)')
axes[1, 1].set_title(f'Значимые признаки (тест)\nR² = {test_r2_sig:.4f}')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab7_multivariate_results.png', dpi=150)
print("График результатов сохранен как 'lab7_multivariate_results.png'")

print("Предсказание для нового объекта")

new_abalone = pd.DataFrame({
    'Length': [0.55],
    'Diameter': [0.44],
    'Height': [0.15],
    'Whole weight': [0.85],
    'Shucked weight': [0.35],
    'Viscera weight': [0.18],
    'Shell weight': [0.28],
    'Sex_I': [0],
    'Sex_M': [1]
})

print("Новый объект (моллюск мужского пола):")
for col in new_abalone.columns:
    print(f"  {col}: {new_abalone[col].values[0]}")

prediction_all = regressor_all.predict(new_abalone)
print(f"\nПредсказание по модели All-in: {prediction_all[0]:.2f} колец")
print(f"Округленное значение: {int(round(prediction_all[0]))}")

if significant_vars:
    prediction_sig = regressor_sig.predict(new_abalone[significant_vars])
    print(f"Предсказание по модели с значимыми признаками: {prediction_sig[0]:.2f} колец")

print("Сравнение с одномерной регрессией")

X_simple = df[['Length']]
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

regressor_simple = LinearRegression()
regressor_simple.fit(X_train_simple, y_train_simple)
y_pred_test_simple = regressor_simple.predict(X_test_simple)
test_r2_simple = r2_score(y_test_simple, y_pred_test_simple)

print(f"Одномерная модель (Length): R² = {test_r2_simple:.4f}")
print(f"Многомерная модель (All-in): R² = {test_r2_all:.4f}")
print(f"Улучшение качества: {(test_r2_all - test_r2_simple) * 100:.2f}%")

plt.show()