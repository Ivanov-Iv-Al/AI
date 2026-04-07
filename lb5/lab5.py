import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

df = pd.read_csv('abalone.csv')
print(f"\nРазмер набора данных: {df.shape}")
print(f"Количество записей: {df.shape[0]}")
print(f"Количество признаков: {df.shape[1]}")

print("\nПервые 5 строк данных:")
print(df.head())

print("\nИнформация о данных:")
print(df.info())

print("\nСтатистическое описание числовых признаков:")
print(df.describe())

print("\nПроверка наличия пропусков:")
print(df.isnull().sum())

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sex_counts = df['Sex'].value_counts()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
plt.bar(sex_counts.index, sex_counts.values, color=colors, edgecolor='black')
plt.title('Распределение категориального признака Sex')
plt.xlabel('Пол')
plt.ylabel('Количество')
for i, v in enumerate(sex_counts.values):
    plt.text(i, v + 10, str(v), ha='center')

plt.subplot(1, 2, 2)
plt.hist(df['Rings'], bins=20, color='#95E77E', edgecolor='black', alpha=0.7)
plt.title('Распределение целевой переменной Rings')
plt.xlabel('Количество колец')
plt.ylabel('Частота')
plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Корреляционная матрица числовых признаков')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.ravel()
numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight',
                    'Shucked weight', 'Viscera weight', 'Shell weight']

for idx, feature in enumerate(numeric_features):
    sns.boxplot(x='Sex', y=feature, data=df, ax=axes[idx], palette='Set2')
    axes[idx].set_title(f'Распределение {feature} по полу')

axes[-1].set_visible(False)
plt.tight_layout()
plt.savefig('boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

X = df.drop('Rings', axis=1)
y = df['Rings']

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Числовые признаки: {numeric_features}")
print(f"Категориальные признаки: {categorical_features}")

print("\nКатегориальные признаки до One-Hot Encoding:")
for cat_feat in categorical_features:
    print(f"  {cat_feat}: {X[cat_feat].unique()}")

X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)

print("\nРезультат One-Hot Encoding:")
print(f"Размер после кодирования: {X_encoded.shape}")
print(f"Новые признаки: {[col for col in X_encoded.columns if 'Sex' in col]}")

print("\nСтатистика признаков ДО масштабирования:")
for col in numeric_features[:3]:
    print(f"  {col}: min={X_encoded[col].min():.4f}, max={X_encoded[col].max():.4f}, "
          f"mean={X_encoded[col].mean():.4f}, std={X_encoded[col].std():.4f}")

scaler_standard = StandardScaler()
X_standard_scaled = scaler_standard.fit_transform(X_encoded)
X_standard_df = pd.DataFrame(X_standard_scaled, columns=X_encoded.columns)

scaler_minmax = MinMaxScaler()
X_minmax_scaled = scaler_minmax.fit_transform(X_encoded)
X_minmax_df = pd.DataFrame(X_minmax_scaled, columns=X_encoded.columns)

print("\nСтатистика признаков после StandardScaler:")
for col in numeric_features[:3]:
    print(f"  {col}: min={X_standard_df[col].min():.4f}, max={X_standard_df[col].max():.4f}, "
          f"mean={X_standard_df[col].mean():.4f}, std={X_standard_df[col].std():.4f}")

print("\nСтатистика признаков после MinMaxScaler:")
for col in numeric_features[:3]:
    print(f"  {col}: min={X_minmax_df[col].min():.4f}, max={X_minmax_df[col].max():.4f}, "
          f"mean={X_minmax_df[col].mean():.4f}, std={X_minmax_df[col].std():.4f}")

X = df.drop('Rings', axis=1)
y = df['Rings']

numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Числовые признаки: {numeric_features}")
print(f"Категориальные признаки: {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

print("\nСтруктура пайплайна:")
print(pipeline)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Размер обучающей выборки: {X_train.shape}")
print(f"Размер тестовой выборки: {X_test.shape}")
print(f"Соотношение тест:обучение = 20:80")

print("\nРаспределение целевой переменной в обучающей выборке:")
print(y_train.value_counts().sort_index().head(10))
print("\nРаспределение целевой переменной в тестовой выборке:")
print(y_test.value_counts().sort_index().head(10))

pipeline.fit(X_train, y_train)

y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\nРезультаты обучения модели линейной регрессии:")
print(f"  MSE на обучающей выборке: {train_mse:.4f}")
print(f"  MSE на тестовой выборке: {test_mse:.4f}")
print(f"  R² на обучающей выборке: {train_r2:.4f}")
print(f"  R² на тестовой выборке: {test_r2:.4f}")

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5, color='blue', edgecolors='k')
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],
         'r--', lw=2, label='Идеальное предсказание')
plt.xlabel('Реальные значения (Rings)')
plt.ylabel('Предсказанные значения (Rings)')
plt.title(f'Обучающая выборка\nR² = {train_r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_test_pred, alpha=0.5, color='green', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label='Идеальное предсказание')
plt.xlabel('Реальные значения (Rings)')
plt.ylabel('Предсказанные значения (Rings)')
plt.title(f'Тестовая выборка\nR² = {test_r2:.4f}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('predictions.png', dpi=150, bbox_inches='tight')
plt.show()

errors_train = y_train - y_train_pred
errors_test = y_test - y_test_pred

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(errors_train, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.xlabel('Ошибка предсказания')
plt.ylabel('Частота')
plt.title(f'Обучающая выборка\nСр. ошибка = {errors_train.mean():.4f}')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(errors_test, bins=30, color='green', alpha=0.7, edgecolor='black')
plt.xlabel('Ошибка предсказания')
plt.ylabel('Частота')
plt.title(f'Тестовая выборка\nСр. ошибка = {errors_test.mean():.4f}')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('errors.png', dpi=150, bbox_inches='tight')
plt.show()

new_abalone = pd.DataFrame({
    'Sex': ['M'],
    'Length': [0.55],
    'Diameter': [0.44],
    'Height': [0.15],
    'Whole weight': [0.85],
    'Shucked weight': [0.35],
    'Viscera weight': [0.18],
    'Shell weight': [0.28]
})

print("Новый объект для предсказания:")
print(new_abalone)

prediction = pipeline.predict(new_abalone)
print(f"\nПредсказанное количество колец: {prediction[0]:.2f}")
print(f"Округленное значение: {int(round(prediction[0]))}")


