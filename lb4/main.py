import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

# Не переключаем бэкенд, чтобы рисунки отображались
# plt.switch_backend('Agg')

df = pd.read_csv('abalone.csv')

def age_category(rings):
    if rings <= 9:
        return 0
    elif rings < 16:
        return 1
    else:
        return 2

df['AgeCategory'] = df['Rings'].apply(age_category)
df = pd.get_dummies(df, columns=['Sex'], prefix=['Sex'])

feature_columns = ['Length', 'Diameter', 'Height', 'Whole weight',
                   'Shucked weight', 'Viscera weight', 'Shell weight',
                   'Sex_F', 'Sex_I', 'Sex_M']

X = df[feature_columns].values
y = df['AgeCategory'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №4. ЛОГИЧЕСКИЕ МЕТОДЫ КЛАССИФИКАЦИИ")
print("=" * 60)
print(f"Размер выборки: {X.shape[0]} объектов, {X.shape[1]} признаков")
print(f"Распределение классов: молодые (0-9): {np.sum(y==0)}, взрослые (10-15): {np.sum(y==1)}, старые (16+): {np.sum(y==2)}")
print()

user_max_depth = 5
user_max_features = 4

user_tree = DecisionTreeClassifier(max_depth=user_max_depth, max_features=user_max_features, random_state=42)
user_tree.fit(X_scaled, y)

print("1. ДЕРЕВО С ПАРАМЕТРАМИ ПОЛЬЗОВАТЕЛЯ")
print(f"   max_depth = {user_max_depth}, max_features = {user_max_features}")
print(f"   Точность на обучающих данных: {user_tree.score(X_scaled, y):.4f}")
print()

plt.figure(figsize=(20, 10))
plot_tree(user_tree, feature_names=feature_columns, class_names=['Young', 'Adult', 'Old'], filled=True, rounded=True)
plt.title(f'Decision Tree (max_depth={user_max_depth}, max_features={user_max_features})')
plt.savefig('user_tree.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Визуализация сохранена: user_tree.png и отображена на экране")
print()

depth_values = range(1, 21)
cv_scores_depth = []
cv_mse_depth = []

for depth in depth_values:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42)
    scores = cross_val_score(tree, X_scaled, y, cv=5, scoring='accuracy')
    mse_scores = cross_val_score(tree, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_depth.append(np.mean(scores))
    cv_mse_depth.append(-np.mean(mse_scores))

best_depth_acc = depth_values[np.argmax(cv_scores_depth)]
best_depth_mse = depth_values[np.argmin(cv_mse_depth)]

print("2. АНАЛИЗ ВЛИЯНИЯ max_depth")
print(f"   Оптимальная глубина по accuracy: {best_depth_acc} (точность: {max(cv_scores_depth):.4f})")
print(f"   Оптимальная глубина по MSE: {best_depth_mse} (MSE: {min(cv_mse_depth):.4f})")
print()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(depth_values, cv_mse_depth, 'b-o', linewidth=2, markersize=6)
plt.xlabel('max_depth')
plt.ylabel('MSE')
plt.title('Зависимость MSE от глубины дерева')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(depth_values, cv_scores_depth, 'r-o', linewidth=2, markersize=6)
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Зависимость точности от глубины дерева')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('depth_analysis.png', dpi=150)
plt.show()
print("   Графики сохранены: depth_analysis.png и отображены на экране")
print()

feature_values = range(1, len(feature_columns) + 1)
cv_scores_features = []
cv_mse_features = []

for n_features in feature_values:
    tree = DecisionTreeClassifier(max_features=n_features, random_state=42)
    scores = cross_val_score(tree, X_scaled, y, cv=5, scoring='accuracy')
    mse_scores = cross_val_score(tree, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores_features.append(np.mean(scores))
    cv_mse_features.append(-np.mean(mse_scores))

best_features_acc = feature_values[np.argmax(cv_scores_features)]
best_features_mse = feature_values[np.argmin(cv_mse_features)]

print("3. АНАЛИЗ ВЛИЯНИЯ max_features")
print(f"   Оптимальное количество признаков по accuracy: {best_features_acc} (точность: {max(cv_scores_features):.4f})")
print(f"   Оптимальное количество признаков по MSE: {best_features_mse} (MSE: {min(cv_mse_features):.4f})")
print()

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(feature_values, cv_mse_features, 'g-o', linewidth=2, markersize=6)
plt.xlabel('max_features')
plt.ylabel('MSE')
plt.title('Зависимость MSE от количества признаков')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(feature_values, cv_scores_features, 'm-o', linewidth=2, markersize=6)
plt.xlabel('max_features')
plt.ylabel('Accuracy')
plt.title('Зависимость точности от количества признаков')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('features_analysis.png', dpi=150)
plt.show()
print("   Графики сохранены: features_analysis.png и отображены на экране")
print()

param_grid = {
    'max_depth': range(3, 16),
    'max_features': range(2, 11),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("4. ПОИСК ОПТИМАЛЬНЫХ ПАРАМЕТРОВ (GridSearchCV)")
print("   Выполняется поиск по сетке параметров...")

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

print(f"   Лучшие параметры: {grid_search.best_params_}")
print(f"   Лучшая точность (cross-validation): {grid_search.best_score_:.4f}")
print()

best_tree = grid_search.best_estimator_

plt.figure(figsize=(25, 15))
plot_tree(best_tree, feature_names=feature_columns, class_names=['Young', 'Adult', 'Old'], filled=True, rounded=True)
plt.title(f'Optimal Decision Tree\nmax_depth={best_tree.max_depth}, max_features={best_tree.max_features}')
plt.savefig('optimal_tree.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Оптимальное дерево сохранено: optimal_tree.png и отображено на экране")
print()

feature_importance = pd.DataFrame({'feature': feature_columns, 'importance': best_tree.feature_importances_}).sort_values('importance', ascending=False)

print("5. ВАЖНОСТЬ ПРИЗНАКОВ")
print("   Топ-5 наиболее важных признаков:")
for i in range(5):
    print(f"   {i+1}. {feature_importance.iloc[i]['feature']}: {feature_importance.iloc[i]['importance']:.4f}")
print()

plt.figure(figsize=(10, 8))
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
plt.barh(feature_importance['feature'], feature_importance['importance'], color=colors)
plt.xlabel('Важность')
plt.ylabel('Признаки')
plt.title('Важность признаков в оптимальном дереве решений')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
print("   График важности признаков сохранен: feature_importance.png и отображен на экране")
print()

top_features = feature_importance['feature'].head(2).values
idx1 = feature_columns.index(top_features[0])
idx2 = feature_columns.index(top_features[1])

print("6. РЕШАЮЩИЕ ГРАНИЦЫ")
print(f"   Построение границ по признакам: {top_features[0]} и {top_features[1]}")

x_min, x_max = X_scaled[:, idx1].min() - 0.5, X_scaled[:, idx1].max() + 0.5
y_min, y_max = X_scaled[:, idx2].min() - 0.5, X_scaled[:, idx2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

other_features_mean = np.mean(X_scaled, axis=0)
grid_points = np.c_[xx.ravel(), yy.ravel()]

X_grid = np.zeros((grid_points.shape[0], X_scaled.shape[1]))
X_grid[:, idx1] = grid_points[:, 0]
X_grid[:, idx2] = grid_points[:, 1]
for i in range(X_scaled.shape[1]):
    if i not in [idx1, idx2]:
        X_grid[:, i] = other_features_mean[i]

Z = best_tree.predict(X_grid)
Z = Z.reshape(xx.shape)

plt.figure(figsize=(12, 10))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
scatter = plt.scatter(X_scaled[:, idx1], X_scaled[:, idx2], c=y, cmap='RdYlBu', edgecolors='k', linewidth=0.5, alpha=0.7)
plt.colorbar(scatter, ticks=[0, 1, 2], label='Класс')
plt.xlabel(top_features[0], fontsize=12)
plt.ylabel(top_features[1], fontsize=12)
plt.title(f'Решающие границы оптимального дерева решений\n(признаки: {top_features[0]} и {top_features[1]})', fontsize=14)
plt.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d73027', alpha=0.7, label='Молодые (0-9)'),
                   Patch(facecolor='#fee090', alpha=0.7, label='Взрослые (10-15)'),
                   Patch(facecolor='#4575b4', alpha=0.7, label='Старые (16+)')]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('decision_boundaries.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Решающие границы сохранены: decision_boundaries.png и отображены на экране")
print()

final_tree = DecisionTreeClassifier(**grid_search.best_params_, random_state=42)
final_tree.fit(X_scaled, y)

y_pred = final_tree.predict(X_scaled)
cm = confusion_matrix(y, y_pred)

print("7. МАТРИЦА ОШИБОК")
print(f"   Точность на всей выборке: {np.mean(y == y_pred):.4f}")
print("   Матрица ошибок:")
print(f"               Предсказано")
print(f"               Young  Adult  Old")
print(f"   Young       {cm[0,0]:5d}  {cm[0,1]:5d}  {cm[0,2]:5d}")
print(f"   Adult       {cm[1,0]:5d}  {cm[1,1]:5d}  {cm[1,2]:5d}")
print(f"   Old         {cm[2,0]:5d}  {cm[2,1]:5d}  {cm[2,2]:5d}")
print()

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Young', 'Adult', 'Old'],
            yticklabels=['Young', 'Adult', 'Old'])
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.title('Матрица ошибок')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("   Матрица ошибок сохранена: confusion_matrix.png и отображена на экране")
print()

print(f"Оптимальная глубина дерева: {best_tree.max_depth}")
print(f"Оптимальное количество признаков: {best_tree.max_features}")
print(f"Минимальное количество объектов для разделения: {best_tree.min_samples_split}")
print(f"Минимальное количество объектов в листе: {best_tree.min_samples_leaf}")
print(f"Точность модели (CV): {grid_search.best_score_:.4f}")
print(f"Точность модели (обучение): {final_tree.score(X_scaled, y):.4f}")
print(f"Глубина полученного дерева: {final_tree.get_depth()}")
print(f"Количество листьев: {final_tree.get_n_leaves()}")
print()