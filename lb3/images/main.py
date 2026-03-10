import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

df = pd.read_csv('abalone.csv')

def age_category(rings):
    if rings <= 9:
        return 0
    elif rings < 16:
        return 1
    else:
        return 2

df['AgeCategory'] = df['Rings'].apply(age_category)
df = pd.get_dummies(df, columns=['Sex'], prefix=['Sex'], drop_first=False)

feature_columns = [
    'Length', 'Diameter', 'Height',
    'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight',
    'Sex_F', 'Sex_I', 'Sex_M'
]

X = df[feature_columns].values
y = df['AgeCategory'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

def predict_with_k(k_value, new_data):
    model = KNeighborsClassifier(n_neighbors=k_value)
    model.fit(X_scaled, y)
    new_data_scaled = scaler.transform(new_data.reshape(1, -1))
    prediction = model.predict(new_data_scaled)[0]
    return prediction

k_values = range(1, 51)
test_sizes = np.arange(0.1, 0.51, 0.05)
holdout_scores = np.zeros((len(k_values), len(test_sizes)))

for i, k in enumerate(k_values):
    for j, test_size in enumerate(test_sizes):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        holdout_scores[i, j] = model.score(X_test, y_test)

plt.figure(figsize=(14, 8))
sns.heatmap(
    holdout_scores.T,
    xticklabels=k_values[::1],
    yticklabels=[f"{ts:.2f}" for ts in test_sizes],
    annot=True, fmt='.3f', cmap='viridis',
    cbar_kws={'label': 'Accuracy'},
    linewidths=0.5, linecolor='gray'
)
plt.xlabel('K')
plt.ylabel('test_size')
plt.title('Hold-out Accuracy')
plt.tight_layout()
plt.show()
plt.close()

k_values = range(1, 51)
cv_folds = range(3, 11)
cv_scores = np.zeros((len(k_values), len(cv_folds)))

for i, k in enumerate(k_values):
    for j, cv in enumerate(cv_folds):
        model = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy')
        cv_scores[i, j] = np.mean(scores)

plt.figure(figsize=(14, 8))
sns.heatmap(
    cv_scores.T,
    xticklabels=k_values[::1],
    yticklabels=list(cv_folds),
    annot=True, fmt='.3f', cmap='plasma',
    cbar_kws={'label': 'Avg Accuracy'},
    linewidths=0.5, linecolor='gray'
)
plt.xlabel('K')
plt.ylabel('folds')
plt.title('Cross-Validation Accuracy')
plt.tight_layout()
plt.show()
plt.close()

optimal_k = 26
final_model = KNeighborsClassifier(n_neighbors=optimal_k)
final_model.fit(X_scaled, y)

y_pred_all = final_model.predict(X_scaled)

cm = confusion_matrix(y, y_pred_all)

print("Точность для разных k")
print(f"{'K':<5} {'Hold-out':<10} {'CV':<10} {'Среднее':<10}")

holdout_avg_by_k = np.mean(holdout_scores, axis=1)

cv_avg_by_k = np.mean(cv_scores, axis=1)

for k in range(1,51):
    siv = []
    holdout_val = holdout_avg_by_k[k-1]
    cv_val = cv_avg_by_k[k-1]
    avg_val = (holdout_val + cv_val) / 2
    print(f"{k} {holdout_val:.4f} {cv_val:.4f} {avg_val:.4f}")

kount = 0
for i in range (0,len(y)):
    if y[i] != y_pred_all[i]:
        print(f"{i}||||||Преположительно {y[i]}, фатически {y_pred_all[i]}")
        kount += 1
print(f"Кол-во неугаданных:{kount}")

plt.figure(figsize=(8, 6))
age_labels = ['Young (<=9)', 'Adult (10-15)', 'Old (>=16)']
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=age_labels, yticklabels=age_labels,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title(f'Confusion Matrix (K={optimal_k})')
plt.tight_layout()
plt.show()
plt.close()

