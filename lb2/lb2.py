import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

data = np.genfromtxt('abalone.csv',
                     delimiter=',',
                     dtype=None,
                     names=True,
                     encoding='utf-8')

sex = np.array([row[0] for row in data])
length = np.array([row[1] for row in data], dtype=float)
diameter = np.array([row[2] for row in data], dtype=float)
height = np.array([row[3] for row in data], dtype=float)
whole_weight = np.array([row[4] for row in data], dtype=float)
shucked_weight = np.array([row[5] for row in data], dtype=float)
viscera_weight = np.array([row[6] for row in data], dtype=float)
shell_weight = np.array([row[7] for row in data], dtype=float)
rings = np.array([row[8] for row in data], dtype=int)

age_category = np.zeros_like(rings)
age_category[rings <= 9] = 0
age_category[(rings > 9) & (rings < 16)] = 1
age_category[rings >= 16] = 2

age_labels = ['Молодые (<=9)', 'Взрослые (10-15)', 'Старые (>=16)']
age_colors = ['lightgreen', 'lightblue', 'lightcoral']

df = pd.DataFrame({
    'Sex': sex,
    'Length': length,
    'Diameter': diameter,
    'Height': height,
    'Whole weight': whole_weight,
    'Shucked weight': shucked_weight,
    'Viscera weight': viscera_weight,
    'Shell weight': shell_weight,
    'Rings': rings,
    'Age category': [age_labels[i] for i in age_category]
})

# 1. Гистограммы распределения количественных признаков
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

features_num = [length, diameter, height, whole_weight,
                shucked_weight, viscera_weight, shell_weight, rings]
feature_names = ['Length', 'Diameter', 'Height', 'Whole weight',
                 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']

for i, (feature, name) in enumerate(zip(features_num, feature_names)):
    axes[i].hist(feature, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    axes[i].set_title(f'Распределение признака {name}')
    axes[i].set_xlabel(name)
    axes[i].set_ylabel('Частота')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lb2_histograms.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. Ящики с усами
fig, axes = plt.subplots(2, 4, figsize=(18, 10))
axes = axes.ravel()

for i, (feature, name) in enumerate(zip(features_num, feature_names)):
    axes[i].boxplot(feature, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(f'Ящик с усами: {name}')
    axes[i].set_ylabel(name)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lb2_boxplots.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. Распределение по полу
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

unique_sex, sex_counts = np.unique(sex, return_counts=True)
sex_labels_full = ['Самец (M)', 'Самка (F)', 'Молодой (I)']

bars = axes[0].bar(sex_labels_full, sex_counts,
                   color=['blue', 'red', 'green'],
                   edgecolor='black', alpha=0.7)
axes[0].set_title('Распределение по полу')
axes[0].set_ylabel('Количество особей')
axes[0].grid(True, alpha=0.3, axis='y')

for bar, count in zip(bars, sex_counts):
    axes[0].text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                 f'{count}\n({count / len(data) * 100:.1f}%)',
                 ha='center', va='bottom')

axes[1].pie(sex_counts, labels=sex_labels_full, autopct='%1.1f%%',
            colors=['blue', 'red', 'green'], startangle=90,
            explode=(0.05, 0.05, 0.05))
axes[1].set_title('Распределение по полу (круговая диаграмма)')

plt.tight_layout()
plt.savefig('lb2_sex_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. Диаграммы рассеяния
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

scatter_configs = [
    (length, whole_weight, 'Длина', 'Общий вес'),
    (length, diameter, 'Длина', 'Диаметр'),
    (height, shell_weight, 'Высота', 'Вес раковины'),
    (whole_weight, shucked_weight, 'Общий вес', 'Вес мяса'),
    (rings, shell_weight, 'Количество колец', 'Вес раковины'),
    (diameter, height, 'Диаметр', 'Высота')
]

for i, (x, y, xlabel, ylabel) in enumerate(scatter_configs):
    for j in range(3):
        mask = age_category == j
        axes[i].scatter(x[mask], y[mask],
                        label=age_labels[j],
                        alpha=0.6, s=15,
                        color=age_colors[j])
    axes[i].set_title(f'{ylabel} vs {xlabel}')
    axes[i].set_xlabel(xlabel)
    axes[i].set_ylabel(ylabel)
    axes[i].legend(fontsize=9)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lb2_scatterplots.png', dpi=300, bbox_inches='tight')
plt.show()

plt.figure(figsize=(20, 16))
g = sns.pairplot(data=df, vars=['Length', 'Diameter', 'Height',
                                'Whole weight', 'Shell weight'],
                 hue='Age category', palette=age_colors,
                 diag_kind='hist', diag_kws={'alpha': 0.6, 'bins': 30},
                 plot_kws={'alpha': 0.5, 's': 15})
g.fig.suptitle('Попарное распределение признаков по возрастным категориям',
               y=1.02, fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('lb2_pairplot.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. Анализ по полу
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

features_by_sex = [length, diameter, height, whole_weight, shell_weight, rings]
feature_names_sex = ['Length', 'Diameter', 'Height', 'Whole weight', 'Shell weight', 'Rings']
sex_list = ['M', 'F', 'I']
sex_labels_plot = ['Самец (M)', 'Самка (F)', 'Молодой (I)']

for i, (feature, name) in enumerate(zip(features_by_sex, feature_names_sex)):
    data_by_sex = [feature[sex == s] for s in sex_list]
    axes[i].boxplot(data_by_sex, labels=sex_labels_plot,
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    medianprops=dict(color='red', linewidth=2))
    axes[i].set_title(f'Распределение {name} по полу')
    axes[i].set_ylabel(name)
    axes[i].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lb2_sex_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. Матрица корреляции
numeric_features = ['Length', 'Diameter', 'Height', 'Whole weight',
                    'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
correlation_matrix = df[numeric_features].corr()

plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix,
            annot=True, fmt='.3f', cmap='RdBu_r',
            center=0, square=True, linewidths=1,
            cbar_kws={"shrink": 0.8}, mask=mask)
plt.title('Матрица корреляции признаков', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('lb2_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. Возрастной анализ (один комбинированный график)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].hist(rings, bins=25, edgecolor='black', alpha=0.7,
             color='purple', density=True)
axes[0].set_title('Распределение количества колец')
axes[0].set_xlabel('Количество колец')
axes[0].set_ylabel('Плотность')
axes[0].axvline(rings.mean(), color='red', linestyle='--',
                label=f'Среднее: {rings.mean():.2f}')
axes[0].axvline(np.median(rings), color='green', linestyle='--',
                label=f'Медиана: {np.median(rings):.2f}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

sorted_rings = np.sort(rings)
y_vals = np.arange(1, len(sorted_rings) + 1) / len(sorted_rings)
axes[1].plot(sorted_rings, y_vals, linewidth=2, color='darkblue')
axes[1].set_title('Накопленная функция распределения')
axes[1].set_xlabel('Количество колец')
axes[1].set_ylabel('Накопленная вероятность')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.25, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
axes[1].axhline(y=0.75, color='blue', linestyle='--', alpha=0.5)

bars = axes[2].bar(age_labels, [np.sum(age_category == i) for i in range(3)],
                   color=age_colors, edgecolor='black')
axes[2].set_title('Распределение возрастных категорий')
axes[2].set_ylabel('Количество особей')
for bar in bars:
    height_val = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width() / 2., height_val,
                 f'{int(height_val)}\n({height_val / len(data) * 100:.1f}%)',
                 ha='center', va='bottom')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('lb2_age_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Размер данных: {len(data)} записей, {len(numeric_features)} признаков")
print(f"Пропущенные значения: отсутствуют")
print(f"\nРаспределение по полу:")
for s, label, count in zip(['M', 'F', 'I'], sex_labels_full, sex_counts):
    print(f"  {label}: {count} особей ({count/len(data)*100:.1f}%)")
print(f"\nРаспределение возрастных категорий:")
for i, label in enumerate(age_labels):
    count = np.sum(age_category == i)
    print(f"  {label}: {count} особей ({count/len(data)*100:.1f}%)")
print(f"\nКорреляция с возрастом:")
for feature in numeric_features[:-1]:
    corr = correlation_matrix.loc[feature, 'Rings']
    print(f"  {feature}: {corr:.3f}")