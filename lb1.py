import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

print("=== ЗАГРУЗКА ДАННЫХ ===\n")

data = np.genfromtxt('abalone.csv',
                     delimiter=',',
                     dtype=None,
                     names=True,
                     encoding='utf-8')

print(f"Тип загруженных данных: {type(data)}")
print(f"Размерность данных: {data.shape}")
print(f"Количество записей: {len(data)}")
print(f"Имена полей: {data.dtype.names}")

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

print("\n=== СОЗДАНИЕ КЛАССОВ ДЛЯ КЛАССИФИКАЦИИ ===\n")
for i, label in enumerate(age_labels):
    count = np.sum(age_category == i)
    print(f"{label}: {count} особей ({count / len(data) * 100:.1f}%)")

print("\n=== ОСНОВНАЯ ИНФОРМАЦИЯ ===\n")
print(f"Всего записей: {len(data)}")
print(f"Количество признаков: {len(data.dtype.names)}")

print("\n=== ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ ===\n")
for name in data.dtype.names:
    col_data = data[name]
    if col_data.dtype.kind in ['f', 'i']:
        missing = np.sum(np.isnan(col_data))
    else:
        missing = np.sum(col_data == None)
    print(f"{name}: {missing} пропусков")

print("\n=== СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ ===\n")
features = ['length', 'diameter', 'height', 'whole_weight',
            'shucked_weight', 'viscera_weight', 'shell_weight', 'rings']

for feature in features:
    data_array = eval(feature)
    print(f"\n{feature.upper()}:")
    print(f"  Среднее: {np.mean(data_array):.3f}")
    print(f"  Медиана: {np.median(data_array):.3f}")
    print(f"  Мин: {np.min(data_array):.3f}")
    print(f"  Макс: {np.max(data_array):.3f}")
    print(f"  Стд. отклонение: {np.std(data_array):.3f}")

print("\n=== РАСПРЕДЕЛЕНИЕ ПО ПОЛУ ===\n")
unique_sex, sex_counts = np.unique(sex, return_counts=True)
for s, count in zip(unique_sex, sex_counts):
    print(f"Пол {s}: {count} особей ({count / len(data) * 100:.1f}%)")

fig = plt.figure(figsize=(20, 12))

plt.subplot(2, 3, 1)
colors = ['lightgreen', 'lightblue', 'lightcoral']
bars = plt.bar(age_labels, [np.sum(age_category == i) for i in range(3)],
               color=colors, edgecolor='black')
plt.title('Распределение возрастных классов', fontsize=14, fontweight='bold')
plt.ylabel('Количество особей')
for bar in bars:
    height_val = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height_val,
             f'{int(height_val)}', ha='center', va='bottom')

plt.subplot(2, 3, 2)
plt.hist(rings, bins=30, edgecolor='black', alpha=0.7, color='purple')
plt.title('Распределение количества колец', fontsize=14, fontweight='bold')
plt.xlabel('Количество колец')
plt.ylabel('Частота')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 3)
for i in range(3):
    mask = age_category == i
    plt.scatter(length[mask], whole_weight[mask],
                label=age_labels[i], alpha=0.6, s=20)
plt.title('Зависимость веса от длины', fontsize=14, fontweight='bold')
plt.xlabel('Длина')
plt.ylabel('Общий вес')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 4)
sex_list = ['M', 'F', 'I']
data_by_sex = [rings[sex == s] for s in sex_list]
plt.boxplot(data_by_sex, tick_labels=['Самец (M)', 'Самка (F)', 'Молодой (I)'],
            patch_artist=True, boxprops=dict(facecolor='lightblue'))
plt.title('Распределение возраста по полу', fontsize=14, fontweight='bold')
plt.ylabel('Количество колец')
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 3, 5)
for i in range(3):
    mask = age_category == i
    plt.scatter(length[mask], diameter[mask],
                label=age_labels[i], alpha=0.6, s=20)
plt.title('Длина vs Диаметр', fontsize=14, fontweight='bold')
plt.xlabel('Длина')
plt.ylabel('Диаметр')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
for s, color, label in zip(['M', 'F', 'I'], ['blue', 'red', 'green'],
                           ['Самец', 'Самка', 'Молодой']):
    mask = sex == s
    plt.scatter(rings[mask], shell_weight[mask],
                c=color, label=label, alpha=0.5, s=15)
plt.title('Вес раковины vs Возраст', fontsize=14, fontweight='bold')
plt.xlabel('Количество колец')
plt.ylabel('Вес раковины')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n=== КОРРЕЛЯЦИЯ С ВОЗРАСТОМ (КОЛЬЦАМИ) ===\n")

numeric_data = np.column_stack([length, diameter, height, whole_weight,
                                shucked_weight, viscera_weight, shell_weight])

feature_names = ['Length', 'Diameter', 'Height', 'Whole weight',
                 'Shucked weight', 'Viscera weight', 'Shell weight']

correlations = []
for i in range(numeric_data.shape[1]):
    corr = np.corrcoef(numeric_data[:, i], rings)[0, 1]
    correlations.append(corr)
    print(f"{feature_names[i]}: {corr:.3f}")

sorted_idx = np.argsort(correlations)[::-1]
print("\n=== ПРИЗНАКИ, ОТСОРТИРОВАННЫЕ ПО КОРРЕЛЯЦИИ ===\n")
for idx in sorted_idx:
    print(f"{feature_names[idx]}: {correlations[idx]:.3f}")

print("\n=== АНАЛИЗ РАЗДЕЛИМОСТИ КЛАССОВ ===\n")

for i, name in enumerate(feature_names):
    print(f"\n{name}:")
    class_means = []
    for c in range(3):
        mask = age_category == c
        mean_val = np.mean(numeric_data[mask, i])
        std_val = np.std(numeric_data[mask, i])
        class_means.append(mean_val)
        print(f"  {age_labels[c]}: среднее={mean_val:.3f}, стд={std_val:.3f}")

    separability = max(class_means) - min(class_means)
    print(f"  Размах между классами: {separability:.3f}")

print("\n=== ОСНОВНЫЕ ВЫВОДЫ ДЛЯ КЛАССИФИКАЦИИ ===\n")
print("1. Размер данных: {} записей, {} признаков".format(len(data), len(feature_names)))
print("2. Пропущенные значения: отсутствуют")
print("3. Целевая переменная (возраст) разбита на 3 класса:")
for i, label in enumerate(age_labels):
    print(f"   - {label}: {np.sum(age_category == i)} особей")
print("\n4. Наиболее информативные признаки для классификации:")
for i in sorted_idx[:3]:
    print(f"   - {feature_names[i]} (корреляция: {correlations[i]:.3f})")
print("\n5. Наблюдения:")
print("   - Самки в среднем старше самцов")
print("   - Молодые особи имеют меньший разброс параметров")
print("   - Весовые характеристики лучше всего коррелируют с возрастом")
print("   - Классы перекрываются, потребуется сложная модель классификации")
