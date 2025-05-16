import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("wine-quality-white-and-red.csv")

# Проверка на наличие пропущенных значений
print("Количество пропущенных значений по столбцам:")
print(df.isna().sum()) #подсчитывает количество пропусков NaN в каждом столбце

# Вывод описательной статистики по каждому признаку
print("\nСтатистическое описание признаков:")
print(df.describe())

# Преобразование категориального признака 'type' в числовой (0 — красное, 1 — белое)
df["type"] = df["type"].map({"red": 0, "white": 1})

# Группировка по типу вина и расчёт различных статистических характеристик
statistical_analysis = df.groupby("type").agg(
    [
        "mean",              # среднее значение
        "median",            # медиана
        "std",               # стандартное отклонение
        "var",               # дисперсия
        lambda x: x.max() - x.min(),  # размах
    ]
)
print("\nСтатистика по каждому типу вина:")
print(statistical_analysis)

# Анализ корреляции между признаками и типом вина
correlation = df.corr()["type"].abs().sort_values(ascending=False)
print("\nКорреляция признаков с типом вина:")
print(correlation)

# Визуализация: гистограмма летучей кислотности по типу вина
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="volatile acidity", hue="type", palette="Set2")
plt.title("Распределение летучей кислотности по типу вина")
plt.xlabel("Летучая кислотность")
plt.ylabel("Количество образцов")
plt.legend(title="Тип вина", labels=["Красное", "Белое"])
plt.tight_layout()
plt.show()

# Визуализация: boxplot содержания алкоголя по типу вина
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="type", y="alcohol", palette="Set2")
plt.title("Сравнение содержания алкоголя в красном и белом вине")
plt.xlabel("Тип вина")
plt.ylabel("Алкоголь (%)")
plt.xticks([0, 1], ["Красное", "Белое"])
plt.tight_layout()
plt.show()

# Тепловая карта корреляций
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Корреляционная матрица признаков")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# Выбор трёх признаков с наибольшей корреляцией с типом вина
top_features = ["total sulfur dioxide", "volatile acidity", "chlorides"]
X = df[top_features]
y = df["type"]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Обучение модели логистической регрессии
model = LogisticRegression()
model.fit(X_train, y_train)

# Предсказание и оценка модели
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=["Красное", "Белое"])

print(f"\nТочность модели: {accuracy:.2%}")
print("\nОтчёт по классификации:")
print(report)

# Визуализация распределения трёх выбранных признаков
sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

russian_labels = {
    "total sulfur dioxide": "Общее содержание диоксида серы",
    "volatile acidity": "Летучая кислотность",
    "chlorides": "Хлориды"
}

for i, feature in enumerate(top_features):
    sns.histplot(
        data=df,
        x=feature,
        hue="type",
        kde=True,
        ax=axes[i],
        palette="Set1",
        element="step",
        common_norm=False,
    )
    axes[i].set_title(f"Распределение признака: {russian_labels[feature]}")
    axes[i].set_xlabel(russian_labels[feature])
    axes[i].set_ylabel("Количество образцов")
    axes[i].legend(title="Тип вина", labels=["Красное", "Белое"])

plt.tight_layout()
plt.show()
