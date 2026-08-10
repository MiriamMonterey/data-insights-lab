import pandas as pd

df = pd.read_csv("C:/Users/user/data-insights-lab/train.csv")

print("=== РАЗМЕР ДАТАСЕТА ===")
print("Строк (записей):", df.shape[0])
print("Колонок:", df.shape[1])
print("Память (байт):", df.memory_usage(deep=True).sum())
print()

print("=== ПРОПУСКИ ПО КОЛОНКАМ ===")
missing = df.isna().sum()
missing_pct = (df.isna().mean() * 100).round(2)
for col in df.columns:
    print(f"{col:12} -> {missing[col]:4d} пропусков  ({missing_pct[col]:5.2f}%)")
print()

print("=== ПАТТЕРН ПРОПУСКОВ ===")
# Как пропуски связаны с выживаемостью
print("-- Доля выживших (Survived) при наличии/отсутствии Age --")
df["Age_missing"] = df["Age"].isna()
print(df.groupby("Age_missing")["Survived"].agg(["mean", "count"]))
print()

print("-- Пропуски Cabin по классу (Pclass) --")
df["Cabin_missing"] = df["Cabin"].isna()
print(df.groupby("Pclass")["Cabin_missing"].agg(["sum", "count", "mean"]))
print()

print("-- Пропуски Embarked по классу --")
df["Embarked_missing"] = df["Embarked"].isna()
print(df.groupby("Pclass")["Embarked_missing"].agg(["sum", "count", "mean"]))
print()

print("-- Корреляция между пропусками колонок --")
print(df[["Age_missing", "Cabin_missing", "Embarked_missing"]].corr())
