import pandas as pd

df = pd.read_csv("C:/Users/user/data-insights-lab/train.csv")

num_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
print("=== БАЗОВАЯ СТАТИСТИКА ПО ЧИСЛОВЫМ КОЛОНКАМ ===")
print("(для Age учтены только заполненные значения)")
stats = df[num_cols].describe(percentiles=[.25, .5, .75, .9]).T
stats["median"] = df[num_cols].median()
stats["std"] = df[num_cols].std()
print(stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "90%", "max"]].round(3).to_string())
print()

print("=== ДОПОЛНИТЕЛЬНО ===")
print("Survived: 0=%d, 1=%d  (доля выживших=%.1f%%)" % (
    (df.Survived == 0).sum(), (df.Survived == 1).sum(), df.Survived.mean() * 100))
print("Pclass (распределение):", df["Pclass"].value_counts().sort_index().to_dict())
print("SibSp max:", df.SibSp.max(), "| Parch max:", df.Parch.max())
print("Fare: min=%.2f max=%.2f  (без 1%% верхних: %.2f)" % (
    df.Fare.min(), df.Fare.max(), df.Fare.quantile(0.99)))
