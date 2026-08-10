import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/user/data-insights-lab/train.csv")
OUT = "C:/Users/user/data-insights-lab/"

# ---------- 1. Гистограммы числовых переменных ----------
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
df["Age"].dropna().hist(bins=30, ax=axes[0, 0], color="#4C72B0", edgecolor="white")
axes[0, 0].set_title("Age (возраст)")
df["Fare"].hist(bins=40, ax=axes[0, 1], color="#55A868", edgecolor="white")
axes[0, 1].set_title("Fare (цена билета)")
df["SibSp"].hist(bins=range(0, 10), ax=axes[1, 0], color="#C44E52", edgecolor="white")
axes[1, 0].set_title("SibSp (братья/сёстры/супруги)")
df["Parch"].hist(bins=range(0, 8), ax=axes[1, 1], color="#8172B3", edgecolor="white")
axes[1, 1].set_title("Parch (родители/дети)")
fig.suptitle("Гистограммы распределений числовых переменных", fontsize=14)
fig.tight_layout()
fig.savefig(OUT + "hist_numeric.png", dpi=110)
plt.close(fig)

# ---------- 2. Boxplot: Age и Fare по выживаемости ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df.boxplot(column="Age", by="Survived", ax=axes[0])
axes[0].set_title("Age по Survived")
axes[0].set_xlabel("Survived (0=погиб, 1=выжил)")
df.boxplot(column="Fare", by="Survived", ax=axes[1])
axes[1].set_title("Fare по Survived")
axes[1].set_xlabel("Survived (0=погиб, 1=выжил)")
axes[1].set_ylim(0, 300)  # обрезаем экстремальный выброс 512 для читаемости
fig.suptitle("Boxplot: Age и Fare в разрезе выживаемости", fontsize=14)
fig.tight_layout()
fig.savefig(OUT + "box_survival.png", dpi=110)
plt.close(fig)

# ---------- 3. Столбчатые: выживаемость по категориям ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col in zip(axes, ["Sex", "Pclass", "Embarked"]):
    df.groupby(col)["Survived"].mean().plot(kind="bar", ax=ax, color="#4C72B0", edgecolor="black")
    ax.set_title(f"Доля выживших по {col}")
    ax.set_ylabel("Доля выживших")
    ax.set_ylim(0, 1)
    ax.axhline(df["Survived"].mean(), color="red", ls="--", lw=1, label="средняя (0.38)")
    ax.legend()
fig.suptitle("Выживаемость по категориальным признакам", fontsize=14)
fig.tight_layout()
fig.savefig(OUT + "bar_survival_cat.png", dpi=110)
plt.close(fig)

print("Сохранены графики:")
print(" - hist_numeric.png       (гистограммы Age/Fare/SibSp/Parch)")
print(" - box_survival.png       (boxplot Age/Fare по Survived)")
print(" - bar_survival_cat.png   (выживаемость по Sex/Pclass/Embarked)")
