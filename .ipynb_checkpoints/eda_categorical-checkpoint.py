import pandas as pd

df = pd.read_csv("C:/Users/user/data-insights-lab/train.csv")

cat_cols = ["Sex", "Embarked", "Pclass"]
print("=== КАТЕГОРИАЛЬНЫЕ КОЛОНКИ (низкой кардинальности) ===")
for col in cat_cols:
    print(f"\n-- {col} -- (уникальных: {df[col].nunique(dropna=False)})")
    vc = df[col].value_counts(dropna=False)
    for val, cnt in vc.items():
        surv = df[df[col] == val]["Survived"].mean() if pd.notna(val) else float('nan')
        print(f"   {str(val):8} : {cnt:4d}  ({cnt/len(df)*100:5.1f}%)  выживаемость={surv:.3f}" if pd.notna(val) else f"   NaN      : {cnt:4d}")

print("\n\n=== ВЫСОКОЙ КАРДИНАЛЬНОСТИ (сырые признаки) ===")
for col in ["Name", "Ticket", "Cabin"]:
    print(f"{col}: уникальных {df[col].nunique(dropna=False)} из {len(df)}  ->  нужна инженерия признаков")

print("\n-- Cabin: первая буква (палуба) как признак --")
df["Deck"] = df["Cabin"].astype(str).str[0].replace("nan", "Unknown")
print(df["Deck"].value_counts(dropna=False).to_string())

print("\n-- Name: титул (Mr/Mrs/Miss/Master/др.) --")
df["Title"] = df["Name"].str.extract(r",\s*([^.]+)\.")[0].str.strip()
print(df["Title"].value_counts().to_string())
