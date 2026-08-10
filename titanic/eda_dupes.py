import pandas as pd

df = pd.read_csv("C:/Users/user/data-insights-lab/train.csv")

print("=== ДУБЛИКАТЫ ===")
print("Полных дубликатов (все колонки):", df.duplicated().sum())
print()

print("-- Дубликаты по PassengerId (должен быть уникальным ключом) --")
print("Дубликатов PassengerId:", df["PassengerId"].duplicated().sum())
print()

print("-- Дубликаты по Name (имена могут повторяться у разных людей) --")
print("Дубликатов Name:", df["Name"].duplicated().sum())
print()

print("-- Дубликаты по Ticket --")
print("Уникальных Ticket:", df["Ticket"].nunique(), "из", len(df))
dup_tickets = df[df["Ticket"].duplicated(keep=False)].sort_values("Ticket")
print("Примеры повторяющихся билетов:")
print(dup_tickets[["Ticket", "Name", "Pclass", "Fare", "Cabin"]].head(12).to_string(index=False))
print()

print("-- Дубликаты по полному набору признаков БЕЗ PassengerId/Name --")
feat_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]
print("Дубликатов по признакам:", df.duplicated(subset=feat_cols).sum())
