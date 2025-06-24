import pandas as pd

# 1. Read the Excel file
df = pd.read_excel("data/online_retail_II.xlsx")

# 2.Check number of rows and columns
print("Rows:", len(df), "Columns:", df.shape[1])

# 3. Save everything to sales.csv
# df.to_csv("sales.csv", index=False)
# print("sales.csv created with", len(df), "rows")

df_sample = df.sample(n=100_000, random_state=42)
df_sample.to_csv("data/sales.csv", index=False)
print("✅ sales.csv created with 100,000 sampled rows")
print("Rows:", len(df_sample), "Columns:", df_sample.shape[1])