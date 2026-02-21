import numpy as np
import pandas as pd

print(pd.read_parquet("data/processed/student_features.parquet"))
print(pd.read_parquet("data/processed/labels.parquet"))

df = pd.read_parquet("data/processed/student_features.parquet")


print("Columns:", df.columns)
print("Row count:", len(df))
print(df.head())

print("Timestamp dtype:", df["event_timestamp"].dtype)
print("Min timestamp:", df["event_timestamp"].min())
print("Max timestamp:", df["event_timestamp"].max())
