import numpy as np
import pandas as pd

d1 = pd.read_csv("dataset/student-mat.csv", sep=";")
d2 = pd.read_csv("dataset/student-por.csv", sep=";")

df = pd.concat([d1, d2], ignore_index=True)

# ETL Process
# Apply log transform for skewed dataset
df["abs_log"] = np.log1p(df["absences"])

# Create new feature using 'Failures' x 'absences'
df["fail_abs"] = df["failures"] * df["absences"]

# Add student_id and timestamp for Feast feature store
df["student_id"] = range(len(df))
df["event_timestamp"] = pd.Timestamp.now()
MARK_THRESHOLD = 10
df["dropout"] = (df["G3"] < MARK_THRESHOLD).astype(int)

print(df.info())

df.to_parquet("student_feature/data/student_features.parquet", index=False)
df[["student_id", "event_timestamp", "dropout"]].to_parquet(
    "student_feature/data/labels.parquet", index=False
)

print("-" * 100)
# Store in columnar data storage for Big data
print(pd.read_parquet("student_feature/data/student_features.parquet"))
print(pd.read_parquet("student_feature/data/labels.parquet"))
