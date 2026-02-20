import numpy as np
import pandas as pd

print(pd.read_parquet("data/processed/student_features.parquet"))
print(pd.read_parquet("data/processed/labels.parquet"))
