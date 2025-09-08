#!/usr/bin/env python3
import pandas as pd
import numpy as np
import json

# Load the dataset
data_path = '/Users/avivnahon/ds-agent-projects/session_89600b04-b810-4506-b66d-91e28f4f611b/data/train_set.csv'
df = pd.read_csv(data_path)

print("Dataset Analysis:")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")
print(f"Target column unique values: {df['target'].unique()}")
print(f"Target distribution:\n{df['target'].value_counts()}")

# Analyze each column in detail
for col in df.columns:
    print(f"\n=== {col} ===")
    print(f"Data type: {df[col].dtype}")
    print(f"Unique values: {df[col].nunique()}")
    print(f"Missing values: {df[col].isnull().sum()}")
    
    if df[col].dtype == 'object':
        print(f"Value counts:\n{df[col].value_counts()}")
    else:
        print(f"Statistics:\n{df[col].describe()}")

# Let's also check for any patterns in the column names and values
print("\nColumn analysis for mapping:")
print(f"V1 (likely age): min={df['V1'].min()}, max={df['V1'].max()}")
print(f"V2 unique values: {df['V2'].unique()}")
print(f"V3 unique values: {df['V3'].unique()}")
print(f"V4 unique values: {df['V4'].unique()}")
print(f"V5 unique values: {df['V5'].unique()}")

# Analyze missing values pattern
print(f"\nMissing values analysis:")
missing_info = df.isnull().sum()
print(missing_info[missing_info > 0])