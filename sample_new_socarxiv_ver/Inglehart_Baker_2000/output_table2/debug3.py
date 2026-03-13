#!/usr/bin/env python3
"""Find the God importance 1-10 variable."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Look for F variables that might be God importance
f_vars = [h for h in header if h.startswith('F')]
print("F-prefix variables:", sorted(f_vars))

# Check F001 (which might be "importance of God" in WVS)
print("\n=== Checking F001 ===")
check_cols = ['S002VS', 'COUNTRY_ALPHA']
for v in ['F001', 'F063', 'F028', 'A006']:
    if v in header:
        check_cols.append(v)

df = pd.read_csv(DATA_PATH, usecols=check_cols, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

for v in ['F001', 'A006']:
    if v in df.columns:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals >= 0]
        print(f"\n{v}: N={len(pos.dropna())}, range={pos.min()}-{pos.max()}")
        for c in ['NGA', 'SWE', 'JPN', 'USA']:
            sub = df[df['COUNTRY_ALPHA'] == c]
            sv = pd.to_numeric(sub[v], errors='coerce')
            sv = sv[sv >= 0]
            if len(sv) > 0:
                print(f"  {c}: mean={sv.mean():.2f}, values={sorted(sv.unique())[:15]}")
