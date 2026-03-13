#!/usr/bin/env python3
"""Check B008, F024, G007 directly from raw CSV (not through load_combined_data)."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

# Read just the columns we need to check
check_cols = ['S002VS', 'COUNTRY_ALPHA', 'B008', 'F024', 'G007_01', 'G007_02', 'G007_03',
              'G007_04', 'G007_05', 'G007_06', 'G007_07', 'G007_08', 'G007_09', 'G007_10',
              'A029', 'D054', 'A003']

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

avail = [c for c in check_cols if c in header]
print(f'Available from check list: {avail}')
print(f'Missing from check list: {[c for c in check_cols if c not in header]}')

df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)

# Filter to waves 2 and 3
df23 = df[df['S002VS'].isin([2, 3])]
print(f'\nRows in waves 2-3: {len(df23)}')

for col in avail:
    if col in ['S002VS', 'COUNTRY_ALPHA']:
        continue
    vals = pd.to_numeric(df23[col], errors='coerce')
    pos = vals[vals >= 0].dropna()
    n = len(pos)
    if n > 0:
        print(f'{col}: N={n}, values={sorted(pos.unique())[:10]}')
    else:
        # Check if there's data in other waves
        vals_all = pd.to_numeric(df[col], errors='coerce')
        pos_all = vals_all[vals_all >= 0].dropna()
        print(f'{col}: N=0 in waves 2-3, N={len(pos_all)} in all waves')

# Also check what G007 really is - let's look at wave 3 specifically
print('\n=== Wave 3 G007 sub-items ===')
w3 = df[df['S002VS'] == 3]
for col in [c for c in avail if c.startswith('G007')]:
    vals = pd.to_numeric(w3[col], errors='coerce')
    pos = vals[vals >= 0].dropna()
    if len(pos) > 0:
        print(f'{col} (wave3): N={len(pos)}, values={sorted(pos.unique())[:10]}')

print('\n=== Wave 2 G007 sub-items ===')
w2 = df[df['S002VS'] == 2]
for col in [c for c in avail if c.startswith('G007')]:
    vals = pd.to_numeric(w2[col], errors='coerce')
    pos = vals[vals >= 0].dropna()
    if len(pos) > 0:
        print(f'{col} (wave2): N={len(pos)}, values={sorted(pos.unique())[:10]}')
