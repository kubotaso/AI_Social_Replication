#!/usr/bin/env python3
"""Verify F063 coding in WVS time series."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

df = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'F063'], low_memory=False)
df = df[df['S002VS'].isin([2, 3])]
df['F063'] = pd.to_numeric(df['F063'], errors='coerce')
df = df[df['F063'] >= 0]

print("=== F063 distribution across countries ===")
for c in ['NGA', 'SWE', 'JPN', 'USA', 'BRA', 'POL', 'CHN', 'IND']:
    sub = df[df['COUNTRY_ALPHA'] == c]['F063']
    if len(sub) > 0:
        print(f"{c}: mean={sub.mean():.2f}, values={sorted(sub.unique())}")

# This looks like "importance of God" (1=not at all, 10=very important)
# NOT church attendance (which would be 1-7 or 1-8 scale)
# In the WVS Time Series V5, F063 IS "How important is God in your life" (1-10)
# Church attendance might be a different variable

# Check what column has church attendance coding (1=weekly... 7=never)
print("\n=== Looking for church attendance variable ===")
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Check F064, F065, etc.
# Also check A058-A062 area
candidates = ['F028B', 'F031', 'F032', 'F033', 'F034', 'F035', 'F036', 'F037',
              'F038', 'F040', 'F041', 'F042', 'F043', 'F044', 'F045', 'F046', 'F047',
              'F048', 'F049', 'F062', 'F064', 'F065', 'F066', 'F067']
avail = [v for v in candidates if v in header]
df2 = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA'] + avail, low_memory=False)
df2 = df2[df2['S002VS'].isin([2, 3])]

for v in avail:
    vals = pd.to_numeric(df2[v], errors='coerce')
    pos = vals[vals >= 0].dropna()
    if len(pos) > 10000:
        mx = pos.max()
        if mx in [7, 8, 9]:  # Church attendance typically 7 or 8 categories
            nga = df2[df2['COUNTRY_ALPHA'] == 'NGA']
            swe = df2[df2['COUNTRY_ALPHA'] == 'SWE']
            nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            print(f"  {v}: range={pos.min():.0f}-{mx:.0f}, N={len(pos)}")
            if len(nga_v) > 0:
                print(f"    NGA: mean={nga_v.mean():.2f} (should be low=frequent if church attend)")
            if len(swe_v) > 0:
                print(f"    SWE: mean={swe_v.mean():.2f} (should be high=infrequent)")
