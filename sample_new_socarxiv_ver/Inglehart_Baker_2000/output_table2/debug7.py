#!/usr/bin/env python3
"""Find church attendance variable and verify all factor analysis items."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Look for church attendance - typically coded as frequency
# 1=more than once a week, 2=once a week, ..., 7 or 8=never
# In WVS harmonized data, this could be F028B or in A-series
# Or maybe F064/F065 area

# Check all F-prefix with 5-8 range that might be church attendance
candidates = [h for h in header if h.startswith('F0')]
check = ['S002VS', 'COUNTRY_ALPHA'] + candidates
df = pd.read_csv(DATA_PATH, usecols=check, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

print("=== Looking for church attendance (7-8 point scale, NGA=low/frequent, SWE=high/infrequent) ===")
for v in candidates:
    vals = pd.to_numeric(df[v], errors='coerce')
    pos = vals[vals > 0].dropna()
    if len(pos) > 20000 and pos.max() in [7, 8, 9]:
        nga = df[df['COUNTRY_ALPHA'] == 'NGA']
        swe = df[df['COUNTRY_ALPHA'] == 'SWE']
        pol = df[df['COUNTRY_ALPHA'] == 'POL']
        nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>0]).dropna()
        swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>0]).dropna()
        pol_v = pd.to_numeric(pol[v], errors='coerce').pipe(lambda x: x[x>0]).dropna()
        # Church attendance: NGA should be low (frequent), SWE should be high (infrequent)
        if len(nga_v) > 100 and len(swe_v) > 100:
            if nga_v.mean() < swe_v.mean():
                print(f"  {v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}")
                pol_str = f"{pol_v.mean():.2f}" if len(pol_v)>0 else "N/A"
                print(f"    NGA={nga_v.mean():.2f}, SWE={swe_v.mean():.2f}, POL={pol_str}")

# Also check within the A-series
print("\n=== Checking broader range ===")
for v in ['A062']:
    if v in header:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals > 0].dropna()
        if len(pos) > 0:
            print(f"{v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}")

# Maybe church attendance is F028B
print("\n=== F028B ===")
if 'F028B' in df.columns:
    vals = pd.to_numeric(df['F028B'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f"F028B: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}")
    for c in ['NGA', 'SWE', 'USA', 'POL']:
        sub = df[df['COUNTRY_ALPHA'] == c]
        sv = pd.to_numeric(sub['F028B'], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
        if len(sv) > 0:
            print(f"  {c}: mean={sv.mean():.2f}, values={sorted(sv.unique())[:10]}")
