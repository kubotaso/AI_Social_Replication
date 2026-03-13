#!/usr/bin/env python3
"""Final variable mapping verification."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

# Check: F034 (religious person), F022 (importance in life?)
# Also A004 (religion important in life), A005 (work important)
# And verify D054 coding

check_cols = ['S002VS', 'COUNTRY_ALPHA', 'F034', 'F022', 'A004', 'A005',
              'D054', 'A025', 'F024', 'E023', 'E033', 'E114', 'D017',
              'E069_01', 'B008', 'F064']

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

avail = ['S002VS', 'COUNTRY_ALPHA'] + [v for v in check_cols[2:] if v in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

for v in check_cols[2:]:
    if v in df.columns:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals >= 0].dropna()
        if len(pos) > 0:
            nga = df[df['COUNTRY_ALPHA'] == 'NGA']
            swe = df[df['COUNTRY_ALPHA'] == 'SWE']
            usa = df[df['COUNTRY_ALPHA'] == 'USA']
            nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            usa_v = pd.to_numeric(usa[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()

            nga_m = f"{nga_v.mean():.2f}" if len(nga_v) > 0 else "N/A"
            swe_m = f"{swe_v.mean():.2f}" if len(swe_v) > 0 else "N/A"
            usa_m = f"{usa_v.mean():.2f}" if len(usa_v) > 0 else "N/A"

            print(f"{v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}, NGA={nga_m}, SWE={swe_m}, USA={usa_m}")
            print(f"  values: {sorted(pos.unique())[:10]}")

# F034: religious person (1=religious, 2=not religious, 3=atheist)
# NGA should be ~1 (very religious), SWE should be ~2
# F022: "How often, if at all, do you think about the meaning of life?" 1=Often...3=Never
# A004: "Important in life: Religion" 1=Very important...4=Not at all
# A005: "Important in life: Work" 1=Very important...4=Not at all

# What about F024 (clear guidelines about good and evil)?
# F024: 1=Clear guidelines, 2=Can never be clear guidelines?
# But we saw F024 N=0 for waves 2-3 when loaded through load_combined_data
# Let me check directly

print("\n=== F024 direct check ===")
if 'F024' in df.columns:
    vals = pd.to_numeric(df['F024'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f"F024: N={len(pos)}")
    if len(pos) > 0:
        print(f"  values: {sorted(pos.unique())[:10]}")
else:
    print("F024 not in columns")

# Check F064 for "believe in God"
print("\n=== F064 (believe in God?) ===")
if 'F064' in df.columns:
    vals = pd.to_numeric(df['F064'], errors='coerce')
    pos = vals[vals >= 0].dropna()
    print(f"F064: N={len(pos)}")
    for c in ['NGA', 'SWE', 'USA', 'JPN', 'CHN']:
        sub = df[df['COUNTRY_ALPHA'] == c]
        sv = pd.to_numeric(sub['F064'], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
        if len(sv) > 0:
            print(f"  {c}: mean={sv.mean():.2f}")

# Check B008 coding
print("\n=== B008 ===")
if 'B008' in df.columns:
    vals = pd.to_numeric(df['B008'], errors='coerce')
    pos = vals[vals > 0].dropna()
    print(f"B008: N={len(pos)}, values={sorted(pos.unique())[:10]}")
    for c in ['NGA', 'SWE', 'USA']:
        sub = df[df['COUNTRY_ALPHA'] == c]
        sv = pd.to_numeric(sub['B008'], errors='coerce').pipe(lambda x: x[x>0]).dropna()
        if len(sv) > 0:
            print(f"  {c}: mean={sv.mean():.2f}")
