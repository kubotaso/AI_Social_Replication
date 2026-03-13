#!/usr/bin/env python3
"""Find the 1-10 importance of God variable."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# F034 = religious person (1-3)
# F063 = church attendance
# The "importance of God" 1-10 is typically stored as F063 or in the F-series
# Actually in WVS it's usually stored as specific variable
# Let me check ALL variables with range 1-10 in the F-series

# Check F variables that might have 1-10 range
f_candidates = ['F001', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                'F022', 'F025', 'F027']
check_cols = ['S002VS', 'COUNTRY_ALPHA'] + [v for v in f_candidates if v in header]

df = pd.read_csv(DATA_PATH, usecols=check_cols, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

for v in f_candidates:
    if v in df.columns:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals >= 0].dropna()
        if len(pos) > 0 and pos.max() >= 9:
            # This might be 1-10 scale
            nga = df[df['COUNTRY_ALPHA'] == 'NGA']
            swe = df[df['COUNTRY_ALPHA'] == 'SWE']
            nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            print(f"{v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}")
            if len(nga_v) > 0:
                print(f"  NGA: mean={nga_v.mean():.2f}")
            if len(swe_v) > 0:
                print(f"  SWE: mean={swe_v.mean():.2f}")
        elif len(pos) > 0:
            print(f"{v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)} (not 1-10)")

# Also check if the 1-10 God importance is simply A006 in the EVS data
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")
if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    if 'A006' in evs.columns:
        a006_evs = pd.to_numeric(evs['A006'], errors='coerce')
        a006_pos = a006_evs[a006_evs >= 0].dropna()
        print(f"\nEVS A006: range={a006_pos.min():.0f}-{a006_pos.max():.0f}, N={len(a006_pos)}")
        print(f"  values: {sorted(a006_pos.unique())[:15]}")
