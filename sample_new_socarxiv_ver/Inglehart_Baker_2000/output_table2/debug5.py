#!/usr/bin/env python3
"""Find 1-10 God importance variable in WVS time series."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Search all variables for 1-10 range that might be "importance of God"
# In WVS, the "importance of God" variable is typically F063 or in the religion section
# But it might also be stored as a different variable name in the time series
#
# The original WVS variable for "How important is God in your life" (1-10) is
# typically v186 or similar in raw WVS. In the harmonized time series, it might be
# stored under a different variable name.
#
# Let me check ALL variables with 1-10 range for NGA (should be ~9-10) vs SWE (~3-4)
candidates = ['F025', 'F062', 'F064', 'F065', 'F066', 'F067']
# Also check some A-series
for prefix in header:
    pass  # header too large to iterate

# Check specific candidates
check_cols = ['S002VS', 'COUNTRY_ALPHA'] + [v for v in candidates if v in header]
df = pd.read_csv(DATA_PATH, usecols=check_cols, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

for v in candidates:
    if v in df.columns:
        vals = pd.to_numeric(df[v], errors='coerce')
        pos = vals[vals >= 0].dropna()
        if len(pos) > 0:
            nga = df[df['COUNTRY_ALPHA'] == 'NGA']
            swe = df[df['COUNTRY_ALPHA'] == 'SWE']
            nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>=0]).dropna()
            print(f"{v}: range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}")
            if len(nga_v) > 0:
                print(f"  NGA: mean={nga_v.mean():.2f}")
            if len(swe_v) > 0:
                print(f"  SWE: mean={swe_v.mean():.2f}")

# The key insight: the paper uses "God is important in respondent's life" on a 1-10 scale
# In WVS Time Series v5, this is likely stored as variable F063? No, F063 is church attendance.
# Let me try reading the data documentation or looking for any variable with 1-10 range
# where Nigeria is ~9-10 and Sweden is ~3-4

# A brute force approach: check ALL columns for waves 2-3
# Look for columns where NGA mean > 8 and SWE mean < 5 and range is 1-10
print("\n=== Brute force search for God importance (1-10) ===")
print("Looking for vars where NGA > 8 and SWE < 5...")

# Read all columns that might be religion-related
all_f = [h for h in header if h.startswith('F') and '_' not in h]
check2 = ['S002VS', 'COUNTRY_ALPHA'] + all_f
df2 = pd.read_csv(DATA_PATH, usecols=check2, low_memory=False)
df2 = df2[df2['S002VS'].isin([2, 3])]

for v in all_f:
    vals = pd.to_numeric(df2[v], errors='coerce')
    pos = vals[vals > 0].dropna()
    if len(pos) > 10000 and pos.max() >= 9:
        nga = df2[df2['COUNTRY_ALPHA'] == 'NGA']
        swe = df2[df2['COUNTRY_ALPHA'] == 'SWE']
        nga_v = pd.to_numeric(nga[v], errors='coerce').pipe(lambda x: x[x>0]).dropna()
        swe_v = pd.to_numeric(swe[v], errors='coerce').pipe(lambda x: x[x>0]).dropna()
        if len(nga_v) > 100 and len(swe_v) > 100:
            if nga_v.mean() > 7 and swe_v.mean() < 6:
                print(f"  CANDIDATE: {v}, range={pos.min():.0f}-{pos.max():.0f}, N={len(pos)}, NGA={nga_v.mean():.2f}, SWE={swe_v.mean():.2f}")
