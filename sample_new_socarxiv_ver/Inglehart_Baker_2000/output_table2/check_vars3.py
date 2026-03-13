#!/usr/bin/env python3
"""Check specific variable values and distributions for Table 2 items."""
import pandas as pd
import numpy as np
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared_factor_analysis import load_combined_data, clean_missing, get_latest_per_country

df = load_combined_data(waves_wvs=[2, 3], include_evs=True)
df = get_latest_per_country(df)

# Check G007 sub-items - which one is about foreign goods/protectionism
# In WVS, G007 is typically about protectionism (limits on imports)
# Let's check which G007_XX variables have data in waves 2-3
for col in sorted([c for c in df.columns if c.startswith('G007')]):
    valid = df[col].apply(pd.to_numeric, errors='coerce')
    valid = valid[valid >= 0]
    n = len(valid.dropna())
    if n > 1000:
        print(f'{col}: N={n}, values={sorted(valid.dropna().unique())[:10]}')

# Check A029 (make parents proud?)
a029_vals = sorted(df['A029'].apply(pd.to_numeric, errors='coerce').dropna().unique())[:10]
print(f'\nA029: values={a029_vals}')
a029_pos = df['A029'].apply(pd.to_numeric, errors='coerce').pipe(lambda x: x[x>=0]).dropna()
print(f'A029 N (>=0): {len(a029_pos)}')

# Check A003 (make parents proud might be here?)
a003_vals = sorted(df['A003'].apply(pd.to_numeric, errors='coerce').dropna().unique())[:10]
print(f'\nA003: values={a003_vals}')

# Check variables A038-A042 (children qualities)
for v in ['A029', 'A030', 'A032', 'A034', 'A035', 'A038', 'A039', 'A040', 'A041', 'A042']:
    if v in df.columns:
        vals = df[v].apply(pd.to_numeric, errors='coerce')
        vals = vals[vals >= 0]
        print(f'{v}: N={len(vals.dropna())}, unique={sorted(vals.dropna().unique())[:10]}')

# Check B008 (environmental problems / international)
if 'B008' in df.columns:
    vals = df['B008'].apply(pd.to_numeric, errors='coerce')
    vals = vals[vals >= 0]
    print(f'\nB008: N={len(vals.dropna())}, values={sorted(vals.dropna().unique())[:10]}')

# Check E114 (army rule)
if 'E114' in df.columns:
    vals = df['E114'].apply(pd.to_numeric, errors='coerce')
    vals = vals[vals >= 0]
    print(f'\nE114: N={len(vals.dropna())}, values={sorted(vals.dropna().unique())[:10]}')

# Check D017 (ideal number of children)
if 'D017' in df.columns:
    vals = df['D017'].apply(pd.to_numeric, errors='coerce')
    vals = vals[vals >= 0]
    print(f'\nD017: N={len(vals.dropna())}, values={sorted(vals.dropna().unique())[:15]}')

# Number of countries
print(f'\nTotal countries: {df["COUNTRY_ALPHA"].nunique()}')
print(f'Countries: {sorted(df["COUNTRY_ALPHA"].unique())}')
