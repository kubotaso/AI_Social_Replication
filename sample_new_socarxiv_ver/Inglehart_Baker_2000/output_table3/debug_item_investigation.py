#!/usr/bin/env python3
"""
Investigate item distributions and try ALL possible codings to maximize correlations.
Also investigate the outgroup index construction more carefully.
"""

import sys, os, csv
import pandas as pd
import numpy as np
from scipy import stats

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

with open(DATA_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]

# Check which A124_ variables exist
a124_vars = [h for h in header if h.startswith('A124')]
print(f"A124 variables in dataset: {a124_vars}")

# Load relevant data
cols = ['COUNTRY_ALPHA', 'S002VS', 'S020',
        'D059', 'D018', 'D022', 'D058', 'D060',
        'A124_02', 'A124_03', 'A124_06', 'A124_07', 'A124_08', 'A124_09',
        'E019', 'B002', 'B003', 'E015', 'E036', 'E037',
        'A030', 'A032', 'A035', 'A002', 'A003',
        'C001', 'C006', 'C011',
        'A009', 'A025', 'A173', 'E014',
        'E114', 'E026', 'E117',
        'F063', 'F125', 'F118', 'F120',
        'G006', 'Y002', 'A008', 'E025', 'A165', 'E018',
        'A006', 'A029', 'A034', 'A042',
        ]
avail = [c for c in cols if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2, 3])]

# Clean
for c in avail:
    if c not in ['COUNTRY_ALPHA', 'S002VS', 'S020']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].where(df[c] >= 0, np.nan)

# Get latest per country
latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
latest.columns = ['COUNTRY_ALPHA', 'latest_year']
df = df.merge(latest, on='COUNTRY_ALPHA')
df = df[df['S020'] == df['latest_year']].drop('latest_year', axis=1)

print(f"\nCountries in dataset: {df['COUNTRY_ALPHA'].nunique()}")
print(f"Countries: {sorted(df['COUNTRY_ALPHA'].unique())}")

# Check D022 distribution
print("\n=== D022 (Child needs both parents) ===")
d022 = df['D022'].dropna()
print(f"Value counts:\n{d022.value_counts().sort_index()}")
# Check by country
d022_by_c = df.groupby('COUNTRY_ALPHA')['D022'].agg(['mean', 'count', 'std'])
d022_by_c = d022_by_c[d022_by_c['count'] >= 30]
print(f"\nD022 country means (top 10 highest):")
print(d022_by_c.sort_values('mean', ascending=False).head(10))
print(f"\nD022 country means (bottom 10):")
print(d022_by_c.sort_values('mean').head(10))

# Check D058 distribution
print("\n=== D058 (University for boy) ===")
d058 = df['D058'].dropna()
print(f"Value counts:\n{d058.value_counts().sort_index()}")
d058_by_c = df.groupby('COUNTRY_ALPHA')['D058'].agg(['mean', 'count'])
d058_by_c = d058_by_c[d058_by_c['count'] >= 30]
print(f"\nD058 country means (top 10):")
print(d058_by_c.sort_values('mean', ascending=False).head(10))
print(f"\nD058 country means (bottom 10):")
print(d058_by_c.sort_values('mean').head(10))

# Check outgroup variables individually
print("\n=== Outgroup variables ===")
for v in ['A124_02', 'A124_03', 'A124_06', 'A124_07']:
    if v in df.columns:
        vals = df[v].dropna()
        print(f"\n{v}: {vals.value_counts().sort_index().to_dict()}, n={len(vals)}")

# Check B002, B003 in wave 2 vs 3
print("\n=== B002/B003 by wave ===")
df_raw = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA', 'B002', 'B003'], low_memory=False)
df_raw = df_raw[df_raw['S002VS'].isin([2, 3])]
for v in ['B002', 'B003']:
    if v in df_raw.columns:
        df_raw[v] = pd.to_numeric(df_raw[v], errors='coerce')
        df_raw[v] = df_raw[v].where(df_raw[v] >= 0, np.nan)
        for wave in [2, 3]:
            wdf = df_raw[df_raw['S002VS'] == wave]
            valid = wdf[v].dropna()
            countries = wdf[wdf[v].notna()]['COUNTRY_ALPHA'].nunique()
            print(f"  {v} wave {wave}: {len(valid)} valid obs, {countries} countries")

# Check E015 distribution
print("\n=== E015 (Science) ===")
e015 = df['E015'].dropna()
print(f"Value counts:\n{e015.value_counts().sort_index()}")

# Check A030, A032, A035 distribution
print("\n=== A030 (Hard work), A032 (Imagination), A035 (Tolerance) ===")
for v in ['A030', 'A032', 'A035']:
    vals = df[v].dropna()
    print(f"\n{v}: {vals.value_counts().sort_index().to_dict()}, n={len(vals)}")

# Check C001 and C011 (job motivation)
print("\n=== C001 and C011 (Job motivation) ===")
for v in ['C001', 'C011']:
    if v in df.columns:
        vals = df[v].dropna()
        print(f"\n{v}: {vals.value_counts().sort_index().to_dict()}, n={len(vals)}")
