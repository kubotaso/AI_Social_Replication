#!/usr/bin/env python3
"""Investigate the N discrepancy."""
import sys, os, csv
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import clean_missing

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WVS_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

cols = ['S002VS', 'COUNTRY_ALPHA', 'S020',
        'A006', 'A008', 'A029', 'A030', 'A032', 'A034', 'A042',
        'A165', 'E018', 'E025', 'F063', 'F118', 'F120', 'G006', 'Y002']

with open(WVS_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]
avail = [c for c in cols if c in header]

wvs = pd.read_csv(WVS_PATH, usecols=avail, low_memory=False)
wvs['_src'] = 'wvs'

evs = pd.read_csv(EVS_PATH)
evs['_src'] = 'evs'

# Check: total N using BOTH wave 2 and wave 3 for individual analysis
# Paper says "Instead of 123 cases, we now have 165,594 cases"
# The paper uses ALL individual responses, not just latest wave per country

# Option A: waves 2+3 only
w23 = wvs[wvs['S002VS'].isin([2, 3])]
print(f"WVS waves 2+3: {len(w23):,} rows")
print(f"EVS: {len(evs):,} rows")
print(f"Combined: {len(w23) + len(evs):,} rows")

# Option B: wave 3 only
w3 = wvs[wvs['S002VS'] == 3]
print(f"\nWVS wave 3 only: {len(w3):,} rows")
print(f"WVS wave 3 + EVS: {len(w3) + len(evs):,} rows")

# For individual analysis, the paper uses ALL individual data from all waves
# that cover the 65 societies. This includes:
# - EVS 1990-1991 data
# - WVS wave 2 (1990-1994) data
# - WVS wave 3 (1995-1998) data
# Total should be ~165,594

# Let me check N per country for waves 2+3 + EVS
df = pd.concat([w23, evs], ignore_index=True, sort=False)
df = df[~df['COUNTRY_ALPHA'].isin(['MNE'])]
print(f"\nCombined (excl MNE): {len(df):,} rows")

# But wait - EVS and WVS overlap for some countries!
# Check overlap
evs_countries = set(evs['COUNTRY_ALPHA'].unique())
wvs_countries = set(w23['COUNTRY_ALPHA'].unique())
overlap = evs_countries & wvs_countries
print(f"\nEVS countries: {len(evs_countries)}")
print(f"WVS wave 2+3 countries: {len(wvs_countries)}")
print(f"Overlap: {len(overlap)} countries: {sorted(overlap)}")
print(f"EVS only: {sorted(evs_countries - wvs_countries)}")
print(f"WVS only: {sorted(wvs_countries - evs_countries)}")

# The paper says "65 societies" but mentions using the latest survey.
# For individual-level: the paper might use the latest survey per COUNTRY
# but that would give less than 165,594.
# OR: the paper uses ALL observations from ALL waves for individual-level.
# Let's check:
from shared_factor_analysis import get_latest_per_country
df_latest = get_latest_per_country(df)
print(f"\nLatest wave per country N: {len(df_latest):,}")
print(f"All waves N: {len(df):,}")

# How about dedup? For countries in both EVS and WVS wave 2, they might
# use only the WVS version
# Let's try: EVS only for countries NOT in WVS + WVS wave 2+3
evs_only = evs[~evs['COUNTRY_ALPHA'].isin(wvs_countries)]
df_no_overlap = pd.concat([w23, evs_only], ignore_index=True, sort=False)
df_no_overlap = df_no_overlap[~df_no_overlap['COUNTRY_ALPHA'].isin(['MNE'])]
print(f"\nWVS 2+3 + EVS (no overlap): {len(df_no_overlap):,} rows, {df_no_overlap['COUNTRY_ALPHA'].nunique()} countries")

# Or: use EVS for countries in EVS + WVS for the rest
wvs_not_evs = w23[~w23['COUNTRY_ALPHA'].isin(evs_countries)]
df_evs_first = pd.concat([evs, wvs_not_evs], ignore_index=True, sort=False)
df_evs_first = df_evs_first[~df_evs_first['COUNTRY_ALPHA'].isin(['MNE'])]
print(f"EVS first + WVS (non-EVS): {len(df_evs_first):,} rows, {df_evs_first['COUNTRY_ALPHA'].nunique()} countries")

# N per country for the "all data" approach
print("\nN per country (all waves combined):")
cn = df.groupby('COUNTRY_ALPHA').size().sort_values(ascending=False)
for c, n in cn.items():
    src = []
    if c in evs_countries: src.append('EVS')
    wvs_waves = sorted(w23[w23['COUNTRY_ALPHA']==c]['S002VS'].unique())
    if wvs_waves: src.append(f"WVS w{wvs_waves}")
    print(f"  {c}: {n:,} ({', '.join(src)})")

print(f"\nTotal countries: {df['COUNTRY_ALPHA'].nunique()}")
print(f"Target: 65")
