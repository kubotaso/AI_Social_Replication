#!/usr/bin/env python3
"""Check church attendance issue and country set."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

# Check F028 in EVS
if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    if 'F028' in evs.columns:
        evs['F028n'] = pd.to_numeric(evs['F028'], errors='coerce')
        evs.loc[evs['F028n'] < 0, 'F028n'] = np.nan
        print("=== EVS F028 (church attendance) ===")
        for c in sorted(evs['COUNTRY_ALPHA'].unique()):
            sub = evs[evs['COUNTRY_ALPHA'] == c]['F028n'].dropna()
            if len(sub) > 0:
                print(f"  {c}: mean={sub.mean():.2f}, range={sub.min():.0f}-{sub.max():.0f}, N={len(sub)}")
    else:
        print("F028 not in EVS data")

# Check what the paper's 65 societies are
# From the paper footnotes: 65 societies from 1990-1991 and 1995-1998 WVS
# This includes both WVS and EVS societies
# Let me check which countries we get
print("\n=== Country set analysis ===")

# WVS waves 2-3
wvs = pd.read_csv(DATA_PATH, usecols=['S002VS', 'COUNTRY_ALPHA'], low_memory=False)
wvs23 = wvs[wvs['S002VS'].isin([2, 3])]
wvs_countries = sorted(wvs23['COUNTRY_ALPHA'].unique())
print(f"WVS waves 2-3: {len(wvs_countries)} countries")

# EVS
evs_countries = sorted(evs['COUNTRY_ALPHA'].unique()) if os.path.exists(EVS_PATH) else []
print(f"EVS: {len(evs_countries)} countries: {evs_countries}")

# Combined
all_countries = sorted(set(wvs_countries) | set(evs_countries))
print(f"Combined: {len(all_countries)} countries")

# Check which are EVS-only
evs_only = [c for c in evs_countries if c not in wvs_countries]
print(f"EVS-only: {evs_only}")

# Check for the paper's likely 65 societies
# The paper mentions East and West Germany as separate, Northern Ireland separate
# Also might include Puerto Rico
print(f"\nAll: {all_countries}")

# Countries that might be in paper but we might be missing or coding differently:
# GHA (Ghana) - check if in our data
for c in ['GHA', 'DEU', 'DEU_W', 'DEU_E', 'BEL', 'MLT', 'MNE', 'SLV', 'ALB']:
    if c in all_countries:
        print(f"  {c}: present")
    else:
        print(f"  {c}: MISSING")
