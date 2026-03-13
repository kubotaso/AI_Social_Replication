#!/usr/bin/env python3
"""Explore NLD data availability across waves."""
import pandas as pd
import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Check EVS 1990 for NLD with religion variables
evs = pd.read_csv(os.path.join(BASE_DIR, 'data/EVS_1990_wvs_format.csv'))
print('EVS 1990 religion columns:', [c for c in evs.columns if c.startswith('F') or 'COUNTRY' in c.upper()])
print()

if 'COUNTRY_ALPHA' in evs.columns:
    nld_evs = evs[evs['COUNTRY_ALPHA'] == 'NLD']
    print(f'NLD in EVS 1990: {len(nld_evs)} rows')
    for col in ['F025', 'F024', 'F028', 'F034', 'F030']:
        if col in evs.columns:
            vals = nld_evs[col][nld_evs[col] >= 0] if len(nld_evs) > 0 else pd.Series()
            print(f'  {col}: {vals.value_counts().sort_index().to_dict() if len(vals) > 0 else "empty"}')
else:
    print('EVS columns:', list(evs.columns[:30]))

# Check all F-prefix columns in EVS for NLD
print()
if len(nld_evs) > 0:
    f_cols = [c for c in evs.columns if c.startswith('F')]
    print('F columns with data for NLD:')
    for c in f_cols:
        vals = nld_evs[c][pd.to_numeric(nld_evs[c], errors='coerce') >= 0]
        if len(vals) > 0:
            print(f'  {c}: {pd.to_numeric(vals, errors="coerce").value_counts().sort_index().head(10).to_dict()}')

print()
print('=' * 60)
print('WVS waves for NLD:')
wvs = pd.read_csv(os.path.join(BASE_DIR, 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'),
                   usecols=['S002VS', 'COUNTRY_ALPHA', 'F025', 'S020'],
                   low_memory=False)
for wave in [1, 2, 3, 4, 5, 6, 7]:
    nld = wvs[(wvs['COUNTRY_ALPHA']=='NLD') & (wvs['S002VS']==wave)]
    if len(nld) > 0:
        f025_clean = nld['F025'][pd.to_numeric(nld['F025'], errors='coerce') >= 0]
        year = nld['S020'].iloc[0] if len(nld) > 0 else 'N/A'
        print(f'  Wave {wave}: n={len(nld)}, year~{year}')
        print(f'    F025: {pd.to_numeric(f025_clean, errors="coerce").value_counts().sort_index().to_dict()}')
