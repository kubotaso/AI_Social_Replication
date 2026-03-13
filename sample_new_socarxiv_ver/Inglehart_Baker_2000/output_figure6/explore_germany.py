#!/usr/bin/env python3
"""Explore Germany split possibilities."""
import pandas as pd
import numpy as np

wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv',
                   usecols=['S002VS','COUNTRY_ALPHA','S020','S003','S024','X002'],
                   low_memory=False)
deu = wvs[(wvs['COUNTRY_ALPHA']=='DEU') & (wvs['S002VS'].isin([1,2,3]))]
print(f"DEU wave 3: n={len(deu)}")
print(f"S024 values: {sorted(deu['S024'].unique())}")

# Check S003 values for Germany
# 276 = West Germany
# 278 = could be East Germany
# Let's also check other potential codes
deu_s003 = wvs[(wvs['S003'].isin([276, 278])) & (wvs['S002VS'].isin([1,2,3]))]
print(f"\nS003=276 or 278 in waves 1-3:")
for s3 in sorted(deu_s003['S003'].unique()):
    sub = deu_s003[deu_s003['S003']==s3]
    print(f"  S003={s3}: waves={sorted(sub.S002VS.unique())}, "
          f"countries={sorted(sub.COUNTRY_ALPHA.unique())}, n={len(sub)}")

# In WVS documentation, S024 encodes regions:
# For Germany: 2761=West, 2762=East (perhaps)
# Or: 2763 = unified
# Let's check all S024 values for Germany
for w in [1,2,3]:
    sub = deu[deu['S002VS']==w]
    if len(sub) > 0:
        print(f"\nDEU wave {w}:")
        print(f"  S024: {sorted(sub.S024.unique())}")
        if 'X002' in sub.columns:
            print(f"  X002 (region): {sorted(sub.X002.dropna().unique())}")

# Also check EVS for East/West Germany
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
print(f"\nEVS DEU: n={len(deu_evs)}")
for col in evs.columns:
    if col.startswith('S0') or col.startswith('X0'):
        if col in deu_evs.columns:
            unique_vals = deu_evs[col].dropna().unique()
            if len(unique_vals) <= 20:
                print(f"  {col}: {sorted(unique_vals)}")
