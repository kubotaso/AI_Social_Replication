#!/usr/bin/env python3
"""Debug: Check raw data before cleaning, and check which wave countries appear in."""
import pandas as pd
import numpy as np
import csv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
EVS_PATH = 'data/EVS_1990_wvs_format.csv'

with open(DATA_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]

usecols = ['S002VS','COUNTRY_ALPHA','S020','D058','E015','D022','F125','B003','C001']
avail = [c for c in usecols if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)

# Check waves 2 and 3 separately for D058
print("D058 raw distribution by wave:")
for wave in [2, 3]:
    wdf = df[df['S002VS']==wave]
    print(f"  Wave {wave}: n={len(wdf)}")
    if 'D058' in wdf.columns:
        print(f"    D058 value_counts:")
        vc = wdf['D058'].value_counts().sort_index()
        for v, c in vc.items():
            print(f"      {v}: {c}")
    print()

# D058 by wave and country for Sweden
print("D058 for Sweden by wave:")
for wave in [2, 3]:
    wdf = df[(df['S002VS']==wave) & (df['COUNTRY_ALPHA']=='SWE')]
    if len(wdf) > 0 and 'D058' in wdf.columns:
        vc = wdf['D058'].value_counts().sort_index()
        print(f"  Wave {wave}: {dict(vc)}")

# D058 for Nigeria by wave
print("\nD058 for Nigeria by wave:")
for wave in [2, 3]:
    wdf = df[(df['S002VS']==wave) & (df['COUNTRY_ALPHA']=='NGA')]
    if len(wdf) > 0 and 'D058' in wdf.columns:
        vc = wdf['D058'].value_counts().sort_index()
        print(f"  Wave {wave}: {dict(vc)}")

# D058 for Pakistan
print("\nD058 for Pakistan by wave:")
for wave in [2, 3]:
    wdf = df[(df['S002VS']==wave) & (df['COUNTRY_ALPHA']=='PAK')]
    if len(wdf) > 0 and 'D058' in wdf.columns:
        vc = wdf['D058'].value_counts().sort_index()
        print(f"  Wave {wave}: {dict(vc)}")

# Check F125 and B003 by wave
for var in ['F125', 'B003']:
    print(f"\n{var} non-missing counts by wave:")
    for wave in [2, 3]:
        wdf = df[df['S002VS']==wave]
        if var in wdf.columns:
            valid = wdf[var].apply(lambda x: x >= 0 if pd.notna(x) else False).sum()
            print(f"  Wave {wave}: {valid} valid obs")

# Check how many countries per wave
print("\nCountries per wave:")
for wave in [2, 3]:
    wdf = df[df['S002VS']==wave]
    print(f"  Wave {wave}: {wdf['COUNTRY_ALPHA'].nunique()} countries")

# Check C001 coding by wave
print("\nC001 by wave:")
for wave in [2, 3]:
    wdf = df[df['S002VS']==wave]
    if 'C001' in wdf.columns:
        vc = wdf['C001'].value_counts().sort_index()
        print(f"  Wave {wave}:")
        for v, c in vc.items():
            print(f"    {v}: {c}")

# Check E015 by wave
print("\nE015 by wave:")
for wave in [2, 3]:
    wdf = df[df['S002VS']==wave]
    if 'E015' in wdf.columns:
        vc = wdf['E015'].value_counts().sort_index()
        print(f"  Wave {wave}:")
        for v, c in vc.items():
            print(f"    {v}: {c}")

# Check D022 more carefully
print("\nD022 raw by wave and country for SWE, NGA, TUR:")
for country in ['SWE','NGA','TUR','AZE']:
    for wave in [2, 3]:
        wdf = df[(df['S002VS']==wave) & (df['COUNTRY_ALPHA']==country)]
        if len(wdf) > 0 and 'D022' in wdf.columns:
            vc = wdf['D022'].value_counts().sort_index()
            print(f"  {country} wave {wave}: {dict(vc)}")
