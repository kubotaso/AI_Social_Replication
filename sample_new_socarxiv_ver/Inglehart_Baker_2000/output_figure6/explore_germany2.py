#!/usr/bin/env python3
"""Check for region variables to split East/West Germany."""
import pandas as pd
import csv

with open('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

# Look for region variables
region_vars = [c for c in header if c.startswith('X') or c.startswith('G') or 'REGION' in c.upper()]
print(f"Potential region variables: {region_vars[:30]}")

# Load Germany data with potential region vars
needed = ['S002VS', 'COUNTRY_ALPHA', 'S020', 'S024', 'X047', 'X048', 'X049', 'X002']
available = [c for c in needed if c in header]
wvs = pd.read_csv('data/WVS_Time_Series_1981-2022_csv_v5_0.csv', usecols=available, low_memory=False)
deu = wvs[(wvs['COUNTRY_ALPHA']=='DEU') & (wvs['S002VS']==3)]

print(f"\nDEU wave 3: n={len(deu)}")
for col in available:
    if col not in ['S002VS', 'COUNTRY_ALPHA', 'S020']:
        vals = deu[col].dropna().unique()
        if len(vals) <= 20:
            print(f"  {col}: {sorted(vals)}")
        else:
            print(f"  {col}: {len(vals)} unique values, range [{min(vals)}, {max(vals)}]")

# The X002 values (1907-1979) are birth years. Not helpful for East/West.
# Let's look at the EVS data more carefully
evs = pd.read_csv('data/EVS_1990_wvs_format.csv')
deu_evs = evs[evs['COUNTRY_ALPHA']=='DEU']
print(f"\nEVS DEU columns: {list(evs.columns)}")
print(f"EVS DEU n={len(deu_evs)}")

# In the original paper, East Germany 1990 data came from the EVS 1990 survey
# which had a separate East Germany survey (after reunification in late 1990).
# EVS had separate codes for East and West Germany.
# But in our EVS_1990_wvs_format.csv, they may all be coded as DEU.
# Check if S001 could help
if 'S001' in evs.columns:
    deu_s001 = deu_evs['S001'].unique()
    print(f"EVS DEU S001: {sorted(deu_s001)}")
