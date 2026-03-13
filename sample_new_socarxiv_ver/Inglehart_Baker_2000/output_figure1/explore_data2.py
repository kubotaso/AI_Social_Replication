#!/usr/bin/env python3
"""Explore Germany region codes and EVS data for E/W split."""
import pandas as pd
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

# In the WVS, X048WVS for Germany:
# East German states (new Bundeslaender): 276012-276016 + Berlin-East
# West German states (old Bundeslaender): 276001-276011 + Berlin-West
# Common split: codes >= 276012 = East Germany

# Check wave 3 Germany with X048WVS
cols = ['S002VS', 'COUNTRY_ALPHA', 'X048WVS'] + ['A006', 'A042', 'F120', 'G006', 'E018', 'Y002', 'A008', 'E025', 'F118', 'A165']
df = pd.read_csv(DATA_PATH, usecols=[c for c in cols if c in open(DATA_PATH).readline().replace('"','').split(',')], low_memory=False)
deu3 = df[(df['COUNTRY_ALPHA'] == 'DEU') & (df['S002VS'] == 3)]
print(f"Wave 3 Germany rows: {len(deu3)}")
print(f"X048WVS distribution:")
vc = deu3['X048WVS'].value_counts().sort_index()
for code, count in vc.items():
    label = "East" if code >= 276012 else "West"
    print(f"  {int(code)}: {count} ({label})")

east = deu3[deu3['X048WVS'] >= 276012]
west = deu3[deu3['X048WVS'] < 276012]
print(f"\nEast Germany: {len(east)} respondents")
print(f"West Germany: {len(west)} respondents")

# Also check EVS for S020 (year) - to confirm it's 1990
evs = pd.read_csv(EVS_PATH)
deu_evs = evs[evs['COUNTRY_ALPHA'] == 'DEU']
print(f"\nEVS Germany year(s): {sorted(deu_evs['S020'].unique())}")
print(f"EVS Germany rows: {len(deu_evs)}")
# EVS might not have region codes, but it likely covers both E and W Germany from 1990
# since EVS 1990 surveyed both just after reunification
