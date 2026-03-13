#!/usr/bin/env python3
"""Explore data for East/West Germany split and Pakistan availability."""
import pandas as pd
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

# Read WVS header
with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

region_cols = [c for c in header if c.startswith('X048') or c.startswith('S014')]
print('Region-like columns:', region_cols)

# Check Germany across waves
df = pd.read_csv(DATA_PATH, usecols=['S002VS','COUNTRY_ALPHA','S024'], low_memory=False)
deu = df[df['COUNTRY_ALPHA'] == 'DEU']
for wave in sorted(deu['S002VS'].unique()):
    sub = deu[deu['S002VS'] == wave]
    print(f"Wave {wave}: {len(sub)} rows, S024={sorted(sub['S024'].unique())}")

# Check Pakistan
pak = df[df['COUNTRY_ALPHA'] == 'PAK']
for wave in sorted(pak['S002VS'].unique()):
    sub = pak[pak['S002VS'] == wave]
    print(f"Pakistan Wave {wave}: {len(sub)} rows")

# Check EVS
evs = pd.read_csv(EVS_PATH)
print(f"\nEVS Germany rows: {len(evs[evs['COUNTRY_ALPHA']=='DEU'])}")
print(f"EVS columns: {list(evs.columns)}")

# Check if X048WVS exists for region splitting
if 'X048WVS' in header:
    deu_r = pd.read_csv(DATA_PATH, usecols=['S002VS','COUNTRY_ALPHA','S024','X048WVS'], low_memory=False)
    deu_r = deu_r[(deu_r['COUNTRY_ALPHA']=='DEU') & (deu_r['S002VS'].isin([2,3]))]
    print(f"\nX048WVS for DEU waves 2-3: {sorted(deu_r['X048WVS'].dropna().unique())}")
