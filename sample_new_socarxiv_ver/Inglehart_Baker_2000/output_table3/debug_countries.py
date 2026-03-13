#!/usr/bin/env python3
"""Check which countries are missing from item data vs factor scores."""
import pandas as pd
import numpy as np
import csv
import sys, os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE_DIR, "data/EVS_1990_wvs_format.csv")

with open(DATA_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]

# Load minimal data to see which countries have which items
needed = ['S002VS','COUNTRY_ALPHA','S020','D059','D022','D058','E019','B002','A124_07']
avail = [c for c in needed if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2,3])]

if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    df = pd.concat([df, evs], ignore_index=True, sort=False)

# Latest per country
if 'S020' in df.columns:
    latest = df.groupby('COUNTRY_ALPHA')['S020'].max().reset_index()
    latest.columns = ['COUNTRY_ALPHA', 'latest_year']
    df = df.merge(latest, on='COUNTRY_ALPHA')
    df = df[df['S020'] == df['latest_year']].drop('latest_year', axis=1)

# Clean
for c in ['D059','D022','D058','E019','B002','A124_07']:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].where(df[c] >= 0, np.nan)

all_countries = sorted(df['COUNTRY_ALPHA'].unique())
print(f"Total countries in data: {len(all_countries)}")
print(f"Countries: {all_countries}")

# For each item, show which countries have valid data
for c in ['D059','D022','D058','E019','B002','A124_07']:
    if c in df.columns:
        valid = df[df[c].notna()].groupby('COUNTRY_ALPHA')[c].count()
        valid_countries = valid[valid >= 30].index.tolist()
        missing = [x for x in all_countries if x not in valid_countries]
        print(f"\n{c}: {len(valid_countries)} countries with data, missing: {missing}")

# Check which countries are only in wave 2 vs wave 3
print("\n\nCountry wave availability:")
df_all = pd.read_csv(DATA_PATH, usecols=['S002VS','COUNTRY_ALPHA'], low_memory=False)
df_all = df_all[df_all['S002VS'].isin([2,3])]
for country in sorted(all_countries):
    waves = sorted(df_all[df_all['COUNTRY_ALPHA']==country]['S002VS'].unique())
    if len(waves) == 1:
        print(f"  {country}: wave {waves[0]} only")
