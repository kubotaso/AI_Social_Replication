#!/usr/bin/env python3
"""Debug script 2: check country-level patterns."""
import pandas as pd
import numpy as np
import csv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import compute_nation_level_factor_scores, clean_missing, get_latest_per_country

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
EVS_PATH = 'data/EVS_1990_wvs_format.csv'

with open(DATA_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]

usecols = ['S002VS','COUNTRY_ALPHA','S020','A032','A035','D022','D058','E019',
           'E015','D018','A124_02','A124_06','A124_07','F125','B003']
avail = [c for c in usecols if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2,3])]

# Also load EVS
if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    df = pd.concat([df, evs], ignore_index=True, sort=False)

df = get_latest_per_country(df)

val_cols = [c for c in avail if c not in ['S002VS','COUNTRY_ALPHA','S020']]
df = clean_missing(df, val_cols)

# Check D022 country means with value=1 and value=0
if 'D022' in df.columns:
    print("D022 (Child needs both parents):")
    temp = df[df['D022'].notna()]
    # Show a few countries
    for country in ['SWE','NGA','USA','JPN','RUS']:
        ct = temp[temp['COUNTRY_ALPHA']==country]
        if len(ct) > 0:
            mean_val = ct['D022'].mean()
            pct_1 = (ct['D022']==1).mean()
            pct_0 = (ct['D022']==0).mean()
            print(f"  {country}: mean={mean_val:.3f}, %=1: {pct_1:.3f}, %=0: {pct_0:.3f}, n={len(ct)}")
    print()

# Check D058 country means
if 'D058' in df.columns:
    print("D058 (University for boy):")
    temp = df[df['D058'].notna()]
    for country in ['NGA','PAK','IND','SWE','USA','JPN']:
        ct = temp[temp['COUNTRY_ALPHA']==country]
        if len(ct) > 0:
            pct_agree = (ct['D058']==1).mean()
            print(f"  {country}: %agree(1)={pct_agree:.3f}, mean={ct['D058'].mean():.3f}, n={len(ct)}")
    print()

# Check E019 country means
if 'E019' in df.columns:
    print("E019 (Technology emphasis):")
    temp = df[df['E019'].notna()]
    for country in ['NGA','USA','SWE','JPN','RUS']:
        ct = temp[temp['COUNTRY_ALPHA']==country]
        if len(ct) > 0:
            pct_1 = (ct['E019']==1).mean()
            print(f"  {country}: %=1(more emphasis): {pct_1:.3f}, n={len(ct)}")
    print()

# Check E015 country means
if 'E015' in df.columns:
    print("E015 (Science helps):")
    temp = df[df['E015'].notna()]
    for country in ['NGA','USA','SWE','JPN','RUS']:
        ct = temp[temp['COUNTRY_ALPHA']==country]
        if len(ct) > 0:
            pct_1 = (ct['E015']==1).mean()
            pct_3 = (ct['E015']==3).mean()
            print(f"  {country}: %=1: {pct_1:.3f}, %=3: {pct_3:.3f}, mean={ct['E015'].mean():.3f}, n={len(ct)}")
    print()

# Check D018
if 'D018' in df.columns:
    print("D018 (Woman needs children):")
    temp = df[df['D018'].notna()]
    for country in ['NGA','SWE','USA','JPN']:
        ct = temp[temp['COUNTRY_ALPHA']==country]
        if len(ct) > 0:
            pct_1 = (ct['D018']==1).mean()
            pct_0 = (ct['D018']==0).mean()
            print(f"  {country}: %=1: {pct_1:.3f}, %=0: {pct_0:.3f}, n={len(ct)}")
    print()

# Check A032 imagination
if 'A032' in df.columns:
    print("A032 (Imagination):")
    temp = df[df['A032'].notna()]
    for country in ['NGA','SWE','USA','JPN','RUS']:
        ct = temp[temp['COUNTRY_ALPHA']==country]
        if len(ct) > 0:
            pct_1 = (ct['A032']==1).mean()
            pct_0 = (ct['A032']==0).mean()
            print(f"  {country}: %mentioned(1): {pct_1:.3f}, %not(0): {pct_0:.3f}, n={len(ct)}")
    print()

# Check outgroup items
for oc in ['A124_02','A124_06','A124_07']:
    if oc in df.columns:
        print(f"{oc}:")
        temp = df[df[oc].notna()]
        for country in ['NGA','SWE','USA','JPN','RUS']:
            ct = temp[temp['COUNTRY_ALPHA']==country]
            if len(ct) > 0:
                pct_1 = (ct[oc]==1).mean()
                print(f"  {country}: %mentioned(1): {pct_1:.3f}, n={len(ct)}")
        print()

# F125 and B003 availability by country
for vc in ['F125','B003']:
    if vc in df.columns:
        temp = df[df[vc].notna()]
        countries = temp['COUNTRY_ALPHA'].unique()
        print(f"{vc}: available for {len(countries)} countries: {sorted(countries)}")
        print()
