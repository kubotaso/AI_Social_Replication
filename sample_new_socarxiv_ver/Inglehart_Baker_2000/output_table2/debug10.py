#!/usr/bin/env python3
"""Deep investigation of problematic items."""
import pandas as pd
import numpy as np
import csv
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE, "data/WVS_Time_Series_1981-2022_csv_v5_0.csv")
EVS_PATH = os.path.join(BASE, "data/EVS_1990_wvs_format.csv")

# 1. Check F050 in detail - is it really "comfort from religion"?
# In WVS Time Series V5:
# F050 might be "Do you get comfort and strength from religion?" 1=Yes, 0=No
# But NGA=1.00 (all say yes) and SWE=0.56 is suspiciously high for Sweden

# 2. Check F059 as alternative
# F059 might be another religion variable

# 3. Check the EVS data F050
cols = ['S002VS', 'COUNTRY_ALPHA', 'F050', 'F059', 'F028', 'F054', 'F053',
        'F055', 'F064', 'F029']

with open(DATA_PATH, 'r') as f:
    reader = csv.reader(f)
    header = [h.strip('"') for h in next(reader)]

avail = ['S002VS', 'COUNTRY_ALPHA'] + [v for v in cols[2:] if v in header]
df_wvs = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df_wvs = df_wvs[df_wvs['S002VS'].isin([2, 3])]

# Get latest per country
latest = df_wvs.groupby('COUNTRY_ALPHA')['S002VS'].max().reset_index()
latest.columns = ['COUNTRY_ALPHA', 'lw']
df_wvs = df_wvs.merge(latest, on='COUNTRY_ALPHA')
df_wvs = df_wvs[df_wvs['S002VS'] == df_wvs['lw']].drop('lw', axis=1)

print("=== F050 country means (WVS only, latest wave per country) ===")
if 'F050' in df_wvs.columns:
    df_wvs['F050n'] = pd.to_numeric(df_wvs['F050'], errors='coerce')
    df_wvs.loc[df_wvs['F050n'] < 0, 'F050n'] = np.nan
    f050_means = df_wvs.groupby('COUNTRY_ALPHA')['F050n'].mean().dropna()
    print("Countries with data:", len(f050_means))
    for c in sorted(f050_means.index):
        print(f"  {c}: {f050_means[c]:.3f}")

# Check F059 as "prays"
print("\n=== F059 (pray?) country means ===")
if 'F059' in df_wvs.columns:
    df_wvs['F059n'] = pd.to_numeric(df_wvs['F059'], errors='coerce')
    df_wvs.loc[df_wvs['F059n'] < 0, 'F059n'] = np.nan
    f059_means = df_wvs.groupby('COUNTRY_ALPHA')['F059n'].mean().dropna()
    print("Countries with data:", len(f059_means))
    for c in ['NGA', 'SWE', 'JPN', 'USA', 'CHN', 'DEU']:
        if c in f059_means.index:
            print(f"  {c}: {f059_means[c]:.3f}")

# Church attendance: F028 check different thresholds
print("\n=== F028 church attendance country means ===")
if 'F028' in df_wvs.columns:
    df_wvs['F028n'] = pd.to_numeric(df_wvs['F028'], errors='coerce')
    df_wvs.loc[df_wvs['F028n'] < 0, 'F028n'] = np.nan
    f028_pct_monthly = df_wvs.groupby('COUNTRY_ALPHA').apply(
        lambda g: (g['F028n'] <= 3).mean() if g['F028n'].notna().sum() > 0 else np.nan
    ).dropna()
    f028_pct_weekly = df_wvs.groupby('COUNTRY_ALPHA').apply(
        lambda g: (g['F028n'] <= 2).mean() if g['F028n'].notna().sum() > 0 else np.nan
    ).dropna()
    f028_mean = df_wvs.groupby('COUNTRY_ALPHA')['F028n'].mean().dropna()
    print(f"Countries with data: {len(f028_mean)}")
    for c in ['NGA', 'SWE', 'JPN', 'USA', 'POL', 'CHN', 'BRA', 'IND']:
        if c in f028_mean.index:
            m = f028_mean.get(c, np.nan)
            pm = f028_pct_monthly.get(c, np.nan)
            pw = f028_pct_weekly.get(c, np.nan)
            print(f"  {c}: mean={m:.2f}, %monthly={pm:.3f}, %weekly={pw:.3f}")

# EVS check
print("\n=== EVS F050 check ===")
if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    if 'F050' in evs.columns:
        evs['F050n'] = pd.to_numeric(evs['F050'], errors='coerce')
        evs.loc[evs['F050n'] < 0, 'F050n'] = np.nan
        evs_f050 = evs.groupby('COUNTRY_ALPHA')['F050n'].mean().dropna()
        for c in sorted(evs_f050.index):
            print(f"  {c}: {evs_f050[c]:.3f}")
