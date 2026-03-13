#!/usr/bin/env python3
"""Debug: Check D058 coding definitively, plus D022 and B002 patterns."""
import pandas as pd
import numpy as np
from scipy import stats
import csv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_factor_analysis import compute_nation_level_factor_scores, clean_missing, get_latest_per_country

DATA_PATH = 'data/WVS_Time_Series_1981-2022_csv_v5_0.csv'
EVS_PATH = 'data/EVS_1990_wvs_format.csv'

scores_df, _, _ = compute_nation_level_factor_scores()
fs = scores_df[['COUNTRY_ALPHA','surv_selfexp']].copy()
fs['survival'] = -fs['surv_selfexp']

with open(DATA_PATH, 'r') as f:
    header = [h.strip('"') for h in next(csv.reader(f))]

usecols = ['S002VS','COUNTRY_ALPHA','S020','D058','D022','B002','E019','E015',
           'A124_02','A124_06','A124_07','C001','C011']
avail = [c for c in usecols if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2,3])]
if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    df = pd.concat([df, evs], ignore_index=True, sort=False)
df = get_latest_per_country(df)
val_cols = [c for c in avail if c not in ['S002VS','COUNTRY_ALPHA','S020']]
df = clean_missing(df, val_cols)

# D058: Try different codings and see which gives best correlation
print("=== D058 (University more for boy) ===")
if 'D058' in df.columns:
    temp = df[df['D058'].notna()]
    # Try different approaches
    approaches = {
        'pct_val1': lambda x: (x == 1).astype(float),
        'pct_val2': lambda x: (x == 2).astype(float),
        'pct_val34': lambda x: (x >= 3).astype(float),
        'pct_val12': lambda x: (x <= 2).astype(float),
        'mean': lambda x: x,
        'mean_rev': lambda x: 5 - x,
    }
    for name, func in approaches.items():
        vals = temp.copy()
        vals['v'] = func(vals['D058'])
        cm = vals.groupby('COUNTRY_ALPHA')['v'].mean()
        merged = pd.merge(fs, cm.reset_index().rename(columns={'v':'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5 and merged['item'].std() > 1e-10:
            r, p = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  {name:15s}: r={r:.3f}, p={p:.4f}, n={len(merged)}")

# D022: Try different codings
print("\n=== D022 (Child needs both parents) ===")
if 'D022' in df.columns:
    temp = df[df['D022'].notna()]
    approaches = {
        'pct_val0': lambda x: (x == 0).astype(float),
        'pct_val1': lambda x: (x == 1).astype(float),
        'mean': lambda x: x,
        'mean_rev': lambda x: 1 - x,
    }
    for name, func in approaches.items():
        vals = temp.copy()
        vals['v'] = func(vals['D022'])
        cm = vals.groupby('COUNTRY_ALPHA')['v'].mean()
        merged = pd.merge(fs, cm.reset_index().rename(columns={'v':'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5 and merged['item'].std() > 1e-10:
            r, p = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  {name:15s}: r={r:.3f}, p={p:.4f}, n={len(merged)}")

# B002: Try different codings
print("\n=== B002 (Environmental meeting) ===")
if 'B002' in df.columns:
    temp = df[df['B002'].notna()]
    approaches = {
        'pct_not1': lambda x: (x != 1).astype(float),
        'pct_3or4': lambda x: (x >= 3).astype(float),
        'mean': lambda x: x,
        'pct_34': lambda x: (x >= 3).astype(float),
    }
    for name, func in approaches.items():
        vals = temp.copy()
        vals['v'] = func(vals['B002'])
        cm = vals.groupby('COUNTRY_ALPHA')['v'].mean()
        merged = pd.merge(fs, cm.reset_index().rename(columns={'v':'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5 and merged['item'].std() > 1e-10:
            r, p = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  {name:15s}: r={r:.3f}, p={p:.4f}, n={len(merged)}")

# E019: Try different codings
print("\n=== E019 (Technology emphasis) ===")
if 'E019' in df.columns:
    temp = df[df['E019'].notna()]
    approaches = {
        'pct_val1': lambda x: (x == 1).astype(float),
        'mean': lambda x: x,
        'mean_rev': lambda x: 4 - x,
    }
    for name, func in approaches.items():
        vals = temp.copy()
        vals['v'] = func(vals['E019'])
        cm = vals.groupby('COUNTRY_ALPHA')['v'].mean()
        merged = pd.merge(fs, cm.reset_index().rename(columns={'v':'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5 and merged['item'].std() > 1e-10:
            r, p = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  {name:15s}: r={r:.3f}, p={p:.4f}, n={len(merged)}")

# E015: Try different codings
print("\n=== E015 (Science helps) ===")
if 'E015' in df.columns:
    temp = df[df['E015'].notna()]
    approaches = {
        'pct_val1': lambda x: (x == 1).astype(float),
        'pct_val3': lambda x: (x == 3).astype(float),
        'mean': lambda x: x,
        'mean_rev': lambda x: 4 - x,
    }
    for name, func in approaches.items():
        vals = temp.copy()
        vals['v'] = func(vals['E015'])
        cm = vals.groupby('COUNTRY_ALPHA')['v'].mean()
        merged = pd.merge(fs, cm.reset_index().rename(columns={'v':'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5 and merged['item'].std() > 1e-10:
            r, p = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  {name:15s}: r={r:.3f}, p={p:.4f}, n={len(merged)}")

# Outgroup: Try individual items and combined
print("\n=== Outgroup items ===")
for oc in ['A124_02','A124_06','A124_07']:
    if oc in df.columns:
        temp = df[df[oc].notna()]
        cm = temp.groupby('COUNTRY_ALPHA')[oc].mean()
        merged = pd.merge(fs, cm.reset_index().rename(columns={oc:'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5:
            r, p = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  {oc}: r={r:.3f}, n={len(merged)}")

# Try outgroup as composite
avail_out = [c for c in ['A124_02','A124_06','A124_07'] if c in df.columns]
temp = df.copy()
temp['outgroup'] = temp[avail_out].mean(axis=1)
cm = temp.groupby('COUNTRY_ALPHA')['outgroup'].mean()
merged = pd.merge(fs, cm.reset_index().rename(columns={'outgroup':'item'}), on='COUNTRY_ALPHA')
merged = merged.dropna()
if len(merged) >= 5:
    r, p = stats.pearsonr(merged['survival'], merged['item'])
    print(f"  composite(mean): r={r:.3f}, n={len(merged)}")

# Try A124_07 (AIDS) as main driver
# Paper says: "foreigners, homosexuals, and people with AIDS"
# A124_06 = immigrants/foreign workers, A124_07 = people who have AIDS
# But also need "homosexuals" - check A124_03 ?
print("\n=== Check A124 item labels ===")
for c in [f'A124_{i:02d}' for i in range(1, 20)]:
    if c in header:
        temp2 = df[c].dropna() if c in df.columns else pd.Series()
        if len(temp2) > 0:
            cm2 = df.groupby('COUNTRY_ALPHA')[c].mean()
            merged2 = pd.merge(fs, cm2.reset_index().rename(columns={c:'item'}), on='COUNTRY_ALPHA')
            merged2 = merged2.dropna()
            if len(merged2) >= 5 and merged2['item'].std() > 1e-10:
                r2, _ = stats.pearsonr(merged2['survival'], merged2['item'])
                print(f"  {c}: r={r2:.3f}, n={len(merged2)}")

# C001/C011 job items
print("\n=== Job motivation items ===")
if 'C001' in df.columns:
    temp = df[df['C001'].notna()]
    # C001 appears to be: 1=?, 2=?, 3=?
    for v in [1, 2, 3]:
        cm = temp.groupby('COUNTRY_ALPHA').apply(lambda g: (g['C001']==v).mean())
        merged = pd.merge(fs, cm.reset_index().rename(columns={0:'item'}), on='COUNTRY_ALPHA')
        merged = merged.dropna()
        if len(merged) >= 5 and merged['item'].std() > 1e-10:
            r, _ = stats.pearsonr(merged['survival'], merged['item'])
            print(f"  C001 %={v}: r={r:.3f}, n={len(merged)}")

if 'C011' in df.columns:
    temp = df[df['C011'].notna()]
    # C011: 0=not mentioned, 1=mentioned
    cm = temp.groupby('COUNTRY_ALPHA')['C011'].mean()
    merged = pd.merge(fs, cm.reset_index().rename(columns={'C011':'item'}), on='COUNTRY_ALPHA')
    merged = merged.dropna()
    if len(merged) >= 5:
        r, _ = stats.pearsonr(merged['survival'], merged['item'])
        print(f"  C011 mean: r={r:.3f}, n={len(merged)}")

from scipy import stats
