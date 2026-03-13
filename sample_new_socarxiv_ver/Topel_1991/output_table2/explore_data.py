#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os

os.chdir('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3')

df = pd.read_csv('data/psid_panel.csv')
print('=== BASIC INFO ===')
print(f'Shape: {df.shape}')
print(f'Unique persons: {df.person_id.nunique()}')
print(f'Years: {sorted(df.year.unique())}')
print()

print('=== KEY VARIABLE STATS ===')
for col in ['age', 'education_clean', 'experience', 'tenure_topel', 'hourly_wage', 'log_hourly_wage', 'govt_worker', 'white', 'sex', 'self_employed', 'lives_in_smsa', 'union_member', 'disabled', 'hours', 'labor_inc']:
    if col in df.columns:
        print(f'{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.3f}, missing={df[col].isna().sum()}, zeros={(df[col]==0).sum()}')
print()

print('=== GOVT_WORKER by year ===')
for yr in sorted(df.year.unique()):
    sub = df[df.year==yr]
    nonmissing = sub.govt_worker.notna().sum()
    govt1 = (sub.govt_worker==1).sum()
    print(f'  {yr}: N={len(sub)}, govt_worker nonmissing={nonmissing}, govt=1: {govt1}')
print()

print('=== self_employed distribution ===')
print(df.self_employed.value_counts(dropna=False))
print()

print('=== tenure_topel stats ===')
print(df.tenure_topel.describe())
print()

print('=== hourly_wage quantiles ===')
print(df.hourly_wage.describe(percentiles=[.01,.05,.10,.25,.50,.75,.90,.95,.99]))
print()

print('=== d_log_wage stats ===')
print(df.d_log_wage.describe())
print()

print('=== Experience distribution ===')
print(df.experience.describe())
print()

print('=== Obs per year ===')
for yr in sorted(df.year.unique()):
    print(f'  {yr}: {len(df[df.year==yr])}')
