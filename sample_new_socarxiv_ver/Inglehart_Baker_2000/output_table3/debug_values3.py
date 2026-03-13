#!/usr/bin/env python3
"""Debug: check D058 coding more carefully and check what D018=1 means."""
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

# Get factor scores
scores_df, loadings_df, _ = compute_nation_level_factor_scores()
factor_scores = scores_df[['COUNTRY_ALPHA', 'surv_selfexp']].copy()
factor_scores['survival_score'] = -factor_scores['surv_selfexp']

# Check D058 more carefully - look at both agree percentages
usecols = ['S002VS','COUNTRY_ALPHA','S020','D058','D018','D022','E015','E019','C001','C011','B002']
avail = [c for c in usecols if c in header]
df = pd.read_csv(DATA_PATH, usecols=avail, low_memory=False)
df = df[df['S002VS'].isin([2,3])]
if os.path.exists(EVS_PATH):
    evs = pd.read_csv(EVS_PATH)
    df = pd.concat([df, evs], ignore_index=True, sort=False)
df = get_latest_per_country(df)
val_cols = [c for c in avail if c not in ['S002VS','COUNTRY_ALPHA','S020']]
df = clean_missing(df, val_cols)

# D058: Check if it's really 1=agree, 2=disagree or 1=agree strongly...4=disagree strongly
# Paper says "university education more important for boy than girl"
# For survival countries (Bangladesh, Pakistan, Nigeria), this should be HIGH agreement
# For self-expression countries (Sweden, Netherlands), this should be LOW agreement
print("D058 full distribution by country (survival -> self-expression):")
merged = pd.merge(df[['COUNTRY_ALPHA','D058']].dropna(), factor_scores, on='COUNTRY_ALPHA')
merged = merged.sort_values('survival_score', ascending=False)
for country in merged['COUNTRY_ALPHA'].unique()[:20]:
    ct = merged[merged['COUNTRY_ALPHA']==country]
    vc = ct['D058'].value_counts().sort_index()
    pct = vc / len(ct) * 100
    surv = ct['survival_score'].iloc[0]
    dist = ', '.join([f'{int(v)}:{p:.0f}%' for v,p in pct.items()])
    print(f"  {country:5s} (surv={surv:+.2f}): {dist}")

print()

# Check C001 coding
print("C001 distribution:")
print(df['C001'].value_counts().sort_index())
print()

# Check C011 coding
if 'C011' in df.columns:
    print("C011 distribution:")
    print(df['C011'].value_counts().sort_index())
    print()

# Check B002 coding
if 'B002' in df.columns:
    print("B002 distribution:")
    print(df['B002'].value_counts().sort_index())
    print()

# D018 - Check what values mean
print("D018 by country (top survival vs bottom):")
for country in ['AZE','MDA','BLR','NGA','SWE','NLD','NZL','AUS']:
    temp = df[(df['COUNTRY_ALPHA']==country) & (df['D018'].notna())]
    if len(temp) > 0:
        pct_1 = (temp['D018']==1).mean()
        pct_0 = (temp['D018']==0).mean()
        surv = factor_scores[factor_scores['COUNTRY_ALPHA']==country]['survival_score'].values
        s = surv[0] if len(surv) > 0 else 0
        print(f"  {country:5s} (surv={s:+.2f}): %=1: {pct_1:.3f}, %=0: {pct_0:.3f}")

print()

# E015 check - which value means "science helps"?
# Paper says survival correlates with "scientific discoveries will help, rather than harm"
# So higher survival countries should say science helps MORE (positive correlation)
print("E015 by country:")
for country in ['AZE','NGA','RUS','USA','SWE','NLD']:
    temp = df[(df['COUNTRY_ALPHA']==country) & (df['E015'].notna())]
    if len(temp) > 0:
        pct_1 = (temp['E015']==1).mean()
        pct_2 = (temp['E015']==2).mean()
        pct_3 = (temp['E015']==3).mean()
        surv = factor_scores[factor_scores['COUNTRY_ALPHA']==country]['survival_score'].values
        s = surv[0] if len(surv) > 0 else 0
        print(f"  {country:5s} (surv={s:+.2f}): 1={pct_1:.3f}, 2={pct_2:.3f}, 3={pct_3:.3f}")
