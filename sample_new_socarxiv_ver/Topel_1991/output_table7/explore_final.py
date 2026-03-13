#!/usr/bin/env python3
"""Final exploration: try 3-region vs 4-region dummies, different missing data."""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/Users/administrator/Library/CloudStorage/Dropbox/lib/AI_WVS/Replication_Claude/Replication_Claude_Topel_v3/data/psid_panel.csv')

# Check what region columns exist
print("Region columns:", [c for c in df.columns if 'region' in c.lower()])
print("Region value counts:")
for c in [c for c in df.columns if c.startswith('region_')]:
    print(f"  {c}: sum={df[c].sum()}, mean={df[c].mean():.3f}")

# Check what year dummies exist
yr_cols_all = [c for c in df.columns if c.startswith('year_')]
print(f"\nYear columns: {yr_cols_all}")

# How many unique persons and jobs?
print(f"\nUnique persons: {df['person_id'].nunique()}")
print(f"Unique jobs: {df['job_id'].nunique()}")

# Distribution of tenure_topel
print(f"\nTenure_topel stats:")
print(df['tenure_topel'].describe())

# Distribution of ct_obs
df['ct_obs'] = df.groupby('job_id')['tenure_topel'].transform('max')
print(f"\nCompleted tenure (ct_obs) stats:")
print(df['ct_obs'].describe())
print(f"ct_obs value counts (top 15):")
print(df.groupby('job_id')['ct_obs'].first().value_counts().head(15))

# What fraction of jobs are censored?
df['censor'] = (df.groupby('job_id')['year'].transform('max') >= 1983).astype(float)
job_censor = df.groupby('job_id')['censor'].first()
print(f"\nCensored jobs: {job_censor.sum()}/{len(job_censor)} ({100*job_censor.mean():.1f}%)")

# Check if there are any multi-year gaps in the panel
df_s = df.sort_values(['person_id', 'year'])
df_s['year_diff'] = df_s.groupby('person_id')['year'].diff()
print(f"\nYear gaps:")
print(df_s['year_diff'].value_counts().head())

# Is there a variable for starting tenure / pre-panel tenure?
print(f"\nColumns that might indicate pre-panel tenure:")
for c in df.columns:
    if 'tenure' in c.lower() or 'start' in c.lower() or 'begin' in c.lower():
        print(f"  {c}")

# Try: what if we use 4 region dummies (NE, NC, South, West) with West as reference?
# Currently using 3 (NE, NC, South) with West as omitted
# This shouldn't change anything since it's the same specification

# Key insight: The paper's Table 4 note says "8 census regions"
# We only have 4 (NE, NC, South, West). Maybe we should have 8?
print("\n=== Region analysis ===")
if 'region' in df.columns:
    print(f"Region variable values: {df['region'].unique()}")
    print(df['region'].value_counts())
