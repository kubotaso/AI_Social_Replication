#!/usr/bin/env python3
"""Explore tenure reconstruction and full panel."""
import pandas as pd
import numpy as np

# Check the full panel for more years
df_full = pd.read_csv('data/psid_panel_full.csv')
print("=== FULL PANEL ===")
print(f"Shape: {df_full.shape}")
print(f"Columns: {sorted(df_full.columns.tolist())}")
print(f"Year distribution:")
print(df_full['year'].value_counts().sort_index())
print(f"Persons: {df_full['person_id'].nunique()}")

# Check if the full panel has the necessary variables
for col in ['hourly_wage', 'log_hourly_wage', 'hours', 'labor_inc', 'wages',
            'tenure_topel', 'tenure', 'tenure_mos', 'job_id', 'education_clean',
            'age', 'experience']:
    if col in df_full.columns:
        print(f"  {col}: non_null={df_full[col].notna().sum()}, mean={df_full[col].dropna().mean():.2f}")
    else:
        print(f"  {col}: NOT IN FULL PANEL")

# Main panel
df = pd.read_csv('data/psid_panel.csv')

# Try using tenure_mos to reconstruct proper starting tenure
# For years where tenure_mos is available, convert to years
print("\n=== TENURE_MOS DETAILS ===")
for y in sorted(df['year'].unique()):
    sub = df[df['year'] == y]
    tm = sub['tenure_mos'].dropna()
    # Filter out 999 (missing/NA)
    tm_valid = tm[tm < 900]
    if len(tm_valid) > 0:
        print(f"  {y}: n_valid={len(tm_valid)}, mean_months={tm_valid.mean():.1f}, mean_yrs={tm_valid.mean()/12:.1f}")
    else:
        print(f"  {y}: no valid tenure_mos data")

# For the 'tenure' column (raw from PSID)
print("\n=== RAW TENURE COLUMN ===")
for y in sorted(df['year'].unique()):
    sub = df[df['year'] == y]
    t = sub['tenure'].dropna()
    t_valid = t[t < 900]
    if len(t_valid) > 0:
        print(f"  {y}: n_valid={len(t_valid)}, mean={t_valid.mean():.1f}, max={t_valid.max():.0f}")
    else:
        print(f"  {y}: no valid tenure data")

# Let's also check the full panel
print("\n=== FULL PANEL TENURE ===")
for col in ['tenure_topel', 'tenure', 'tenure_mos']:
    if col in df_full.columns:
        for y in sorted(df_full['year'].unique()):
            sub = df_full[df_full['year'] == y]
            t = sub[col].dropna()
            t_valid = t[t < 900]
            if len(t_valid) > 0 and len(t_valid) > 100:
                print(f"  {col} {y}: n={len(t_valid)}, mean={t_valid.mean():.1f}")

# Use the REPORTED tenure to initialize tenure for jobs that were in progress
# For each person's first observation, if they have a reported tenure, use it
print("\n=== USING REPORTED TENURE TO RECONSTRUCT ===")
df_sorted = df.sort_values(['person_id', 'job_id', 'year']).copy()

# For each job spell, get the first observation's reported tenure
first_obs = df_sorted.groupby(['person_id', 'job_id']).first().reset_index()

# Where reported tenure (in months) is available and valid
mask = (first_obs['tenure_mos'].notna()) & (first_obs['tenure_mos'] < 900)
print(f"Job spells with valid first-obs tenure_mos: {mask.sum()} out of {len(first_obs)}")

# Also check raw tenure
mask2 = (first_obs['tenure'].notna()) & (first_obs['tenure'] < 900)
print(f"Job spells with valid first-obs raw tenure: {mask2.sum()} out of {len(first_obs)}")

# What about using lag_tenure?
if 'lag_tenure' in df.columns:
    mask3 = first_obs['lag_tenure'].notna()
    print(f"Job spells with valid first-obs lag_tenure: {mask3.sum()} out of {len(first_obs)}")
