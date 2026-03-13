#!/usr/bin/env python3
"""
Check additional sample restrictions from the paper:
- Age 18-60 (Appendix, p.173)
- Exclude self-employed, agriculture, government
- Exclude poverty subsample (SRC random sample only)
- Positive earnings
- Current job tenure >= 1 year
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/psid_panel.csv')
print(f"Total: {len(df)} obs, {df['person_id'].nunique()} persons")

# Check age range
print(f"\nAge range: {df['age'].min()} - {df['age'].max()}")
print(f"Obs with age 18-60: {((df['age'] >= 18) & (df['age'] <= 60)).sum()}")
print(f"Obs outside 18-60: {((df['age'] < 18) | (df['age'] > 60)).sum()}")

# Check self-employed
print(f"\nSelf-employed: {df['self_employed'].sum()} obs")
print(f"Self-employed values: {df['self_employed'].value_counts().to_dict()}")

# Check agriculture
print(f"\nAgriculture: {df['agriculture'].sum()} obs")

# Check government
print(f"\nGovt worker: {df['govt_worker'].value_counts().to_dict()}")

# Check disabled
print(f"\nDisabled: {df['disabled'].value_counts().to_dict()}")

# id_1968 can tell us about SRC vs SEO
# SRC: id_1968 < 3000; SEO: id_1968 >= 5001
print(f"\nSRC sample (id_1968 < 3000): {(df['id_1968'] < 3000).sum()}")
print(f"SEO poverty sample (id_1968 >= 5001): {(df['id_1968'] >= 5001).sum()}")
print(f"Latino (id_1968 >= 7001): {(df['id_1968'] >= 7001).sum()}")
print(f"id_1968 range: {df['id_1968'].min()} - {df['id_1968'].max()}")

# How many unique persons in each sample?
src_persons = df[df['id_1968'] < 3000]['person_id'].nunique()
seo_persons = df[(df['id_1968'] >= 5001) & (df['id_1968'] < 7001)]['person_id'].nunique()
print(f"SRC persons: {src_persons}")
print(f"SEO persons: {seo_persons}")

# Apply ALL paper restrictions
# 1. White males (already restricted in panel)
# 2. Age 18-60
# 3. Not self-employed, agriculture, or government
# 4. Positive earnings
# 5. SRC random sample (not poverty subsample)
# 6. Job tenure >= 1 year (already restricted in panel via tenure_topel >= 1)

df_r = df.copy()
n0 = len(df_r)

# Age 18-60
df_r = df_r[(df_r['age'] >= 18) & (df_r['age'] <= 60)]
print(f"\nAfter age 18-60: {len(df_r)} (dropped {n0 - len(df_r)})")

# Not self-employed
n0 = len(df_r)
df_r = df_r[df_r['self_employed'] != 1]
print(f"After not self-employed: {len(df_r)} (dropped {n0 - len(df_r)})")

# Not agriculture
n0 = len(df_r)
df_r = df_r[df_r['agriculture'] != 1]
print(f"After not agriculture: {len(df_r)} (dropped {n0 - len(df_r)})")

# Not government
n0 = len(df_r)
if 'govt_worker' in df_r.columns:
    df_r = df_r[df_r['govt_worker'] != 1]
print(f"After not government: {len(df_r)} (dropped {n0 - len(df_r)})")

# SRC random sample only (exclude poverty/SEO subsample)
n0 = len(df_r)
df_r = df_r[df_r['id_1968'] < 3000]
print(f"After SRC only: {len(df_r)} (dropped {n0 - len(df_r)})")

# Positive earnings
n0 = len(df_r)
df_r = df_r[df_r['labor_inc'] > 0]
print(f"After positive earnings: {len(df_r)} (dropped {n0 - len(df_r)})")

print(f"\nFinal: {len(df_r)} obs, {df_r['person_id'].nunique()} persons")
print(f"Paper: 13,128 job-years on 1,540 individuals")

# This gives us the LEVEL data count
# The paper says 13,128 job-years - let's see if our restricted sample is close
