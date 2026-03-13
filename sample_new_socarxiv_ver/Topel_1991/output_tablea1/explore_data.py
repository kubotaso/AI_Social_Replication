#!/usr/bin/env python3
"""Explore data to understand issues for Table A1 replication."""
import pandas as pd
import numpy as np

# Check main panel
df = pd.read_csv('data/psid_panel.csv')
print("=== MAIN PANEL ===")
print(f"Shape: {df.shape}")
print(f"Years: {sorted(df['year'].unique())}")
print(f"Year counts:")
for yr, cnt in df['year'].value_counts().sort_index().items():
    print(f"  {yr}: {cnt}")

# Extract ER30002 (person number) from person_id
df['pn'] = df['person_id'] % 1000
print(f"\npn (ER30002) stats:")
print(f"  min: {df['pn'].min()}, max: {df['pn'].max()}")
print(f"  < 170: {(df['pn'] < 170).sum()} ({(df['pn'] < 170).mean():.3f})")

# Apply pn < 170 filter
df_filt = df[df['pn'] < 170]
print(f"\nAfter pn < 170: {len(df_filt)} obs")
print(f"Year counts after filter:")
for yr, cnt in df_filt['year'].value_counts().sort_index().items():
    print(f"  {yr}: {cnt}")

# Check SMSA in raw files
print("\n=== CHECKING RAW SMSA ===")
# 1977 has SMSA at cols 10-10 (1-based) = python [9:10]
for year in [1977, 1978, 1979, 1980]:
    filepath = f'psid_raw/fam{year}/FAM{year}.txt'
    try:
        with open(filepath, 'r') as f:
            smsa_vals = []
            for i, line in enumerate(f):
                if i >= 100: break
                smsa_val = line[9:10].strip()  # col 10 (1-based)
                smsa_vals.append(smsa_val)
            vc = pd.Series(smsa_vals).value_counts()
            print(f"  {year} SMSA (col 10): {dict(vc)}")
    except:
        print(f"  {year}: file not found")

# Check 1976 SMSA at cols 337-339
try:
    filepath = 'psid_raw/fam1976/FAM1976.txt'
    with open(filepath, 'r') as f:
        smsa_vals = []
        for i, line in enumerate(f):
            if i >= 100: break
            smsa_val = line[336:339].strip()
            smsa_vals.append(smsa_val)
        vc = pd.Series(smsa_vals).value_counts()
        print(f"  1976 SMSA (cols 337-339): {dict(vc)}")
except:
    print(f"  1976: file not found")

# Check full panel
print("\n=== FULL PANEL ===")
df2 = pd.read_csv('data/psid_panel_full.csv')
print(f"Shape: {df2.shape}")
print(f"Years: {sorted(df2['year'].unique())}")
print(f"SMSA unique: {df2['lives_in_smsa'].unique()}")

# Check married, disabled variables
for year in [1975]:
    sub = df[df['year'] == year]
    print(f"\n{year} married unique: {sorted(sub['married'].unique())}")
    print(f"{year} married mean: {sub['married'].mean():.3f}")
    print(f"{year} disabled unique: {sorted(sub['disabled'].dropna().unique())}")
    print(f"{year} disabled mean: {sub['disabled'].mean():.3f}")

# Check what tenure_topel looks like
print(f"\ntenure_topel stats: mean={df['tenure_topel'].mean():.3f}, sd={df['tenure_topel'].std():.3f}")
print(f"tenure_topel < 1: {(df['tenure_topel'] < 1).sum()}")

# Check education
print(f"\neducation_clean unique values by year:")
for yr in [1975, 1976, 1977]:
    sub = df[df['year'] == yr]
    print(f"  {yr}: {sorted(sub['education_clean'].dropna().unique())}")
