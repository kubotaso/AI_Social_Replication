#!/usr/bin/env python3
"""Test married variable options for 1975."""
import pandas as pd, numpy as np

# Check what the full panel has for married in 1975
df_full = pd.read_csv('data/psid_panel_full.csv')
m75 = df_full[df_full['year'] == 1975]
print("Full panel 1975:")
print(f"  married values: {sorted(m75['married'].unique())}")
print(f"  married mean: {m75['married'].mean():.3f}")
print(f"  N: {len(m75)}")

# Check 1974 and 1976
m74 = df_full[df_full['year'] == 1974]
m76 = df_full[df_full['year'] == 1976]
print(f"  1974 married mean: {m74['married'].mean():.3f}")
print(f"  1976 married mean: {m76['married'].mean():.3f}")

# Check the raw 1975 family file for marital status
# 1975 build doesn't have 'marital_status' defined in the build script!
# Let's check what's available
print("\n=== Checking raw 1975 file for marital status ===")

# Read the .do file for 1975
import os
do_file = 'psid_raw/fam1975/FAM1975.do'
with open(do_file, 'r') as f:
    content = f.read()
lines = content.split('\n')
for line in lines:
    if 'marital' in line.lower() or 'married' in line.lower() or 'marri' in line.lower():
        print(f"  {line.strip()}")

# Now check what code the build script uses for married
# It creates 'married' from 'marital_status' using codes
# For years without explicit marital_status, it might be derived differently

# Check psid_panel.csv married in 1975
df = pd.read_csv('data/psid_panel.csv')
m75_p = df[df['year'] == 1975]
print(f"\nMain panel 1975:")
print(f"  married values: {sorted(m75_p['married'].unique())}")
print(f"  married mean: {m75_p['married'].mean():.3f}")

# The issue: in 1975, 'married' = 0.041 which is wrong.
# The raw marital_status is probably miscoded or missing.
# Let's check if there's a marital_status column in the full panel
print(f"\nFull panel columns: {list(df_full.columns)}")
