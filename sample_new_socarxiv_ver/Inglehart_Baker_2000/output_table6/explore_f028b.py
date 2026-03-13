"""
Check F028B and COUNTRY_ALPHA for potential 1981 European coverage
"""
import pandas as pd
import numpy as np
import os

base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wvs_path = os.path.join(base, "data", "WVS_Time_Series_1981-2022_csv_v5_0.csv")

wvs = pd.read_csv(wvs_path, usecols=['S002VS', 'S003', 'COUNTRY_ALPHA', 'S020', 'F028', 'F028B', 'S017'],
                   low_memory=False)

print("=== F028B analysis ===")
print("F028B unique values:", sorted(wvs['F028B'].dropna().unique()))
print("F028B non-null count:", wvs['F028B'].notna().sum())
print()

# Check F028B by wave
for wave in [1, 2, 3]:
    wdata = wvs[wvs['S002VS'] == wave]
    f028b_nonnull = wdata['F028B'].notna().sum()
    print(f"Wave {wave}: F028B non-null = {f028b_nonnull}")
    if f028b_nonnull > 0:
        # Which countries have F028B?
        ctries = wdata[wdata['F028B'].notna()]['COUNTRY_ALPHA'].value_counts()
        print(f"  Countries with F028B: {dict(ctries)}")
print()

# Check COUNTRY_ALPHA for wave 1
print("=== COUNTRY_ALPHA in wave 1 ===")
w1 = wvs[wvs['S002VS'] == 1]
print("Countries (COUNTRY_ALPHA):", sorted(w1['COUNTRY_ALPHA'].dropna().unique()))

# Check if there are European countries in wave 1 via COUNTRY_ALPHA
european_alpha = ['BEL', 'CAN', 'FRA', 'GBR', 'ISL', 'IRL', 'ITA', 'NLD', 'NOR', 'ESP', 'SWE', 'DEU', 'USA']
for alpha in european_alpha:
    cnt = len(w1[w1['COUNTRY_ALPHA'] == alpha])
    if cnt > 0:
        print(f"  {alpha} in wave 1: {cnt} rows!")
    else:
        print(f"  {alpha} in wave 1: 0 rows")

# Also check for "EVS" markers
print("\n=== S004 or other version indicators ===")
w1_cols = ['S002VS', 'S003', 'COUNTRY_ALPHA', 'S004', 'S006', 'S020']
available = [c for c in w1_cols if c in wvs.columns]
print("Available cols:", available)
if 'S004' in wvs.columns:
    print("S004 in wave 1:", wvs[wvs['S002VS'] == 1]['S004'].value_counts().head())
